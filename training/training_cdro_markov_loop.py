import copy
import json
import os
import pickle
import time

import dnnlib
import numpy as np
import psutil
import torch

from torch_utils import distributed as dist
from torch_utils import misc
from torch_utils import training_stats
from training.markov_utils import (
    build_image_markov_schedule,
    compute_control_cost,
    initialize_terminal_stats,
    local_denoise_loss,
    rollout_markov_forward_images,
    sample_markov_reverse_images,
    update_terminal_stats,
)


def _activation_scale(cur_nimg, *, start_kimg, ramp_kimg):
    current_kimg = float(cur_nimg) / 1000.0
    start_kimg = max(float(start_kimg), 0.0)
    ramp_kimg = max(float(ramp_kimg), 0.0)
    if current_kimg < start_kimg:
        return 0.0
    if ramp_kimg <= 0.0:
        return 1.0
    return min(max((current_kimg - start_kimg) / ramp_kimg, 0.0), 1.0)


def _set_grad(module, enabled):
    for p in module.parameters():
        p.requires_grad_(enabled)


@torch.no_grad()
def _update_ema(ema, net, decay):
    for p_ema, p_net in zip(ema.parameters(), net.parameters()):
        p_ema.copy_(p_net.detach().lerp(p_ema, decay))
    for b_ema, b_net in zip(ema.buffers(), net.buffers()):
        b_ema.copy_(b_net)


def training_loop(
    run_dir='.',
    dataset_kwargs={},
    data_loader_kwargs={},
    network_kwargs={},
    loss_kwargs={},
    optimizer_kwargs={},
    augment_kwargs=None,
    seed=0,
    batch_size=64,
    batch_gpu=None,
    total_kimg=200,
    ema_halflife_kimg=500,
    ema_rampup_ratio=0.05,
    lr_rampup_kimg=10000,
    loss_scaling=1,
    kimg_per_tick=25,
    snapshot_ticks=25,
    state_dump_ticks=25,
    resume_pkl=None,
    resume_state_dump=None,
    resume_kimg=0,
    cudnn_benchmark=True,
    device=torch.device('cuda'),
    markov_num_steps=8,
    markov_total_time=1.0,
    markov_beta_min=0.1,
    markov_beta_max=12.0,
    markov_sde_family='vp_cosine',
    markov_weight_schedule='uniform',
    markov_control_lr=2e-4,
    markov_lambda_min=0.0,
    markov_reverse_control_scale=1.0,
    markov_reverse_noise_scale=1.0,
    markov_terminal_momentum=0.95,
):
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
    net.train().requires_grad_(True).to(device)
    if not hasattr(net, 'control'):
        raise RuntimeError('Full Markov CDRO requires a network with a control() method.')
    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device)
            misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

    score_optimizer = dnnlib.util.construct_class_by_name(params=net.model.parameters(), **optimizer_kwargs)
    control_optimizer = dnnlib.util.construct_class_by_name(
        params=net.control_model.parameters(),
        class_name=optimizer_kwargs["class_name"],
        lr=markov_control_lr,
        betas=optimizer_kwargs.get("betas", [0.9, 0.999]),
        eps=optimizer_kwargs.get("eps", 1e-8),
    )
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None
    ddp = torch.nn.parallel.DistributedDataParallel(
        net,
        device_ids=[device],
        find_unused_parameters=True,
    )
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier()
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier()
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data

    lambda_dual = float(loss_kwargs.get('lambda_init', 1e-3))
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'), weights_only=False)
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        score_optimizer.load_state_dict(data['score_optimizer_state'])
        control_optimizer.load_state_dict(data['control_optimizer_state'])
        lambda_dual = float(data.get('lambda_dual', lambda_dual))
        del data

    schedule = build_image_markov_schedule(
        num_steps=markov_num_steps,
        total_time=markov_total_time,
        beta_min=markov_beta_min,
        beta_max=markov_beta_max,
        sde_family=markov_sde_family,
        device=device,
        weight_schedule=markov_weight_schedule,
    )
    terminal_stats = None

    dist.print0(f'Training Markov CDRO for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    stats_jsonl = None

    while True:
        activation_scale = _activation_scale(
            cur_nimg,
            start_kimg=loss_kwargs.get('start_kimg', 0.0),
            ramp_kimg=loss_kwargs.get('ramp_kimg', 0.0),
        )
        control_scale = activation_scale * float(loss_kwargs.get('max_delta', 0.03))
        score_optimizer.zero_grad(set_to_none=True)
        score_losses = []
        control_costs = []

        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = images.to(device).to(torch.float32) / 127.5 - 1
                labels = labels.to(device)
                clean, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)

                if activation_scale > 0:
                    _set_grad(net.model, False)
                    _set_grad(net.control_model, True)
                    control_optimizer.zero_grad(set_to_none=True)
                    rollout = rollout_markov_forward_images(
                        clean_images=clean,
                        net=ddp.module,
                        schedule=schedule,
                        class_labels=labels,
                        augment_labels=augment_labels,
                        zero_control=False,
                        control_scale=control_scale,
                    )
                    score_loss_control = local_denoise_loss(
                        net=ddp,
                        rollout=rollout,
                        schedule=schedule,
                        sigma_data=net.sigma_data,
                        class_labels=labels,
                        augment_labels=augment_labels,
                    )
                    control_cost = compute_control_cost(rollout=rollout, schedule=schedule)
                    objective = score_loss_control - lambda_dual * control_cost
                    (-objective).backward()
                    control_optimizer.step()
                    lambda_dual = max(
                        float(markov_lambda_min),
                        lambda_dual - float(loss_kwargs.get('lambda_lr', 5e-4)) * (float(loss_kwargs.get('rho_target', 1e-4)) - float(control_cost.detach().item())),
                    )
                    _set_grad(net.model, True)
                    _set_grad(net.control_model, False)
                else:
                    control_cost = torch.zeros([], device=device)

                rollout = rollout_markov_forward_images(
                    clean_images=clean,
                    net=ddp.module,
                    schedule=schedule,
                    class_labels=labels,
                    augment_labels=augment_labels,
                    zero_control=activation_scale <= 0,
                    control_scale=control_scale,
                )
                score_loss = local_denoise_loss(
                    net=ddp,
                    rollout=rollout,
                    schedule=schedule,
                    sigma_data=net.sigma_data,
                    class_labels=labels,
                    augment_labels=augment_labels,
                )
                score_loss.sum().mul(loss_scaling / batch_gpu_total).backward()
                score_losses.append(float(score_loss.detach().item()))
                control_costs.append(float(control_cost.detach().item()))

                with torch.no_grad():
                    terminal_stats = update_terminal_stats(
                        terminal_states=rollout["states"][:, -1, ...],
                        current=terminal_stats if terminal_stats is not None else initialize_terminal_stats(example=clean),
                        momentum=markov_terminal_momentum,
                    )

        for g in score_optimizer.param_groups:
            g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for param in net.model.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        score_optimizer.step()

        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        _update_ema(ema, net, ema_beta)

        training_stats.report('Loss/loss', np.mean(score_losses) if score_losses else 0.0)
        training_stats.report('Markov/activation_scale', activation_scale)
        training_stats.report('Markov/control_scale', control_scale)
        training_stats.report('Markov/lambda_dual', lambda_dual)
        training_stats.report('Markov/control_cost', np.mean(control_costs) if control_costs else 0.0)

        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(
                ema=ema,
                augment_pipe=augment_pipe,
                dataset_kwargs=dict(dataset_kwargs),
                markov_state={
                    "schedule": schedule.to_snapshot_dict(),
                    "terminal_mean": terminal_stats["mean"].detach().cpu() if terminal_stats is not None else None,
                    "terminal_var": terminal_stats["var"].detach().cpu() if terminal_stats is not None else None,
                    "reverse_control_scale": float(markov_reverse_control_scale),
                    "reverse_noise_scale": float(markov_reverse_noise_scale),
                    "base_control_scale": float(loss_kwargs.get('max_delta', 0.03)),
                },
            )
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)

        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            state = dict(
                net=net,
                score_optimizer_state=score_optimizer.state_dict(),
                control_optimizer_state=control_optimizer.state_dict(),
                lambda_dual=float(lambda_dual),
            )
            torch.save(state, os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))

        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()

        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    dist.print0()
    dist.print0('Exiting...')
