import copy
import json
import os
import pickle
import time

import dnnlib
import numpy as np
import torch
import torch.distributed as dist_torch
from torch_utils import distributed as dist
from torch_utils import misc
from torch_utils import training_stats

from training.wdro_utils import (
    _delta_to_bw_uint8,
    _images_to_uint8,
    _run_quick_eval,
    _safe_cpu_mem_gb,
    _save_image_grid,
    _save_individual_images,
    _save_triplet_rows_grid,
)

#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg      = 10000,    # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_kimg         = 0,        # Start from the given training progress.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    wdro_warmup_ratio   = 0.4,      # WDRO warmup ratio Sw/S.
    wdro_m_epochs       = 100,      # WDRO refresh interval in epochs.
    wdro_k              = 2,        # WDRO inner ascent steps.
    wdro_step_size      = 1e-3,     # WDRO inner ascent step size.
    wdro_gamma          = 1.0,      # WDRO penalty coefficient.
    wdro_p_adv          = 0.3,      # Probability of generating adversarial batch.
    wdro_attack_mode    = 'single', # WDRO attack mode: single or casual.
    wdro_m_times        = 4,        # Number of attacked timesteps per sample (casual mode).
    wdro_time_bins      = 40,       # Number of discretized timestep bins (casual mode).
    wdro_time_chunk     = 1,        # Number of attacked timesteps processed together.
    debug_eval_enable   = False,    # Run quick eval at init and each WDRO interval.
    debug_eval_init     = True,     # Run quick eval before main training iterations.
    debug_eval_num_images = 512,    # Number of generated images for quick FID.
    debug_eval_steps    = 18,       # Sampling steps for quick eval generation.
    debug_eval_batch_size = 64,     # Batch size for quick eval generation/FID.
    debug_eval_num_visual = 32,     # Number of sample images to save per quick eval.
    debug_eval_ref_path = None,     # Optional local .npz for reference stats.
    debug_adv_num_visual = 16,      # Number of adversarial/raw debug images per WDRO refresh.
    device              = torch.device('cuda'),
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    wdro_attack_mode = str(wdro_attack_mode).lower()
    if wdro_attack_mode == 'causal':
        wdro_attack_mode = 'casual'
    if wdro_attack_mode not in ['single', 'casual']:
        raise ValueError(f'Unsupported wdro_attack_mode="{wdro_attack_mode}". Expected one of: single, casual.')
    wdro_m_times = int(max(wdro_m_times, 1))
    wdro_time_bins = int(max(wdro_time_bins, 2))
    wdro_time_chunk = int(max(wdro_time_chunk, 1))
    if wdro_attack_mode == 'casual' and wdro_m_times > wdro_time_bins:
        raise ValueError(f'wdro_m_times ({wdro_m_times}) must be <= wdro_time_bins ({wdro_time_bins}).')

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    # assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    dist.print0()
    dist.print0('Num images: ', len(dataset_obj))
    dist.print0('Image shape:', dataset_obj.image_shape)
    dist.print0('Label shape:', dataset_obj.label_shape)
    dist.print0()


    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)
    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device)
            misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.(VP|VE|EDM)Loss
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device],find_unused_parameters=True)
    # ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device])
    # ddp._set_static_graph()
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data # conserve memory
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        resume_next_wdro_kimg = int(data['next_wdro_kimg']) if 'next_wdro_kimg' in data else None
        del data # conserve memory
    else:
        resume_next_wdro_kimg = None

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    debug_eval_state = dict(
        detector_net=None,
        mu_ref=None,
        sigma_ref=None,
        seed_base=int(seed),
        num_workers=int(data_loader_kwargs.get('num_workers', 2)),
        prefetch_factor=int(data_loader_kwargs.get('prefetch_factor', 2)),
    )

    wdro_start_kimg = int(wdro_warmup_ratio * total_kimg)
    if run_dir is None:
        run_dir = '.'
    wdro_dataset_path = os.path.join(run_dir, 'combined_dataset.pt')
    dist.print0(f"[WDRO] cur_path={wdro_dataset_path}, starting WDRO augmentation...")

    wdro_interval_kimg = max(int(wdro_m_epochs * len(dataset_obj) / 1000), 1)
    if resume_next_wdro_kimg is not None:
        # Prefer exact scheduler state from checkpoint when available.
        next_wdro_kimg = max(resume_next_wdro_kimg, wdro_start_kimg)
    elif cur_nimg >= wdro_start_kimg * 1000:
        # On resume from older checkpoints, skip already-passed WDRO intervals
        # to avoid expensive "catch-up" refreshes every single step.
        cur_kimg = cur_nimg // 1000
        passed_intervals = max((cur_kimg - wdro_start_kimg) // wdro_interval_kimg, 0)
        next_wdro_kimg = wdro_start_kimg + (passed_intervals + 1) * wdro_interval_kimg
    else:
        next_wdro_kimg = wdro_start_kimg
    n = len(dataset_obj)
    p_now = 0.12 if n >= 40000 else (0.15 if n >= 20000 else 0.18)
    augment_pipe.p = p_now
    dist.print0(f"[WDRO] schedule start/interval/next (kimg): {wdro_start_kimg}/{wdro_interval_kimg}/{next_wdro_kimg}")
    dist.print0(f"[WDRO] attack mode={wdro_attack_mode}, m_times={wdro_m_times}, time_bins={wdro_time_bins}, time_chunk={wdro_time_chunk}")
    if debug_eval_enable:
        dist.print0(f"[QuickEval] enabled. init={debug_eval_init}, num={debug_eval_num_images}, steps={debug_eval_steps}, batch={debug_eval_batch_size}, visual={debug_eval_num_visual}, adv_visual={debug_adv_num_visual}")

    if debug_eval_enable and debug_eval_init:
        if dist.get_world_size() > 1 and dist.get_rank() != 0:
            dist_torch.barrier()
        if dist.get_rank() == 0:
            try:
                _run_quick_eval(
                    ema=ema,
                    run_dir=run_dir,
                    dataset_path=dataset_kwargs['path'],
                    cur_nimg=cur_nimg,
                    device=device,
                    state=debug_eval_state,
                    num_images=debug_eval_num_images,
                    num_steps=debug_eval_steps,
                    batch_size=debug_eval_batch_size,
                    num_visual=debug_eval_num_visual,
                    ref_path=debug_eval_ref_path,
                )
            except Exception as err:
                dist.print0(f"[QuickEval][WARN] init evaluation failed: {err}")
        if dist.get_world_size() > 1:
            dist_torch.barrier()

    while True:
        if cur_nimg >= next_wdro_kimg * 1000:
            run_dir_list = [run_dir]
            dist_torch.broadcast_object_list(run_dir_list, src=0)
            run_dir = run_dir_list[0]

            rank = dist_torch.get_rank()
            print(f"[Rank {rank}] run_dir = {run_dir}", flush=True)
            dist.print0(f"[WDRO] cur_nimg={cur_nimg}, starting WDRO augmentation...")

            wdro_dataset_path = os.path.join(run_dir, f'combined_dataset-{cur_nimg // 1000:06d}.pt')

            if dist.get_rank() == 0:
                raw_loader = torch.utils.data.DataLoader(
                    dataset_obj, batch_size=batch_gpu, shuffle=False, **data_loader_kwargs
                )
                all_images, all_labels = [], []
                adv_debug_orig = []
                adv_debug_adv = []
                adv_debug_left = max(int(debug_adv_num_visual), 0)
                refresh_t0 = time.time()
                clean_count = 0
                adv_count = 0

                net.eval()
                for images, labels in raw_loader:
                    images = images.to(device).to(torch.float32) / 127.5 - 1
                    labels = labels.to(device)
                    clean_count += int(images.shape[0])
                    if wdro_p_adv >= 1.0 or torch.rand(1).item() < wdro_p_adv:
                        if wdro_attack_mode == 'casual':
                            adv_image_list = wdro_attack_multitime(
                                images,
                                labels,
                                model=net,
                                loss_fn=loss_fn,
                                augment_pipe=augment_pipe,
                                gamma=wdro_gamma,
                                alpha=wdro_step_size,
                                iters=wdro_k,
                                m_times=wdro_m_times,
                                time_bins=wdro_time_bins,
                                time_chunk=wdro_time_chunk,
                            )
                        else:
                            adv_image_list = [wdro_attack(
                                images,
                                labels,
                                model=net,
                                loss_fn=loss_fn,
                                augment_pipe=augment_pipe,
                                gamma=wdro_gamma,
                                alpha=wdro_step_size,
                                iters=wdro_k,
                            )]

                        if adv_debug_left > 0:
                            take = min(adv_debug_left, images.shape[0])
                            adv_debug_orig.append(images[:take].detach().cpu())
                            adv_debug_adv.append(adv_image_list[0][:take].detach().cpu())
                            adv_debug_left -= take

                        for adv_images in adv_image_list:
                            all_images.append(adv_images.cpu())
                            all_labels.append(labels.cpu())
                            adv_count += int(adv_images.shape[0])
                    all_images.append(images.cpu())
                    all_labels.append(labels.cpu())

                combined_images = torch.cat(all_images).detach().clone()
                combined_labels = torch.cat(all_labels)
                if combined_labels.requires_grad:
                    combined_labels = combined_labels.detach().clone()

                torch.save((combined_images, combined_labels), wdro_dataset_path)
                dist.print0(f"[WDRO] Augmented dataset saved to {wdro_dataset_path}")
                refresh_sec = time.time() - refresh_t0
                total_count = int(combined_images.shape[0])
                ratio = (adv_count / max(clean_count, 1))
                dist.print0(
                    f"[WDRO] refresh stats: mode={wdro_attack_mode} "
                    f"clean={clean_count} adv={adv_count} total={total_count} "
                    f"adv/clean={ratio:.3f} sec={refresh_sec:.2f}"
                )
                try:
                    metrics = dict(
                        kimg=int(cur_nimg // 1000),
                        mode=wdro_attack_mode,
                        m_times=int(wdro_m_times),
                        time_bins=int(wdro_time_bins),
                        time_chunk=int(wdro_time_chunk),
                        clean_count=int(clean_count),
                        adv_count=int(adv_count),
                        total_count=int(total_count),
                        adv_to_clean_ratio=float(ratio),
                        refresh_sec=float(refresh_sec),
                    )
                    metrics_path = os.path.join(run_dir, "casual_refresh_metrics.jsonl")
                    with open(metrics_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(metrics) + "\\n")
                except Exception as err:
                    dist.print0(f"[WDRO][WARN] failed to write refresh metrics: {err}")

                if len(adv_debug_orig) > 0 and len(adv_debug_adv) > 0:
                    adv_orig = torch.cat(adv_debug_orig, dim=0)
                    adv_gen = torch.cat(adv_debug_adv, dim=0)
                    orig_u8 = _images_to_uint8(adv_orig)
                    adv_u8 = _images_to_uint8(adv_gen)
                    delta_u8 = _delta_to_bw_uint8(adv_orig, adv_gen)
                    adv_dir = os.path.join(run_dir, 'quick_eval', 'adv_debug', f'kimg-{cur_nimg // 1000:06d}')
                    _save_image_grid(orig_u8, os.path.join(adv_dir, 'orig_grid.png'), ncols=8)
                    _save_image_grid(adv_u8, os.path.join(adv_dir, 'adv_grid.png'), ncols=8)
                    _save_image_grid(delta_u8, os.path.join(adv_dir, 'delta_grid.png'), ncols=8)
                    _save_triplet_rows_grid(
                        orig_u8,
                        adv_u8,
                        delta_u8,
                        os.path.join(adv_dir, 'triplet_grid_3row.png'),
                        max_cols=8,
                    )
                    _save_individual_images(orig_u8, os.path.join(adv_dir, 'orig_images'), prefix='orig')
                    _save_individual_images(adv_u8, os.path.join(adv_dir, 'adv_images'), prefix='adv')
                    dist.print0(f"[QuickEval] Saved adversarial debug visuals to {adv_dir}")

                if debug_eval_enable:
                    try:
                        _run_quick_eval(
                            ema=ema,
                            run_dir=run_dir,
                            dataset_path=dataset_kwargs['path'],
                            cur_nimg=cur_nimg,
                            device=device,
                            state=debug_eval_state,
                            num_images=debug_eval_num_images,
                            num_steps=debug_eval_steps,
                            batch_size=debug_eval_batch_size,
                            num_visual=debug_eval_num_visual,
                            ref_path=debug_eval_ref_path,
                        )
                    except Exception as err:
                        dist.print0(f"[QuickEval][WARN] interval evaluation failed at kimg={cur_nimg // 1000}: {err}")
                net.train()

            dist_torch.barrier()
            dataset_iterator = switch_to_wdro_dataset(
                wdro_dataset_path=wdro_dataset_path,
                batch_size=batch_gpu,
                data_loader_kwargs=data_loader_kwargs,
                device=device,
                net=net,
                loss_fn=loss_fn
            )

            next_wdro_kimg += wdro_interval_kimg

        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                # Raw dataset batches are uint8 [0,255], while WDRO combined batches are
                # already float in [-1,1]. Normalize only raw uint8 to avoid scale mismatch.
                if images.dtype == torch.uint8:
                    images = images.to(device).to(torch.float32) / 127.5 - 1
                else:
                    images = images.to(device).to(torch.float32)
                labels = labels.to(device)
                loss = loss_fn(net=ddp, images=images, labels=labels, augment_pipe=augment_pipe)
                training_stats.report('Loss/loss', loss)
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()

        # Update weights.
        for g in optimizer.param_groups:
            g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', _safe_cpu_mem_gb()):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data # conserve memory

        # Save full dump of the training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            torch.save(
                dict(
                    net=net,
                    optimizer_state=optimizer.state_dict(),
                    next_wdro_kimg=next_wdro_kimg,
                ),
                os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt')
            )

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')

def switch_to_wdro_dataset(wdro_dataset_path, batch_size, data_loader_kwargs, device, net, loss_fn):

    dist.print0(f"[WDRO-DDP] Loading saved WDRO-augmented {wdro_dataset_path} dataset...")
    combined_images, combined_labels = torch.load(wdro_dataset_path, map_location='cpu')
    wdro_dataset = torch.utils.data.TensorDataset(combined_images, combined_labels)
    dataset_sampler = misc.InfiniteSampler(dataset=wdro_dataset, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=0)

    dataloader = torch.utils.data.DataLoader(
        wdro_dataset,
        batch_size=batch_size,
        sampler=dataset_sampler,
        **data_loader_kwargs
    )
    return iter(dataloader)

def _build_edm_sigma_grid(device, time_bins, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    steps = int(max(time_bins, 2))
    idx = torch.arange(steps, device=device, dtype=torch.float32)
    t_steps = (sigma_max ** (1 / rho) + idx / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return t_steps

def _sample_distinct_time_indices(batch_size, m_times, time_bins, device):
    if m_times > time_bins:
        raise ValueError(f'm_times ({m_times}) must be <= time_bins ({time_bins})')
    # Uniform-without-replacement per sample.
    rand = torch.rand(batch_size, time_bins, device=device)
    return rand.argsort(dim=1)[:, :m_times]

def wdro_attack_fixed_sigma(
    images, labels, model, loss_fn, augment_pipe,
    *, sigma_values, gamma=1.0, alpha=1e-3, iters=2, clamp=(-1, 1)
):
    sigma_values = torch.as_tensor(sigma_values, device=images.device, dtype=torch.float32).reshape(-1)
    if sigma_values.shape[0] != images.shape[0]:
        raise ValueError(f"sigma_values batch ({sigma_values.shape[0]}) must match images batch ({images.shape[0]}).")

    x_adv = images.detach().clone().requires_grad_(True)
    fixed_noise = torch.randn_like(images) * sigma_values.reshape(-1, 1, 1, 1)

    model.eval()
    for _ in range(iters):
        with torch.enable_grad():
            loss_cls = loss_fn(
                net=model,
                images=x_adv,
                labels=labels,
                augment_pipe=augment_pipe,
                sigma_override=sigma_values,
                noise_override=fixed_noise,
            )
            delta = (x_adv - images).view(images.size(0), -1)
            C = 0.5 * (delta.pow(2).sum(dim=1)).mean()
            loss = loss_cls.mean() - gamma * C
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv + alpha * grad
        x_adv = torch.clamp(x_adv, *clamp)
        x_adv = x_adv.detach().requires_grad_(True)

    return x_adv.detach()

def wdro_attack_multitime(
    images, labels, model, loss_fn, augment_pipe,
    *, gamma=1.0, alpha=1e-3, iters=2, m_times=4, time_bins=40, time_chunk=1,
    clamp=(-1, 1), sigma_min=0.002, sigma_max=80.0, rho=7.0
):
    batch_size = images.shape[0]
    m_times = int(max(m_times, 1))
    time_chunk = int(max(time_chunk, 1))
    if m_times > time_bins:
        raise ValueError(f'm_times ({m_times}) must be <= time_bins ({time_bins})')

    sigma_grid = _build_edm_sigma_grid(
        device=images.device,
        time_bins=time_bins,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
    )
    time_idx = _sample_distinct_time_indices(
        batch_size=batch_size,
        m_times=m_times,
        time_bins=time_bins,
        device=images.device,
    )
    selected_sigmas = sigma_grid[time_idx]  # [B, M]

    adv_image_list = []
    for start in range(0, m_times, time_chunk):
        end = min(start + time_chunk, m_times)
        chunk = selected_sigmas[:, start:end]  # [B, Mc]
        mc = chunk.shape[1]

        # Repeat by timestep-chunk block so (images, sigma) pairs stay aligned:
        # [j0: all B samples], [j1: all B samples], ...
        images_rep = images.repeat(mc, 1, 1, 1)
        labels_rep = labels.repeat(mc, 1)
        sigma_rep = chunk.transpose(0, 1).reshape(-1)
        adv_rep = wdro_attack_fixed_sigma(
            images_rep,
            labels_rep,
            model=model,
            loss_fn=loss_fn,
            augment_pipe=augment_pipe,
            sigma_values=sigma_rep,
            gamma=gamma,
            alpha=alpha,
            iters=iters,
            clamp=clamp,
        )
        adv_chunk = adv_rep.reshape(mc, batch_size, *images.shape[1:])
        for idx in range(mc):
            adv_image_list.append(adv_chunk[idx])

    return adv_image_list

def wdro_attack(
    images, labels, model, loss_fn, augment_pipe,
    *, gamma=1.0, alpha=1e-3, iters=2, clamp=(-1, 1)
):
    x_adv = images.detach().clone().requires_grad_(True)

    model.eval()
    for _ in range(iters):
        with torch.enable_grad():
            loss_cls = loss_fn(net=model,
                               images=x_adv,
                               labels=labels,
                               augment_pipe=augment_pipe)

            delta = (x_adv - images).view(images.size(0), -1)
            C = 0.5 * (delta.pow(2).sum(dim=1)).mean()
            loss = loss_cls.mean() - gamma * C
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv + alpha * grad
        x_adv = torch.clamp(x_adv, *clamp)
        x_adv = x_adv.detach().requires_grad_(True)

    return x_adv.detach()

#----------------------------------------------------------------------------
