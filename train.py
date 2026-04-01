import os
import re
import json
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from training import training_loop
from training import training_wdro_loop
from training import training_cdro_markov_loop

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()

# Main options.
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, required=True)
@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, required=True)
@click.option('--cond',          help='Train class-conditional model', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--arch',          help='Network architecture', metavar='ddpmpp|ncsnpp|adm',          type=click.Choice(['ddpmpp', 'ncsnpp', 'adm']), default='ddpmpp', show_default=True)
@click.option('--precond',       help='Preconditioning & loss function', metavar='wdroedm|advedm|cdroedm|cdromarkovedm|cdromarkovfull',       type=click.Choice(['wdroedm', 'advedm', 'cdroedm', 'cdromarkovedm', 'cdromarkovfull']), default='wdroedm', show_default=True)
@click.option('--trainer',       help='Training loop', metavar='baseline|wdro',                     type=click.Choice(['baseline', 'wdro']), default='wdro', show_default=True)
@click.option('--wdro-warmup-ratio', help='WDRO warmup ratio (Sw/S)', metavar='FLOAT',                type=click.FloatRange(min=0, max=1), default=0.4, show_default=True)
@click.option('--wdro-m-epochs', help='WDRO refresh interval in epochs (m)', metavar='INT',            type=click.IntRange(min=1), default=100, show_default=True)
@click.option('--wdro-k',        help='WDRO inner ascent steps (K)', metavar='INT',                    type=click.IntRange(min=1), default=2, show_default=True)
@click.option('--wdro-step-size',help='WDRO inner ascent step size', metavar='FLOAT',                  type=click.FloatRange(min=0, min_open=True), default=1e-3, show_default=True)
@click.option('--wdro-gamma',    help='WDRO penalty gamma', metavar='FLOAT',                           type=click.FloatRange(min=0), default=1.0, show_default=True)
@click.option('--wdro-p-adv',    help='Probability of generating adversarial batch', metavar='FLOAT',  type=click.FloatRange(min=0, max=1), default=0.3, show_default=True)
@click.option('--cdro-mix',      help='CDRO robust loss mix weight', metavar='FLOAT',                  type=click.FloatRange(min=0, max=1), default=0.3, show_default=True)
@click.option('--cdro-adv-steps',help='CDRO inner ascent steps', metavar='INT',                        type=click.IntRange(min=1), default=2, show_default=True)
@click.option('--cdro-step-size',help='CDRO inner ascent step size', metavar='FLOAT',                  type=click.FloatRange(min=0, min_open=True), default=0.02, show_default=True)
@click.option('--cdro-max-delta',help='CDRO max RMS perturbation radius in image space', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=0.05, show_default=True)
@click.option('--cdro-rho',      help='CDRO target transport budget', metavar='FLOAT',                 type=click.FloatRange(min=0), default=1e-4, show_default=True)
@click.option('--cdro-lambda-init', help='CDRO initial dual variable', metavar='FLOAT',                type=click.FloatRange(min=0), default=0.1, show_default=True)
@click.option('--cdro-lambda-lr', help='CDRO dual update step size', metavar='FLOAT',                  type=click.FloatRange(min=0, min_open=True), default=1e-3, show_default=True)
@click.option('--cdro-start-kimg', help='CDRO activation start in kimg', metavar='FLOAT',              type=click.FloatRange(min=0), default=0.0, show_default=True)
@click.option('--cdro-ramp-kimg', help='CDRO activation ramp length in kimg', metavar='FLOAT',         type=click.FloatRange(min=0), default=0.0, show_default=True)
@click.option('--cdro-sigma-floor', help='CDRO sigma floor below which control vanishes', metavar='FLOAT', type=click.FloatRange(min=0), default=0.0, show_default=True)
@click.option('--cdro-sigma-cut', help='CDRO sigma cutoff above which control vanishes', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=0.5, show_default=True)
@click.option('--cdro-gate-power', help='CDRO sigma gate exponent', metavar='FLOAT',                   type=click.FloatRange(min=0, min_open=True), default=2.0, show_default=True)
@click.option('--cdro-delta-space', help='CDRO perturbation parameterization', metavar='image|noise',  type=click.Choice(['image', 'noise']), default='image', show_default=True)
@click.option('--cdro-control-cbase', help='Control head base channels for non-plug-in CDRO', metavar='INT', type=click.IntRange(min=8), default=64, show_default=True)
@click.option('--cdro-control-dropout', help='Control head dropout for non-plug-in CDRO', metavar='FLOAT', type=click.FloatRange(min=0, max=1), default=0.0, show_default=True)
@click.option('--markov-num-steps', help='Full Markov forward/reverse steps', metavar='INT', type=click.IntRange(min=1), default=8, show_default=True)
@click.option('--markov-total-time', help='Full Markov total time horizon', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=1.0, show_default=True)
@click.option('--markov-beta-min', help='Full Markov beta min', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=0.1, show_default=True)
@click.option('--markov-beta-max', help='Full Markov beta max', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=12.0, show_default=True)
@click.option('--markov-sde-family', help='Full Markov SDE family', metavar='vp_linear|vp_cosine', type=click.Choice(['vp_linear', 'vp_cosine']), default='vp_cosine', show_default=True)
@click.option('--markov-weight-schedule', help='Full Markov loss weighting', metavar='uniform|sigma_sq|inv_sigma_sq', type=click.Choice(['uniform', 'sigma_sq', 'inv_sigma_sq']), default='uniform', show_default=True)
@click.option('--markov-control-lr', help='Full Markov control optimizer LR', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=2e-4, show_default=True)
@click.option('--markov-lambda-min', help='Full Markov minimum dual value', metavar='FLOAT', type=click.FloatRange(min=0), default=0.0, show_default=True)
@click.option('--markov-reverse-control-scale', help='Full Markov reverse-time control scale', metavar='FLOAT', type=click.FloatRange(min=0), default=1.0, show_default=True)
@click.option('--markov-reverse-noise-scale', help='Full Markov reverse-time noise scale', metavar='FLOAT', type=click.FloatRange(min=0), default=1.0, show_default=True)
@click.option('--markov-terminal-momentum', help='Full Markov terminal stats EMA momentum', metavar='FLOAT', type=click.FloatRange(min=0, max=1), default=0.95, show_default=True)
@click.option('--debug-eval',    help='Run quick debug evaluation at init and each WDRO interval', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--debug-eval-init', help='Run quick debug evaluation at training start', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--debug-eval-num', help='Number of generated images for quick FID', metavar='INT', type=click.IntRange(min=2), default=512, show_default=True)
@click.option('--debug-eval-steps', help='Sampling steps for quick eval generation', metavar='INT', type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--debug-eval-batch', help='Batch size for quick eval generation/FID', metavar='INT', type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--debug-eval-visual', help='Number of generated preview images to save each eval', metavar='INT', type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--debug-eval-ref', help='Optional local .npz file for reference FID stats', metavar='NPZ', type=str, default='')
@click.option('--debug-adv-visual', help='Number of adversarial/raw debug images to save per WDRO refresh', metavar='INT', type=click.IntRange(min=0), default=16, show_default=True)

# Hyperparameters.
@click.option('--duration',      help='Training duration', metavar='MIMG',                          type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--cbase',         help='Channel multiplier  [default: varies]', metavar='INT',       type=int)
@click.option('--cres',          help='Channels per resolution  [default: varies]', metavar='LIST', type=parse_int_list)
@click.option('--lr',            help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=10e-5, show_default=True)
@click.option('--ema',           help='EMA half-life', metavar='MIMG',                              type=click.FloatRange(min=0), default=0.5, show_default=True)
@click.option('--dropout',       help='Dropout probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.13, show_default=True)
@click.option('--augment',       help='Augment probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.12, show_default=True)
@click.option('--xflip',         help='Enable dataset x-flips', metavar='BOOL',                     type=bool, default=False, show_default=True)

# Performance-related.
@click.option('--fp16',          help='Enable mixed-precision training', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--ls',            help='Loss scaling', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=1), default=1, show_default=True)

# I/O-related.
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--tick',          help='How often to print progress', metavar='KIMG',                type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--snap',          help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--dump',          help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=500, show_default=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--transfer',      help='Transfer learning from network pickle', metavar='PKL|URL',   type=str)
@click.option('--resume',        help='Resume from previous training state', metavar='PT',          type=str)
@click.option('-n', '--dry-run', help='Print training options and exit',                            is_flag=True)

def main(**kwargs):
    """Train diffusion-based generative model using the techniques described in the
    paper "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Train DDPM++ model for class-conditional CIFAR-10 using 8 GPUs
    torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \\
        --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp
    """
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    # Initialize config dict.
    c = dnnlib.EasyDict()
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data, use_labels=opts.cond, xflip=opts.xflip, cache=opts.cache)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9,0.999], eps=1e-8)

    # Validate dataset options.
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_name = dataset_obj.name
        c.dataset_kwargs.resolution = dataset_obj.resolution # be explicit about dataset resolution
        c.dataset_kwargs.max_size = len(dataset_obj) # be explicit about dataset size
        if opts.cond and not dataset_obj.has_labels:
            raise click.ClickException('--cond=True requires labels specified in dataset.json')
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    # Network architecture.
    if opts.arch == 'ddpmpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[2,2,2])
    elif opts.arch == 'ncsnpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='fourier', encoder_type='residual', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=2, resample_filter=[1,3,3,1], model_channels=128, channel_mult=[2,2,2])
    else:
        assert opts.arch == 'adm'
        c.network_kwargs.update(model_type='DhariwalUNet', model_channels=192, channel_mult=[1,2,3,4])

    # Preconditioning & loss function.
    if opts.precond == 'advedm':
        c.network_kwargs.class_name = 'training.networks.EDMPrecond'
        c.loss_kwargs.class_name = 'training.loss.EDMLossAdv'
    elif opts.precond == 'cdroedm':
        c.network_kwargs.class_name = 'training.networks.EDMPrecond'
        c.loss_kwargs.class_name = 'training.loss.EDMLossCDRO'
        c.loss_kwargs.update(
            robust_mix=opts.cdro_mix,
            adv_steps=opts.cdro_adv_steps,
            adv_step_size=opts.cdro_step_size,
            max_delta=opts.cdro_max_delta,
            rho_target=opts.cdro_rho,
            lambda_init=opts.cdro_lambda_init,
            lambda_lr=opts.cdro_lambda_lr,
            start_kimg=opts.cdro_start_kimg,
            ramp_kimg=opts.cdro_ramp_kimg,
            sigma_floor=opts.cdro_sigma_floor,
            sigma_cut=opts.cdro_sigma_cut,
            gate_power=opts.cdro_gate_power,
            delta_space=opts.cdro_delta_space,
        )
    elif opts.precond == 'cdromarkovedm':
        c.network_kwargs.class_name = 'training.networks.EDMPrecondControl'
        c.network_kwargs.update(
            control_model_channels=opts.cdro_control_cbase,
            control_dropout=opts.cdro_control_dropout,
        )
        c.loss_kwargs.class_name = 'training.loss.EDMLossCDROMarkov'
        c.loss_kwargs.update(
            robust_mix=opts.cdro_mix,
            max_delta=opts.cdro_max_delta,
            rho_target=opts.cdro_rho,
            lambda_init=opts.cdro_lambda_init,
            lambda_lr=opts.cdro_lambda_lr,
            start_kimg=opts.cdro_start_kimg,
            ramp_kimg=opts.cdro_ramp_kimg,
            sigma_floor=opts.cdro_sigma_floor,
            sigma_cut=opts.cdro_sigma_cut,
            gate_power=opts.cdro_gate_power,
            delta_space=opts.cdro_delta_space,
        )
    elif opts.precond == 'cdromarkovfull':
        c.network_kwargs.class_name = 'training.networks.EDMPrecondControl'
        c.network_kwargs.update(
            control_model_channels=opts.cdro_control_cbase,
            control_dropout=opts.cdro_control_dropout,
        )
        c.loss_kwargs.update(
            max_delta=opts.cdro_max_delta,
            rho_target=opts.cdro_rho,
            lambda_init=opts.cdro_lambda_init,
            lambda_lr=opts.cdro_lambda_lr,
            start_kimg=opts.cdro_start_kimg,
            ramp_kimg=opts.cdro_ramp_kimg,
        )
    else:
        assert opts.precond == 'wdroedm'
        c.network_kwargs.class_name = 'training.networks.EDMPrecond'
        c.loss_kwargs.class_name = 'training.loss.EDMLossWdro'

    if opts.precond in {'cdroedm', 'cdromarkovedm', 'cdromarkovfull'} and opts.trainer != 'baseline':
        raise click.ClickException(f'--precond={opts.precond} currently supports only --trainer=baseline')

    # Network options.
    if opts.cbase is not None:
        c.network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        c.network_kwargs.channel_mult = opts.cres
    if opts.augment:
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', p=opts.augment)
        c.augment_kwargs.update(xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
        c.network_kwargs.augment_dim = 9
    c.network_kwargs.update(dropout=opts.dropout, use_fp16=opts.fp16)

    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)
    c.update(
        wdro_warmup_ratio=opts.wdro_warmup_ratio,
        wdro_m_epochs=opts.wdro_m_epochs,
        wdro_k=opts.wdro_k,
        wdro_step_size=opts.wdro_step_size,
        wdro_gamma=opts.wdro_gamma,
        wdro_p_adv=opts.wdro_p_adv,
        debug_eval_enable=opts.debug_eval,
        debug_eval_init=opts.debug_eval_init,
        debug_eval_num_images=opts.debug_eval_num,
        debug_eval_steps=opts.debug_eval_steps,
        debug_eval_batch_size=opts.debug_eval_batch,
        debug_eval_num_visual=opts.debug_eval_visual,
        debug_eval_ref_path=(opts.debug_eval_ref if opts.debug_eval_ref else None),
        debug_adv_num_visual=opts.debug_adv_visual,
        markov_num_steps=opts.markov_num_steps,
        markov_total_time=opts.markov_total_time,
        markov_beta_min=opts.markov_beta_min,
        markov_beta_max=opts.markov_beta_max,
        markov_sde_family=opts.markov_sde_family,
        markov_weight_schedule=opts.markov_weight_schedule,
        markov_control_lr=opts.markov_control_lr,
        markov_lambda_min=opts.markov_lambda_min,
        markov_reverse_control_scale=opts.markov_reverse_control_scale,
        markov_reverse_noise_scale=opts.markov_reverse_noise_scale,
        markov_terminal_momentum=opts.markov_terminal_momentum,
    )

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        seed = torch.randint(1 << 31, size=[], device=seed_device)
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Transfer learning and resume.
    if opts.transfer is not None:
        if opts.resume is not None:
            raise click.ClickException('--transfer and --resume cannot be specified at the same time')
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None
    elif opts.resume is not None:
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    # Description string.
    cond_str = 'cond' if c.dataset_kwargs.use_labels else 'uncond'
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    desc = f'{dataset_name:s}-{cond_str:s}-{opts.arch:s}-{opts.precond:s}-{opts.trainer:s}-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-{dtype_str:s}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
    dist.print0(f'Network architecture:    {opts.arch}')
    dist.print0(f'Preconditioning & loss:  {opts.precond}')
    dist.print0(f'Training loop:           {opts.trainer}')
    if opts.trainer == 'wdro':
        dist.print0(f'WDRO warmup ratio:       {c.wdro_warmup_ratio}')
        dist.print0(f'WDRO interval m (epoch): {c.wdro_m_epochs}')
        dist.print0(f'WDRO K/step/gamma/padv:  {c.wdro_k}/{c.wdro_step_size}/{c.wdro_gamma}/{c.wdro_p_adv}')
        dist.print0(f'Debug eval enabled:      {c.debug_eval_enable}')
        if c.debug_eval_enable:
            dist.print0(f'Debug eval cfg:          init={c.debug_eval_init} num={c.debug_eval_num_images} steps={c.debug_eval_steps} batch={c.debug_eval_batch_size} visual={c.debug_eval_num_visual}')
            dist.print0(f'Debug eval ref:          {c.debug_eval_ref_path}')
            dist.print0(f'Debug adv visuals:       {c.debug_adv_num_visual}')
    if opts.precond in {'cdroedm', 'cdromarkovedm', 'cdromarkovfull'}:
        dist.print0(f'CDRO mix/steps/step:     {opts.cdro_mix}/{opts.cdro_adv_steps}/{opts.cdro_step_size}')
        dist.print0(f'CDRO max_delta/rho:      {opts.cdro_max_delta}/{opts.cdro_rho}')
        dist.print0(f'CDRO lambda init/lr:     {opts.cdro_lambda_init}/{opts.cdro_lambda_lr}')
        dist.print0(f'CDRO start/ramp kimg:    {opts.cdro_start_kimg}/{opts.cdro_ramp_kimg}')
        dist.print0(f'CDRO sigma floor/cut:    {opts.cdro_sigma_floor}/{opts.cdro_sigma_cut}')
        dist.print0(f'CDRO gate power:         {opts.cdro_gate_power}')
        dist.print0(f'CDRO delta space:        {opts.cdro_delta_space}')
    if opts.precond in {'cdromarkovedm', 'cdromarkovfull'}:
        dist.print0(f'CDRO control cbase/drop: {opts.cdro_control_cbase}/{opts.cdro_control_dropout}')
    if opts.precond == 'cdromarkovfull':
        dist.print0(f'Markov steps/time:       {opts.markov_num_steps}/{opts.markov_total_time}')
        dist.print0(f'Markov beta min/max:     {opts.markov_beta_min}/{opts.markov_beta_max}')
        dist.print0(f'Markov family/weights:   {opts.markov_sde_family}/{opts.markov_weight_schedule}')
        dist.print0(f'Markov ctrl lr/lmin:     {opts.markov_control_lr}/{opts.markov_lambda_min}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0()

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    if opts.precond == 'cdromarkovfull':
        markov_config = dnnlib.EasyDict(c)
        for key in [
            'wdro_warmup_ratio',
            'wdro_m_epochs',
            'wdro_k',
            'wdro_step_size',
            'wdro_gamma',
            'wdro_p_adv',
            'debug_eval_enable',
            'debug_eval_init',
            'debug_eval_num_images',
            'debug_eval_steps',
            'debug_eval_batch_size',
            'debug_eval_num_visual',
            'debug_eval_ref_path',
            'debug_adv_num_visual',
        ]:
            markov_config.pop(key, None)
        training_cdro_markov_loop.training_loop(**markov_config)
    elif opts.trainer == 'baseline':
        baseline_config = dnnlib.EasyDict(c)
        for key in [
            'wdro_warmup_ratio',
            'wdro_m_epochs',
            'wdro_k',
            'wdro_step_size',
            'wdro_gamma',
            'wdro_p_adv',
            'debug_eval_enable',
            'debug_eval_init',
            'debug_eval_num_images',
            'debug_eval_steps',
            'debug_eval_batch_size',
            'debug_eval_num_visual',
            'debug_eval_ref_path',
            'debug_adv_num_visual',
            'markov_num_steps',
            'markov_total_time',
            'markov_beta_min',
            'markov_beta_max',
            'markov_sde_family',
            'markov_weight_schedule',
            'markov_control_lr',
            'markov_lambda_min',
            'markov_reverse_control_scale',
            'markov_reverse_noise_scale',
            'markov_terminal_momentum',
        ]:
            baseline_config.pop(key, None)
        training_loop.training_loop(**baseline_config)
    else:
        wdro_config = dnnlib.EasyDict(c)
        for key in [
            'markov_num_steps',
            'markov_total_time',
            'markov_beta_min',
            'markov_beta_max',
            'markov_sde_family',
            'markov_weight_schedule',
            'markov_control_lr',
            'markov_lambda_min',
            'markov_reverse_control_scale',
            'markov_reverse_noise_scale',
            'markov_terminal_momentum',
        ]:
            wdro_config.pop(key, None)
        training_wdro_loop.training_loop(**wdro_config)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
