import os
import re
import click
import tqdm
import pickle
import torch
import PIL.Image
import dnnlib

from torch_utils import distributed as dist
from training.markov_utils import ImageMarkovSchedule, sample_markov_reverse_images


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])


def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


@click.command()
@click.option('--network', 'network_pkl', type=str, required=True)
@click.option('--outdir', type=str, required=True)
@click.option('--seeds', type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs', is_flag=True)
@click.option('--class', 'class_idx', type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--reverse-control-scale', type=click.FloatRange(min=0), default=None)
@click.option('--noise-scale', type=click.FloatRange(min=0), default=None)
def main(network_pkl, outdir, subdirs, seeds, class_idx, max_batch_size, reverse_control_scale, noise_scale, device=torch.device('cuda')):
    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank()::dist.get_world_size()]

    if dist.get_rank() != 0:
        torch.distributed.barrier()

    dist.print0(f'Loading Markov network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        payload = pickle.load(f)
    net = payload['ema'].to(device)
    markov_state = payload.get('markov_state')
    if markov_state is None:
        raise click.ClickException('Snapshot does not contain markov_state; cannot use generate_markov.py')
    schedule = ImageMarkovSchedule.from_snapshot_dict(markov_state['schedule'], device=device)
    terminal_mean = markov_state['terminal_mean']
    terminal_var = markov_state['terminal_var']
    if terminal_mean is None or terminal_var is None:
        raise click.ClickException('Snapshot markov_state is missing terminal statistics')
    terminal_stats = {
        'mean': terminal_mean.to(device=device, dtype=torch.float32),
        'var': terminal_var.to(device=device, dtype=torch.float32),
    }
    reverse_control_scale = markov_state.get('reverse_control_scale', 1.0) if reverse_control_scale is None else reverse_control_scale
    noise_scale = markov_state.get('reverse_noise_scale', 1.0) if noise_scale is None else noise_scale
    base_control_scale = markov_state.get('base_control_scale', 1.0)

    if dist.get_rank() == 0:
        torch.distributed.barrier()

    dist.print0(f'Generating {len(seeds)} Markov images to "{outdir}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        images = sample_markov_reverse_images(
            net=net,
            schedule=schedule,
            terminal_stats=terminal_stats,
            latents=latents,
            class_labels=class_labels,
            reverse_control_scale=reverse_control_scale,
            noise_scale=noise_scale,
            base_control_scale=base_control_scale,
        )

        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{seed:06d}.png')
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
            else:
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)

    torch.distributed.barrier()
    dist.print0('Done.')


if __name__ == "__main__":
    main()
