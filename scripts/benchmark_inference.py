"""Benchmark: how fast to generate 100 images from 100 noise samples using one-step (B matrix)."""
import argparse
import time
import torch
import wandb

from linearizer.one_step import OneStepLinearizer
from training.flow_matching import FlowMatcher
from utils.model_utils import get_g, get_linear_network


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',  type=str, required=True)
    parser.add_argument('--latent_size', type=int, default=128)
    parser.add_argument('--lora_rank',   type=int, default=8)
    parser.add_argument('--img_size',    type=int, default=64)
    parser.add_argument('--in_ch',       type=int, default=3)
    parser.add_argument('--T',           type=int, default=100,
                        help='Number of ODE steps used to build B')
    parser.add_argument('--runs',        type=int, default=20,
                        help='Number of timed repetitions to average over')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    wandb.init(project='surjective-linearizer', name='benchmark_onestep', config=vars(args))

    # build model
    g = get_g(img_ch=args.in_ch, img_size=args.img_size, latent_size=args.latent_size)
    linear_network = get_linear_network(latent_size=args.latent_size, lora_rank=args.lora_rank)
    linearizer = OneStepLinearizer(gx=g, gy=None, linear_network=linear_network)
    linearizer.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    linearizer = linearizer.to(device)

    fm = FlowMatcher(linearizer, latent_size=args.latent_size)
    fm = fm.to(device)
    fm.eval()

    # precompute B once (this is offline — not counted in inference time)
    print(f'Precomputing B matrix (T={args.T})...')
    t_b0 = time.perf_counter()
    B = fm.get_sampling_terms(device, T=args.T, sampling_method='rk')
    t_b1 = time.perf_counter()
    b_build_sec = t_b1 - t_b0
    print(f'B matrix built in {b_build_sec:.2f}s')

    noise = torch.randn(100, args.in_ch, args.img_size, args.img_size, device=device)

    with torch.no_grad():
        # warmup
        for _ in range(3):
            fm.sample_one_step(noise, device=device, B=B)
        torch.cuda.synchronize()

        # timed runs
        t0 = time.perf_counter()
        for _ in range(args.runs):
            fm.sample_one_step(noise, device=device, B=B)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / args.runs

    per_image_ms = elapsed / 100 * 1000

    print(f'\n100 images generated in: {elapsed*1000:.2f} ms')
    print(f'Per image:               {per_image_ms:.3f} ms')
    print(f'B precompute time:       {b_build_sec:.2f} s (one-time cost)')

    wandb.log({
        'inference/100_images_ms':   elapsed * 1000,
        'inference/per_image_ms':    per_image_ms,
        'inference/B_build_time_s':  b_build_sec,
        'inference/T':               args.T,
    })
    wandb.finish()


if __name__ == '__main__':
    main()
