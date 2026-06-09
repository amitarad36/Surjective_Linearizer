"""Generate images from a checkpoint and compute FID against real CelebA images."""
import os
import argparse
import torch
import wandb
from torchvision.utils import save_image
from cleanfid import fid

from linearizer.one_step import OneStepLinearizer
from training.flow_matching import FlowMatcher
from utils.model_utils import get_g, get_linear_network


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',    type=str, required=True)
    parser.add_argument('--real_dir',      type=str, required=True)
    parser.add_argument('--output_dir',    type=str, default='fid_samples')
    parser.add_argument('--num_images',    type=int, default=50000)
    parser.add_argument('--batch_size',    type=int, default=64)
    parser.add_argument('--latent_size',   type=int, default=128)
    parser.add_argument('--lora_rank',     type=int, default=8)
    parser.add_argument('--img_size',      type=int, default=64)
    parser.add_argument('--in_ch',         type=int, default=3)
    parser.add_argument('--steps',         type=int, default=100)
    parser.add_argument('--method',        type=str, default='rk', choices=['euler', 'rk', 'one_step'])
    parser.add_argument('--T',             type=int, default=100,
                        help='Number of ODE steps to collapse into B (only for one_step method)')
    parser.add_argument('--exp_name',      type=str, default='fid_eval',
                        help='wandb run name')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    wandb.init(project='surjective-linearizer', name=args.exp_name, config=vars(args))

    # build and load model
    g = get_g(img_ch=args.in_ch, img_size=args.img_size, latent_size=args.latent_size)
    linear_network = get_linear_network(latent_size=args.latent_size, lora_rank=args.lora_rank)
    linearizer = OneStepLinearizer(gx=g, gy=None, linear_network=linear_network)
    linearizer.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    linearizer = linearizer.to(device)

    fm = FlowMatcher(linearizer, latent_size=args.latent_size)
    fm = fm.to(device)
    fm.eval()

    # precompute B once for one_step
    B = None
    if args.method == 'one_step':
        print(f'Precomputing B matrix (T={args.T})...')
        B = fm.get_sampling_terms(device, T=args.T, sampling_method='rk')
        print('Done.')

    # generate images
    img_idx = 0
    with torch.no_grad():
        while img_idx < args.num_images:
            bs = min(args.batch_size, args.num_images - img_idx)
            noise = torch.randn(bs, args.in_ch, args.img_size, args.img_size, device=device)
            if args.method == 'one_step':
                samples = fm.sample_one_step(noise, device=device, B=B)
            else:
                samples = fm.sample(noise, device=device, steps=args.steps, method=args.method)
            samples = torch.clamp(samples, 0, 1)
            for i in range(samples.shape[0]):
                save_image(samples[i], os.path.join(args.output_dir, f'{img_idx:05d}.png'))
                img_idx += 1
            print(f'Generated {img_idx}/{args.num_images}', flush=True)

    # compute FID
    print('Computing FID...')
    score = fid.compute_fid(args.output_dir, args.real_dir, device=device)
    print(f'FID: {score:.4f}')

    label = f'one_step_T{args.T}' if args.method == 'one_step' else f'{args.steps}_steps'
    wandb.log({f'fid/{label}': score})
    wandb.finish()


if __name__ == '__main__':
    main()
