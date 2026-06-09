"""Compute FID between an already-generated image folder and a real image folder."""
import argparse
import torch
from cleanfid import fid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated_dir', type=str, required=True)
    parser.add_argument('--real_dir',      type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print('Computing FID...')
    score = fid.compute_fid(args.generated_dir, args.real_dir, device=device)
    print(f'FID: {score:.4f}')


if __name__ == '__main__':
    main()
