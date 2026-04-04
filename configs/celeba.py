import argparse

CELEBA_CONFIG = {
    'dataset': 'celeba',
    'img_size': 64,
    'in_ch': 3,
    'latent_size': 40,
    'lora_rank': 8,
    'batch_size': 16,
    'batch_size_val': 128,
    'epochs': 201,
    'lr': 1e-4,
    'noise_level': 0.0,
    'steps': 100,
    'sampling_method': 'rk',
    'eval_epoch': 10,
    'save_folder': './outputs',
    'exp_name': 'baseline',
}


def get_celeba_parser():
    """Build and return the argparse parser for CelebA flow matching training."""
    parser = argparse.ArgumentParser(description='CelebA Flow Matching Training')

    parser.add_argument('--dataset',         type=str,   default=CELEBA_CONFIG['dataset'])
    parser.add_argument('--img_size',        type=int,   default=CELEBA_CONFIG['img_size'])
    parser.add_argument('--in_ch',           type=int,   default=CELEBA_CONFIG['in_ch'])
    parser.add_argument('--latent_size',     type=int,   default=CELEBA_CONFIG['latent_size'])
    parser.add_argument('--lora_rank',       type=int,   default=CELEBA_CONFIG['lora_rank'])
    parser.add_argument('--batch_size',      type=int,   default=CELEBA_CONFIG['batch_size'])
    parser.add_argument('--batch_size_val',  type=int,   default=CELEBA_CONFIG['batch_size_val'])
    parser.add_argument('--epochs',          type=int,   default=CELEBA_CONFIG['epochs'])
    parser.add_argument('--lr',              type=float, default=CELEBA_CONFIG['lr'])
    parser.add_argument('--noise_level',     type=float, default=CELEBA_CONFIG['noise_level'])
    parser.add_argument('--steps',           type=int,   default=CELEBA_CONFIG['steps'])
    parser.add_argument('--sampling_method', type=str,   default=CELEBA_CONFIG['sampling_method'],
                        choices=['euler', 'rk'])
    parser.add_argument('--eval_epoch',      type=int,   default=CELEBA_CONFIG['eval_epoch'])
    parser.add_argument('--save_folder',     type=str,   default=CELEBA_CONFIG['save_folder'])
    parser.add_argument('--exp_name',        type=str,   default=CELEBA_CONFIG['exp_name'])

    return parser
