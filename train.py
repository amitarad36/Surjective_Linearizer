import os
import json
import sys

import wandb

from configs.celeba import get_celeba_parser
from data.data_utils import get_data_loaders
from linearizer.one_step import OneStepLinearizer
from utils.model_utils import get_g, get_linear_network
from training.flow_matching import train_flow_matching


def main():
    """Parse args, build models, and launch flow matching training on CelebA."""
    parser = get_celeba_parser()
    args = parser.parse_args()
    print(f"Arguments: {args}")

    # --- wandb --- #
    wandb.init(
        project="surjective-linearizer",
        name=args.exp_name,
        config=vars(args),
    )

    # --- data --- #
    dataloader, _ = get_data_loaders(args.dataset, args.batch_size, args.batch_size_val,
                                     target_size=args.img_size)
    print(f"Loaded dataset: {args.dataset}")

    # --- models --- #
    # Single shared g: noise and data are assumed to live in the same space,
    # so one encoder warps both distributions into the linear latent space.
    g = get_g(img_ch=args.in_ch, img_size=args.img_size, latent_size=args.latent_size)
    linear_network = get_linear_network(latent_size=args.latent_size, lora_rank=args.lora_rank)
    linearizer = OneStepLinearizer(gx=g, gy=None, linear_network=linear_network)

    # --- output folder --- #
    # Use exp_name so each experiment has its own checkpoint dir
    save_folder = os.path.join(args.save_folder, args.exp_name)
    os.makedirs(save_folder, exist_ok=True)
    with open(os.path.join(save_folder, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # --- train --- #
    print("Starting Flow Matching training...")
    train_flow_matching(
        linearizer=linearizer,
        dataloader=dataloader,
        epochs=args.epochs,
        lr=args.lr,
        noise_level=args.noise_level,
        steps=args.steps,
        sampling_method=args.sampling_method,
        eval_epoch=args.eval_epoch,
        save_folder=save_folder,
        img_size=args.img_size,
        num_of_ch=args.in_ch,
        latent_size=args.latent_size,
    )
    wandb.finish()
    print("Training completed!")


if __name__ == '__main__':
    main()
