import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import torch
import numpy as np


class CelebAImageFolder(Dataset):
    """Simple image folder dataset for CelebA — no metadata files required."""
    def __init__(self, img_dir, transform=None):
        self.paths = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0  # dummy label


def get_mnist_data_loaders(train_bs, val_bs, target_size, use_ddp=False, world_size=1, rank=0):
    """Return train and val DataLoaders for MNIST, resized to target_size."""
    train_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=train_bs, sampler=train_sampler, num_workers=4,
                                  pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=val_bs, sampler=val_sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=val_bs, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader


def get_celeba_data_loaders(train_bs, val_bs, target_size, use_ddp=False, world_size=1, rank=0):
    """Return train and val DataLoaders for CelebA, resized and center-cropped to target_size."""
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
    ])

    img_dir = os.path.join(os.path.dirname(__file__), '..', 'img_align_celeba')
    img_dir = os.path.abspath(img_dir)
    full_dataset = CelebAImageFolder(img_dir, transform=transform)

    # 90/10 train-val split
    n_train = int(0.9 * len(full_dataset))
    n_val = len(full_dataset) - n_train
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [n_train, n_val])

    num_workers = 0  # avoid multiprocessing hangs on HPC

    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=train_bs, sampler=train_sampler,
                                  num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=val_bs, sampler=val_sampler,
                                num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=val_bs, shuffle=False,
                                num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


def get_data_loaders(dataset_name, train_bs, val_bs, target_size=32, use_ddp=False, world_size=1, rank=0):
    """Route to the correct dataset loader by name ('mnist' or 'celeba')."""
    dataset_name = dataset_name.lower()
    if dataset_name == 'mnist':
        return get_mnist_data_loaders(train_bs, val_bs, target_size, use_ddp, world_size, rank)
    elif dataset_name == 'celeba':
        return get_celeba_data_loaders(train_bs, val_bs, target_size, use_ddp, world_size, rank)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
