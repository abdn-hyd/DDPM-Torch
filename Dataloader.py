import os
import pickle
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Callable, Optional, List


class LocalCIFAR10(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable[[np.ndarray], torch.Tensor]] = None,
    ):
        """
        Args:
            root (str): path to the root folder
            train (bool): using training dataset or not
            transform (callable): transform function for the images
        """
        self.root = root
        self.transform = transform
        self.data = []
        self.labels = []

        if train:
            files = [f"data_batch_{i}" for i in range(1, 6)]
        else:
            files = ["test_batch"]

        for file in files:
            path = os.path.join(root, file)
            with open(path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                self.labels.extend(entry["labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose(0, 2, 3, 1)  # (N, H, W, C)

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self,
        idx: int,
    ):
        image = self.data[idx]  # shape (32, 32, 3)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            image = (image / 255.0 - 0.5) / 0.5  # [0,1] -> [-1,1]

        return image, label


def get_cifar10_dataloader(cfg):
    dataset = LocalCIFAR10(
        root=cfg.cifar10_path,
        train=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=cfg.data_mean if cfg.data_mean else [0.4914, 0.4822, 0.4465],
                    std=cfg.data_std if cfg.data_std else [0.2023, 0.1994, 0.2010],
                ),
            ]
        ),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    return dataloader


class LocalCelebA(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable[[Image.image], torch.Tensor]] = None,
        target_attributes: Optional[List[str]] = None,
    ):
        """
        Args:
            root (str): path to the root folder
            split (str): "train"（0-162,769）、"valid"（162,770-182,637）、"test"（182,638-202,599）
            transform (callable): transform function for the images
            target_attributes (list): return all attributes in default
        """
        self.root = root
        self.transform = transform

        attr_file = os.path.join(root, "list_attr_celeba.csv")
        self.attr_df = pd.read_csv(attr_file, delim_whitespace=True, header=1)

        split_file = os.path.join(root, "list_eval_partition.csv")
        split_df = pd.read_csv(
            split_file, delim_whitespace=True, header=None, index_col=0
        )

        split_code = {"train": 0, "valid": 1, "test": 2}[split]
        self.filenames = split_df[split_df[1] == split_code].index.tolist()

        self.target_attributes = target_attributes or self.attr_df.columns.tolist()
        self.attr_df = self.attr_df[self.target_attributes]

        self.attr_df = (self.attr_df + 1) // 2

    def __len__(self):
        return len(self.filenames)

    def __getitem__(
        self,
        idx: int,
    ):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.root, "img_align_celeba", img_name)
        image = Image.open(img_path)

        attributes = torch.tensor(self.attr_df.loc[img_name].values.astype("int64"))

        if self.transform:
            image = self.transform(image)
        else:
            pass

        return image, attributes


def get_celeba_dataloader(cfg):
    transform = transforms.Compose(
        [
            transforms.Resize(178),
            transforms.CenterCrop(178),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(cfg.image_size),  # 178X178 -> 64X64
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(cfg.data_mean if cfg.data_mean else [0.5063, 0.4258, 0.3832]),
                std=(cfg.data_std if cfg.data_std else [0.3105, 0.2903, 0.2896]),
            ),
        ]
    )

    dataset = LocalCelebA(
        root=cfg.celeba_root,
        split="train" if cfg.is_train else "test",
        transform=transform,
        target_attributes=cfg.target_attributes,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader
