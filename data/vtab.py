from typing import Any
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .tf_dataset import TFDataset
from timm.data import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)

DATASETS = [
    "caltech101",
    "cifar(num_classes=100)",
    "dtd",
    "oxford_flowers102",
    "oxford_iiit_pet",
    "patch_camelyon",
    "sun397",
    "svhn",
    "resisc45",
    "eurosat",
    "dmlab",
    'kitti(task="closest_vehicle_distance")',
    'smallnorb(predicted_attribute="label_azimuth")',
    'smallnorb(predicted_attribute="label_elevation")',
    'dsprites(predicted_attribute="label_x_position",num_classes=16)',
    'dsprites(predicted_attribute="label_orientation",num_classes=16)',
    'clevr(task="closest_object_distance")',
    'clevr(task="count_all")',
    'diabetic_retinopathy(config="btgraham-300")',
]


class VtabData(LightningDataModule):
    def __init__(
        self,
        data_dir,
        dataset,
        train_split="train",
        val_split="val",
        num_workers=4,
        batch_size=64,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        assert train_split in ["train", "trainval"]
        assert val_split in ["val", "test"]
        self.train_split = train_split
        self.val_split = val_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.mean = mean
        self.std = std

    def train_dataloader(self):
        dataset = TFDataset(
            self.dataset, self.data_dir, self.train_split, self.mean, self.std
        )

        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        dataset = TFDataset(
            self.dataset, self.data_dir, self.val_split, self.mean, self.std,
        )

        return DataLoader(
                dataset,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
        )
