import os
import json
import math
from typing import Any
from collections import Counter
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from timm.data import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data.constants import DEFAULT_CROP_PCT


class FGVCData(LightningDataModule):
    def __init__(
        self,
        data_dir,
        dataset,
        train_split="train",
        val_split="val",
        input_size=224,
        batch_size=64,
        num_workers=4,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        interpolation="bicubic",
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.train_split = train_split
        self.val_split = val_split
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean = mean
        self.std = std
        self.interpolation = interpolation

    def train_dataloader(self):
        transform = self._build_transform(is_train=True)
        if self.train_split == "trainval":
            dataset = ConcatDataset(
                [
                    JSONDataset(self.data_dir, self.dataset, "train", transform),
                    JSONDataset(self.data_dir, self.dataset, "val", transform),
                ]
            )
        else:
            dataset = JSONDataset(
                self.data_dir, self.dataset, self.train_split, transform
            )
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        transform = self._build_transform(is_train=False)
        dataset = JSONDataset(self.data_dir, self.dataset, self.val_split, transform)
        dataloader = (
            DataLoader(
                dataset,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
            ),
        )
        return dataloader

    def test_dataloader(self):
        transform = self._build_transform(is_train=False)
        dataset = JSONDataset(self.data_dir, self.dataset, "test", transform)
        dataloader = (
            DataLoader(
                dataset,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
            ),
        )
        return dataloader

    def _build_transform(
        self, is_train, scale=[0.08, 1.0], ratio=[3.0 / 4.0, 4.0 / 3.0], hflip=0.5
    ):
        t = []
        if is_train:
            t.append(
                RandomResizedCropAndInterpolation(
                    self.input_size, scale=scale, ratio=ratio, interpolation="random"
                )
            )
            t.append(transforms.RandomHorizontalFlip(p=hflip))
        else:
            scale_size = int(math.floor(self.input_size / DEFAULT_CROP_PCT))
            t.append(
                transforms.Resize(scale_size, interpolation=Image.Resampling.BICUBIC)
            )
            t.append(transforms.CenterCrop(224))

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(self.mean, self.std))
        return transforms.Compose(t)


class JSONDataset(Dataset):
    def __init__(self, data_dir, dataset, split, transform):
        self._split = split
        self.dataset = dataset
        self.data_dir = data_dir
        self.data_percentage = 1.0
        self._construct_imdb()
        self.transform = transform

    def get_anno(self):
        anno_path = os.path.join(self.data_dir, self.dataset, f"{self._split}.json")
        if "train" in self._split:
            if self.data_percentage < 1.0:
                anno_path = os.path.join(
                    self.data_dir,
                    self.dataset,
                    "{}_{}.json".format(self._split, self.data_percentage),
                )
        assert os.path.exists(anno_path), "{} dir not found".format(anno_path)

        with open(anno_path, "rb") as f:
            anno_data = json.load(f)
        return anno_data

    def get_imagedir(self):
        dataset_to_imagedir = {
            "cub": "images",
            "nabirds": "images",
            "oxfordflower": "",
            "stanfordcars": "",
            "stanforddogs": "Images",
        }
        return os.path.join(
            self.data_dir, self.dataset, dataset_to_imagedir[self.dataset]
        )

    def _construct_imdb(self):
        """Constructs the imdb."""

        img_dir = self.get_imagedir()
        assert os.path.exists(img_dir), "{} dir not found".format(img_dir)

        anno = self.get_anno()
        # Map class ids to contiguous ids
        self._class_ids = sorted(list(set(anno.values())))
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}

        # Construct the image db
        self._imdb = []
        for img_name, cls_id in anno.items():
            cont_id = self._class_id_cont_id[cls_id]
            im_path = os.path.join(img_dir, img_name)
            self._imdb.append({"im_path": im_path, "class": cont_id})

    def __getitem__(self, index):
        # Load the image
        sample = default_loader(self._imdb[index]["im_path"])
        sample = self.transform(sample)
        target = self._imdb[index]["class"]
        return sample, target

    def __len__(self):
        return len(self._imdb)
