import json
import os
import pathlib
from typing import Callable, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from sky_segmentation.data.datasets import SkySegmentationDataset, get_data_splits


class SkySegmentationDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        splits_path: pathlib.Path,
        sky_class: int,
        batch_size: int,
        num_workers: int,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        pin_memory: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.sky_class = sky_class
        self.splits_path = splits_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def prepare_data(self) -> None:
        # check if data_dir is empty
        if os.path.exists(self.splits_path):
            # load splits from disk
            with open(self.splits_path, "r") as f:
                self.splits = json.load(f)
        else:
            splits = get_data_splits(
                train_image_root_path=os.path.join(
                    self.data_dir, "annotations", "training"
                ),
                test_image_root_path=os.path.join(
                    self.data_dir, "annotations", "validation"
                ),
                sky_class=self.sky_class,
                train_split=0.9,
            )
            # save splits to disk
            with open(self.splits_path, "w") as f:
                json.dump(splits, f)
            self.splits = splits

        # check train, validation and test splits are not empty
        if len(self.splits["train"]) == 0:
            raise ValueError("Train split is empty")
        if len(self.splits["validation"]) == 0:
            raise ValueError("Validation split is empty")
        if len(self.splits["test"]) == 0:
            raise ValueError("Test split is empty")

        # print the number of images in each split
        print(f"Number of training images: {len(self.splits['train'])}")
        print(f"Number of validation images: {len(self.splits['validation'])}")
        print(f"Number of test images: {len(self.splits['test'])}")

    def setup(self, stage=None) -> None:
        self.train_dataset = SkySegmentationDataset(
            image_root_path=os.path.join(self.data_dir, "images", "training"),
            segmentation_root_path=os.path.join(
                self.data_dir, "annotations", "training"
            ),
            images=self.splits["train"],
            sky_class=self.sky_class,
            stage="train",
            transform=self.train_transform,
        )
        self.val_dataset = SkySegmentationDataset(
            image_root_path=os.path.join(self.data_dir, "images", "training"),
            segmentation_root_path=os.path.join(
                self.data_dir, "annotations", "training"
            ),
            images=self.splits["validation"],
            sky_class=self.sky_class,
            stage="val",
            transform=self.val_transform,
        )
        self.test_dataset = SkySegmentationDataset(
            image_root_path=os.path.join(self.data_dir, "images", "validation"),
            segmentation_root_path=os.path.join(
                self.data_dir, "annotations", "validation"
            ),
            images=self.splits["test"],
            sky_class=self.sky_class,
            stage="test",
            transform=self.test_transform,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
