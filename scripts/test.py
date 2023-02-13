import argparse
import json
import os
from typing import Callable, Tuple

import albumentations as A
import cv2
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2

from sky_segmentation.data.datamodules import SkySegmentationDataModule
from sky_segmentation.modules.networks import SkySegmentationModel

PROJECT_NAME = "sky_segmentation"
WANDB_PATH = "artifacts/sky_segmentation"
MODEL_DIR = "artifacts/models/"
RESULTS_DIR = "artifacts/results"
DATA_PATH = "artifacts/data/"
METRIC_TO_MONITOR = "valid/dataset_iou"


def get_transforms(encoder_name: str) -> Tuple[Callable, Callable, Callable]:
    params = smp.encoders.get_preprocessing_params(encoder_name)
    train_transform = A.Compose(
        [
            A.LongestMaxSize(512),
            A.PadIfNeeded(min_height=512, min_width=512),
            A.Normalize(mean=params["mean"], std=params["std"]),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.LongestMaxSize(512),
            A.PadIfNeeded(
                min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT
            ),
            A.Normalize(mean=params["mean"], std=params["std"]),
            ToTensorV2(),
        ]
    )

    test_transform = A.Compose(
        [
            A.PadIfNeeded(
                pad_height_divisor=32,
                pad_width_divisor=32,
                min_height=None,
                min_width=None,
                position="top_left",
                border_mode=0,
                value=0,
                always_apply=False,
                p=1.0,
            ),
            A.Normalize(mean=params["mean"], std=params["std"]),
            ToTensorV2(),
        ]
    )

    return train_transform, val_transform, test_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", type=str, default="unet")
    parser.add_argument("--encoder_name", type=str, default="resnet18")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--out_classes", type=int, default=1)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # define logger
    experiment_name = "sky_segmentation-2023-02-13-20-59-24"
    experiment_dir = os.path.join(RESULTS_DIR, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # split data
    splits_path = os.path.join(DATA_PATH, "metadata", "splits.json")

    # define transforms
    train_transform, val_transform, test_transform = get_transforms(args.encoder_name)

    # define datamodule
    data_module = SkySegmentationDataModule(
        data_dir="artifacts/data/ADEChallengeData2016",
        splits_path=splits_path,
        sky_class=3,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
    )

    # define model
    model = SkySegmentationModel.load_from_checkpoint(
        checkpoint_path=os.path.join(MODEL_DIR, experiment_name, "last.ckpt"),
        architecture=args.architecture,
        encoder_name=args.encoder_name,
        learning_rate=args.learning_rate,
        in_channels=args.in_channels,
        out_classes=args.out_classes,
    )

    # define the trainer
    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        detect_anomaly=True,
    )

    # test the model
    results = trainer.test(model, datamodule=data_module)
    # save results
    results_path = os.path.join(experiment_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
