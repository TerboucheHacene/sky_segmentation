from typing import Dict

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch


class SkySegmentationModel(pl.LightningModule):
    def __init__(
        self,
        architecture: str,
        encoder_name: str,
        in_channels: int,
        out_classes: int,
        learning_rate: float,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = smp.create_model(
            architecture,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.learning_rate = learning_rate

    def forward(self, image) -> torch.Tensor:
        # normalize image here
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage) -> Dict[str, torch.Tensor]:

        image = batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        assert image.ndim == 4
        # Check that image dimensions are divisible by 32,
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        mask = batch["mask"]
        assert mask.ndim == 4
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        if stage == "test":
            # crop the mask to the original size
            original_image_size = batch["image_size"]
            logits_mask = logits_mask[
                :, :, : original_image_size[1], : original_image_size[0]
            ]

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways: dataset-wise and image-wise
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
        )

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage) -> None:
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}/per_image_iou": per_image_iou,
            f"{stage}/dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        results = self.shared_step(batch, "train")
        self.log(
            "train/loss",
            results["loss"],
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            logger=True,
        )
        tp, fp, fn, tn = results["tp"], results["fp"], results["fn"], results["tn"]
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )
        self.log(
            "train/per_image_iou",
            per_image_iou,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            logger=True,
        )
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        self.log(
            "train/dataset_iou",
            dataset_iou,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            logger=True,
        )
        return results["loss"]

    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs) -> None:
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs) -> None:
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:

        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
