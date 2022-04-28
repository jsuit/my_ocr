from random import random
from typing import Dict
import hydra
from pytorch_lightning import LightningModule
from ocr_model import backbones
import torch
from ocr_model.fpns.fpn import BasicFPN
from ocr_model.binarize import DiffBinarize
from ocr_model.heads.bbox_pred import BBoxPred
from ocr_model.heads.num_objects import NumObjectsPred
from math import pow
from torch import optim
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ocr_model.losses.losses import FScore, IOUMetric
import torchvision

mean = torch.Tensor((0.485, 0.456, 0.406))
std = torch.Tensor((0.229, 0.224, 0.225))


class AdaptBinarization(LightningModule):
    def __init__(
        self, *, model: dict, optim: dict, losses: dict, image_size: int
    ) -> None:
        super(AdaptBinarization, self).__init__()

        backbone = model["trunk"]["name"]

        self._trunk = backbones.get_backbone(backbone)
        # 224: torch.Size([1, 64, 56, 56]), torch.Size([1, 128, 28, 28]), torch.Size([1, 256, 14, 14]), torch.Size([1, 512, 7, 7])]
        with torch.no_grad():
            x = self._trunk(torch.rand(1, 3, image_size, image_size))

        if isinstance(x, (tuple, list)):
            x = max([max(x_.shape[-2:]) for x_ in x])
        else:
            x = max(x[-2:])
        fpn = model["mid"]["fpn"]
        self._fpn = BasicFPN(
            in_channels=fpn["in_channels"],
            scale=fpn["stride"],
            num_inner_channels=fpn["num_inner_channels"],
        )
        # self._fpn(self._trunk(torch.rand(1, 3, image_size, image_size)))
        mult_to_image = image_size / x
        self._mult_to_image = mult_to_image
        self._binarize = DiffBinarize(
            image_size=image_size,
            mult_factor=mult_to_image,
            input_channels=fpn["num_inner_channels"],
            num_layers=len(model["binary"]["in_channels"]),
            thresh=model["binary"]["adapt_constant"],
        )
        self._n_obj_reg = (
            NumObjectsPred(config=model)
            if "num_objects_regression" in model["heads"]
            else None
        )

        self._bbox_pred = BBoxPred(config=model, image_size=image_size)

        self.criterion = hydra.utils.instantiate(config=losses)
        self._iou_metric = IOUMetric(train=False, image_size=image_size)
        self._fscore = FScore()
        self.save_hyperparameters()

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        fpn_output = self._fpn(self._trunk(image))
        pred = self._binarize(fpn_output)
        pred.update(self._bbox_pred(fpn_output))
        if self._n_obj_reg is not None:
            pred.update(self._n_obj_reg(fpn_output))
        return pred

    def training_step(self, batch: Dict[str, torch.Tensor]):
        pred = self(batch["image"])

        losses = self.criterion(gt=batch, pred=pred)
        losses["loss"] = sum([loss for loss in losses.values()])
        fscore_dict = self._fscore(batch, pred)
        losses.update(fscore_dict)
        self.log_dict(
            losses,
            prog_bar=True,
            logger=True,
            rank_zero_only=True,
            on_epoch=True,
        )

        return losses

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        pred = self(batch["image"])

        losses = self.criterion(gt=batch, pred=pred)
        iou_metric_dict = self._iou_metric(gt=batch, pred=pred)
        losses.update(self._fscore(gt=batch, pred=pred))
        if iou_metric_dict["class"].numel():
            losses.update({"iou": iou_metric_dict["iou"]})
        for loss in losses.values():
            if isinstance(loss, torch.Tensor) and torch.isnan(loss):
                import pdb

                pdb.set_trace()

        monitor_loss = losses["val/DiceLoss"] + losses["val/BBoxLoss_delta_loss"]
        losses.update({"val/monitor_loss": monitor_loss})
        self.log_dict(
            losses,
            prog_bar=True,
            logger=True,
            rank_zero_only=True,
            on_epoch=True,
        )

        # probs = pred["binary"].squeeze(dim=0).permute(1, 2, 0).cpu().numpy()
        # probs = iou_metric_dict["class"].cpu().numpy()
        if random() <= 0.08:
            image = batch["image"].cpu()[0].permute(1, 2, 0).numpy()
            binary = pred["binary"].cpu()[0].permute(1, 2, 0).numpy()
            binary = np.repeat(
                binary, repeats=image.shape[-1] - binary.shape[-1] + 1, axis=-1
            )
            exp = self.logger.experiment

            exp.log_image(
                self.logger.run_id,
                image=np.concatenate(
                    (image, binary),
                    axis=1,
                ),
                artifact_file=f"{self.current_epoch}_{batch_idx}.png",
            )

    def predict(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self(data)

    def test_step(self, data_dict: Dict[str, torch.Tensor], batch_index, **kwargs):
        pred = self(data_dict["image"])

        image = data_dict["image"][0].permute(1, 2, 0)

        de_norm_image = image * std.to(device=image.device) + mean.to(
            device=image.device
        )
        bbox_pred = pred["bbox_pred"].permute(0, 2, 3, 1)  # B, H, W, 4

        bbox_pred = bbox_pred[0]  # H, W, 4
        binary = pred["binary"][0].permute(1, 2, 0)
        indices = binary >= 0.8
        binary = binary[indices]
        gt_bboxes = data_dict["bboxes"][0].cpu().numpy()
        if binary.size:
            y, x = torch.meshgrid(
                torch.arange(0, image.shape[1]),
                torch.arange(0, image.shape[0]),
                indexing="ij",
            )

            y = y.clone().pin_memory()
            y = y.to(device=image.device)
            x = x.clone().pin_memory()
            x = x.to(device=image.device)
            L = x - bbox_pred[..., 0] * image.shape[1]
            R = bbox_pred[..., 1] * image.shape[1] + x
            T = y - bbox_pred[..., 2] * image.shape[0]
            B = bbox_pred[..., 3] * image.shape[0] + y
            bboxes = torch.stack((L, T, R, B), axis=-1)[
                indices.expand_as(bbox_pred)
            ].view(-1, 4)
            indices = torchvision.ops.nms(
                boxes=bboxes, scores=binary, iou_threshold=0.08
            )

            bboxes = bboxes[indices]
            binary = binary[indices]
            if bboxes.numel():

                bboxes = bboxes.cpu().numpy()
                binary = binary.cpu().numpy()

            de_norm_image = np.ascontiguousarray(
                (de_norm_image * 255).cpu().numpy(), dtype=np.uint8
            )

            for bbox, gt_bbox in zip(bboxes, gt_bboxes):

                cv2.rectangle(
                    de_norm_image,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    (255, 0, 0),
                    2,
                )
                cv2.rectangle(
                    de_norm_image,
                    (int(gt_bbox[0]), int(gt_bbox[1])),
                    (int(gt_bbox[2]), int(gt_bbox[3])),
                    (0, 255, 0),
                    2,
                )

            plt.imshow(de_norm_image)
            plt.show()
            pred_binary = pred["binary"][0].squeeze().cpu()
            gt_img = data_dict["gt_img"] * data_dict["gt_mask"]
            gt_img = gt_img.squeeze().cpu()
            pred_gt_img = torch.cat((pred_binary, gt_img), dim=-1).numpy()
            plt.imshow(pred_gt_img)
            plt.show()
            center = data_dict["centerness"].squeeze().cpu()
            plt.imshow(center)
            plt.show()

    def optimizer_zero_grad(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_idx: int,
    ):
        optimizer.zero_grad(set_to_none=True)

    def configure_optimizers(self):

        optim_name = self.hparams["optim"]["optimizer"]["name"]
        optim_params = {
            key: val
            for key, val in self.hparams["optim"]["optimizer"].items()
            if key != "name"
        }

        optimizer = getattr(optim, optim_name)
        optimizer = optimizer(
            params=filter(lambda p: p.requires_grad, self.parameters()),
            **optim_params,
        )

        def func(epoch):
            return pow(
                1.0 - (epoch / float(self.hparams["optim"]["stop"] + 1)),
                sched_params["decay"]["factor"],
            )

        sched_params = self.hparams["optim"]["scheduler"]
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=func)

        return [optimizer], [lr_scheduler]
