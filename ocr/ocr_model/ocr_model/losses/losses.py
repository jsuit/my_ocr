from turtle import forward
from typing import Dict, List, Optional, Tuple
from numpy import argsort
import torch
import torchvision
import torch.nn.functional as F


class Losses:
    def __init__(self, loss_infos: List[Dict], script=False) -> None:
        super().__init__()
        globals_ = globals()
        loss_dicts = {}
        for loss_dict in loss_infos:
            for loss_name, loss_kwargs in loss_dict.items():
                loss_module = globals_[loss_name](**loss_kwargs)
                if script:
                    loss_module = torch.jit.script(loss_module)
            loss_dicts[loss_name] = loss_module

        self._losses = torch.nn.ModuleDict(loss_dicts)


class IcdarLoss(Losses, torch.nn.Module):
    def __init__(self, losses: list[dict], script=True) -> None:

        super(IcdarLoss, self).__init__(losses, script=script)

    def forward(
        self,
        gt: Dict[str, torch.Tensor],
        pred: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        loss_dict = {}

        for loss_key, loss_fn in self._losses.items():

            loss_rv = loss_fn(gt, pred)
            _loss_key = loss_key
            for key, val in loss_rv.items():
                loss_key = (
                    f"val/{_loss_key}_{key}"
                    if key
                    else f"val/{_loss_key}"
                    if not self.training
                    else f"{_loss_key}_{key}"
                    if key
                    else f"val/{_loss_key}"
                )
                loss_dict[loss_key] = val

        # loss_dict["loss"] = sum([loss_value for loss_value in loss_dict.values()])
        return loss_dict


class NumObjectsReg(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, gt: Dict[str, torch.Tensor], pred: Dict[str, torch.Tensor]):
        # B, N bboxes, 4 -> B, N

        mask = gt["bboxes"].sum(dim=-1) > 0.0
        B = mask.sum(dim=-1).to(dtype=torch.float32)
        return {"n_obj_reg_loss": F.mse_loss(pred["num_objects_regression"], B)}


class BBoxLoss(torch.nn.Module):
    def __init__(self, neg_ratio: int = 3, weight: float = 1.0) -> None:
        super().__init__()
        self._neg_ratio = neg_ratio
        self._smooth_l1 = torch.nn.SmoothL1Loss(reduction="none")
        self._loss_mult = weight

    def forward(
        self,
        gt: Dict[str, torch.Tensor],
        pred: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        center = gt["centerness"]

        distance_deltas = gt["distance_deltas"]  # b, H, W, 4
        bbox_pred = pred["bbox_pred"].permute(0, 2, 3, 1)

        mask = (gt["gt_img"] * gt["gt_mask"]).permute(0, 2, 3, 1) * center
        delta_loss = (self._smooth_l1(bbox_pred, distance_deltas)) * mask

        Z = (mask > 1e-4).sum()

        return {
            "delta_loss": self._loss_mult * delta_loss.sum() / max(Z + 1e-4, 1.0),
        }


class FScore(torch.jit.ScriptModule):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, gt: Dict[str, torch.Tensor], pred: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        gt_image = gt["gt_img"]
        pred_image = pred["binary"]
        mask = gt["gt_mask"]
        assert (
            gt_image.ndim == pred_image.ndim
        ), f"gt_image ndims {gt_image.ndim} != predicated image ndims {pred_image.ndim}"

        if mask is None:
            mask = torch.ones_like(gt_image, device=gt_image.device)
        inter = (gt_image * pred_image * mask).sum()

        union = ((gt_image + pred_image) * mask).sum() + 1e-4

        return {"val/fscore": 2.0 * (inter / max(union, 1.0))}


class IOUMetric(torch.jit.ScriptModule):
    def __init__(
        self,
        image_size: List[int],
        train: bool = False,
        thresh: float = 0.51,
        max_objects: int = 1024,
    ) -> None:
        super().__init__()
        self._train = train
        self._thresh = thresh
        self._image_size: List[int] = (
            image_size if isinstance(image_size, list) else [image_size, image_size]
        )
        y, x = torch.meshgrid(
            torch.arange(0, image_size), torch.arange(0, image_size), indexing="ij"
        )
        self.y: torch.Tensor = y.unsqueeze(dim=0).unsqueeze(-1)
        self.x: torch.Tensor = x.unsqueeze(dim=0).unsqueeze(-1)
        self._max_objs = max_objects

    def forward(
        self,
        gt: Dict[str, torch.Tensor],
        pred: Dict[str, torch.Tensor],
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not self._train:

            bbox_pred = pred["bbox_pred"].permute(0, 2, 3, 1)  # B, H, W, 4
            # class_pred = pred["class"].permute(0, 2, 3, 1)
            # preds = torch.sigmoid(class_pred)
            preds = pred["binary"].permute(0, 2, 3, 1)

            # x,y, xmax, ymax
            probs, bbox_preds = self._get_bboxs(
                clss_probs=preds,
                deltas_lrtb=bbox_pred[..., :4],
            )
            gt_bboxes = gt["bboxes"].squeeze(dim=0)
            gt_bboxes = gt_bboxes[gt_bboxes.sum(dim=-1) != 0]

            if probs is not None:
                # bbox_preds = N, 4

                sorted_indices = probs.argsort(dim=-1, descending=True)
                bbox_preds, probs = bbox_preds[sorted_indices, :], probs[sorted_indices]
                bbox_preds, probs = (
                    bbox_preds[: self._max_objs],
                    probs[: self._max_objs],
                )
                indices = torchvision.ops.nms(
                    bbox_preds, scores=probs, iou_threshold=0.32
                )
                bbox_preds, probs = bbox_preds[indices], probs[indices]
                # num_pos = int(
                #    round(float(pred["num_objects_regression"].squeeze().cpu()))
                # )
                # indices = indices[:num_pos]

                if gt_bboxes.numel():
                    if not bbox_preds.numel():
                        bbox_preds = torch.zeros_like(
                            gt_bboxes, device=gt_bboxes.device
                        )

                    iou_matrix = torchvision.ops.box_iou(bbox_preds, gt_bboxes)
                    iou = iou_matrix.max(dim=-1).values.mean()

                    return {
                        "bbox_preds": bbox_preds,
                        "probs": probs,
                        "iou": iou,
                        "class": preds.squeeze(),
                    }
                else:
                    if not bbox_preds.numel():
                        return {
                            "iou": torch.Tensor([0.0]),
                            "class": torch.rand(0, 0, 1),
                        }
                    else:
                        return {
                            "iou": torch.Tensor([0.0]),
                            "class": preds.squeeze(),
                        }

            else:
                if not gt_bboxes.numel():
                    # no bounding box predictions but not gt_bboxes
                    return {"iou": torch.Tensor([0.0]), "class": torch.rand(0, 0, 1)}
            #  either no gt boxes but we think there is
            return {
                "iou": torch.Tensor([0.0]),
                "class": preds.squeeze(),
            }
        else:
            return None

    def _get_bboxs(
        self, deltas_lrtb, clss_probs
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:

        indices_th = clss_probs >= self._thresh
        clss_probs = clss_probs[indices_th]

        if clss_probs.numel():
            clss_probs = clss_probs.view(-1)

            self.y = self.y.to(device=clss_probs.device)
            self.x = self.x.to(device=clss_probs.device)
            deltas_lrtb[..., 0:2] *= self._image_size[1]
            deltas_lrtb[..., 2:4] *= self._image_size[0]
            deltas_lrtb[..., 0:1] = self.x - deltas_lrtb[..., 0:1]
            deltas_lrtb[..., 1:2] = deltas_lrtb[..., 1:2] + self.x

            deltas_lrtb[..., 2:3] = self.y - deltas_lrtb[..., 2:3]
            deltas_lrtb[..., 3:4] = deltas_lrtb[..., 3:4] + self.y

            deltas_lrtb = torch.clip(
                deltas_lrtb, min=0.0, max=max(deltas_lrtb.shape[1:3])
            )

            indices_th = indices_th.expand(-1, -1, -1, 4)

            pos_deltas = deltas_lrtb[indices_th].view(-1, 4)
            pos_deltas = torch.stack(
                (
                    pos_deltas[..., 0],
                    pos_deltas[..., 2],
                    pos_deltas[..., 1],
                    pos_deltas[..., 3],
                ),
                dim=-1,
            )
            # format: left, top, right, bottom

            pos = clss_probs
        else:
            pos, pos_deltas = None, None
        return (pos, pos_deltas)


class DiceLoss(torch.nn.Module):
    def __init__(self, eps=1e-4, weight=1.0) -> None:
        super().__init__()
        self._eps = eps
        self._weight = weight

    def forward(
        self,
        gt_image_dict: Dict[str, torch.Tensor],
        pred_image_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        gt_image = gt_image_dict["centerness"]
        pred_image = pred_image_dict["binary"].permute(0, 2, 3, 1)
        mask = gt_image_dict["gt_mask"].permute(0, 2, 3, 1)
        assert (
            gt_image.ndim == pred_image.ndim
        ), f"gt_image ndims {gt_image.ndim} != predicated image ndims {pred_image.ndim}"

        if mask is None:
            mask = torch.ones_like(gt_image, device=gt_image.device)

        inter = (gt_image * pred_image * mask).sum()

        union = ((gt_image + pred_image) * mask).sum()

        return {"": self._weight * (1.0 - 2.0 * (inter / union))}


class topkBCE(torch.nn.Module):
    def __init__(self, neg_ratio=3.0, eps=1e-4, weight=1.0) -> None:
        super().__init__()
        self._neg_ratio = neg_ratio
        self._eps = eps
        self._bce = torch.nn.BCELoss(reduction="none")
        self._weight = weight

    def forward(self, gt: Dict[str, torch.Tensor], pred: Dict[str, torch.Tensor]):
        gt_labels = gt["centerness"]
        pred_labels = pred["binary"]

        if gt_labels.ndim == 4:
            gt_labels = gt_labels.squeeze(-1)
        if pred_labels.ndim == 4:
            pred_labels = pred_labels.squeeze(1)
        pred_labels = pred_labels.view(pred_labels.shape[0], -1)
        gt_labels = gt_labels.view(gt_labels.shape[0], -1)
        pos_mask = gt_labels >= 0.90

        mask = (gt_labels >= 0.80) * (gt_labels < 0.90)
        ignore = mask.sum()
        mask = 1.0 - mask.to(dtype=pred_labels.dtype)

        loss = self._bce(pred_labels, gt_labels)

        positive_loss = loss * mask
        positive_loss = positive_loss[pos_mask]
        pos = positive_loss.numel()
        neg_mask = (1.0 - (pos_mask * 1.0)) * mask
        neg_loss = loss * (neg_mask)
        neg = gt_labels.numel() - pos - ignore.item()

        topk_neg_losses, _ = torch.topk(
            neg_loss.view(-1),
            k=min(max(pos, 1) * self._neg_ratio, int(neg)),
            dim=-1,
        )

        return (
            self._weight
            * (self._neg_ratio * positive_loss.sum() + topk_neg_losses.sum())
            / (pos + topk_neg_losses.numel() + self._eps)
        )
