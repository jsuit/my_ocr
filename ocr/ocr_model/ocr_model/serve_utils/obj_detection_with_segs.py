from typing import Dict
import torch
from torchvision import transforms
from torchvision import __version__ as torchvision_version
from packaging import version
from ts.torch_handler.vision_handler import VisionHandler
from torchvision.ops import nms

IMAGE_SIZE = 640


class ObjDetectionSegs(VisionHandler):
    image_processing = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def initialize(self, context):
        super().initialize(context)
        self._img_size = IMAGE_SIZE
        properties = context.system_properties
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )
        y, x = torch.meshgrid(
            torch.arange(0, self._img_size.shape[1]),
            torch.arange(0, self._img_size.shape[0]),
            indexing="ij",
        )
        y = y.pin_memory()
        x = x.pin_memory()
        self.y = y.to(device=self.device, non_blocking=True)
        self.x = x.to(device=self.device, non_blocking=True)

    def inference(self, data, *args, **kwargs) -> Dict[str, torch.Tensor]:
        return self.model.predict(data)

    def postprocess(self, data: Dict[str, torch.Tensor]):

        attn = data["binary"]
        attn = attn.permute(0, 2, 3, 1).squeeze(-1)
        indices = attn > 0.51
        bboxes = data["bbox_pred"]  # lrtb
        bboxes = bboxes.permute(0, 2, 3, 1)
        L = self.x - bboxes[..., 0] * bboxes.shape[1]
        R = bboxes[..., 1] * self._img_size + self.x
        T = self.y - bboxes[..., 2] * self._img_size[0]
        B = bboxes[..., 3] * self._img_size[0] + self.y
        bboxes = torch.stack((L, T, R, B), axis=-1)[indices.expand_as(bboxes)].view(
            -1, 4
        )

        bboxes = bboxes[indices].view(-1, 4)
        attn = attn[indices].squeeze()
        indices = nms(boxes=bboxes, scores=attn, iou_threshold=0.32)
        return [
            {"bbox": bboxs.tolist(), "scores": score.tolist()}
            for (bboxs, score) in zip(bboxes, attn)
        ]
