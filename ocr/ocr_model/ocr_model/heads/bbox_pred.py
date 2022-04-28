from typing import Dict
from omegaconf import DictConfig
import torch.nn as nn
import torch
from torch import exp as torch_exp


class ExpModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor):
        return torch_exp(x)


class BBoxPred(nn.Module):
    def __init__(self, config: DictConfig, image_size: int) -> None:
        super().__init__()
        self._config = config
        # self._cls = self.build_module_module(config, image_size=image_size, clss=True)
        self._regressor = self.build_module_module(config, image_size=image_size)

    @staticmethod
    def build_module_module(config, image_size):
        in_channels = config["heads"]["num_input_channels"]
        num_channels = max(in_channels // 2, 1.0)
        conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=num_channels, kernel_size=3, padding=1
        )
        mish = nn.Mish(inplace=True)
        conv2 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            padding=1,
        )

        convT = nn.ConvTranspose2d(
            in_channels=num_channels, out_channels=num_channels, kernel_size=2, stride=2
        )
        up1 = nn.UpsamplingBilinear2d(size=image_size // 2)
        conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            padding=1,
        )
        up2 = nn.UpsamplingBilinear2d(size=image_size)
        conv3 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=4,
            kernel_size=1,
        )

        return nn.Sequential(
            *[
                conv1,
                mish,
                conv2,
                mish,
                convT,
                mish,
                up1,
                conv,
                conv3,
                ExpModule(),
                up2,
            ]
        )

    def forward(self, fpn_output: torch.Tensor) -> Dict[str, torch.Tensor]:

        return {"bbox_pred": self._regressor(fpn_output)}
