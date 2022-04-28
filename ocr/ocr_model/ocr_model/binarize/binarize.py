from typing import Dict
from torch import detach, imag
import torch.nn as nn
from math import log
import torch
from torch import sigmoid as torch_sigmoid


def _expand_to_image_size(mult: float):
    if mult <= 1:
        return 0, 0
    if mult < 2:
        return 0, mult
    num_2s = 0
    rem = 0
    num_2s = log(mult, 2)
    num2s_int = int(num_2s)
    if num2s_int == num_2s:
        rem == 0
    else:
        rem = 1
    return num2s_int, rem


class DiffBinarize(nn.Module):
    def __init__(
        self,
        image_size: int,
        mult_factor: float,
        input_channels: int,
        num_layers: int,
        thresh: float,
        use_centerness=True,
    ) -> None:
        super().__init__()
        if thresh == 0:
            thresh = 1.0
        self._thresh_const = torch.nn.parameter.Parameter(
            torch.Tensor([thresh]), requires_grad=False
        )

        double_times, rem = _expand_to_image_size(mult=mult_factor)
        upsamplers = []
        if not double_times:
            if rem:
                upsamplers.append(nn.Upsample(size=(image_size, image_size)))
            else:
                upsamplers.append(nn.Identity())
        else:
            for _ in range(double_times):
                upsamplers.append(
                    nn.ConvTranspose2d(
                        in_channels=input_channels // num_layers,
                        out_channels=input_channels // num_layers,
                        kernel_size=2,
                        stride=2,
                    )
                )
            if rem:
                upsamplers.append(nn.Upsample(size=(image_size, image_size)))
        self._binary = self._build_binarize(
            upsamplers=upsamplers, in_channels=input_channels, num_layers=num_layers
        )
        self._thresh = self._build_binarize(
            upsamplers=upsamplers, in_channels=input_channels, num_layers=num_layers
        )

    def _build_binarize(self, upsamplers, in_channels, num_layers):
        binary = nn.Sequential()
        binary.add_module(
            name=f"upsample_{0}",
            module=nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels // num_layers,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(in_channels // num_layers),
                nn.Mish(inplace=True),
            ),
        )
        for i in range(len(upsamplers) - 1):
            binary.add_module(
                name=f"upsample_{i+1}",
                module=nn.Sequential(
                    upsamplers[i],
                    nn.BatchNorm2d(in_channels // num_layers),
                    nn.Mish(inplace=True),
                ),
            )
        binary.add_module(
            name="upsample_sigmoid",
            module=nn.Sequential(
                upsamplers[-1],
                nn.Conv2d(
                    in_channels=in_channels // num_layers,
                    out_channels=1,
                    kernel_size=1,
                    bias=False,
                ),
            ),
        )
        return binary

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:

        return {
            "binary": torch_sigmoid(
                self._thresh_const * self._binary(features) - self._thresh(features)
            ),
        }
