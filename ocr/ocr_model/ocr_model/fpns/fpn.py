from typing import Any, List, Optional, Union
import torch.nn as nn
import torch
from math import log


@torch.jit.interface
class ModuleInterface(torch.nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pass


@torch.jit.interface
class Conv2dTInterface(torch.nn.Module):
    def forward(
        self, input: torch.Tensor, output_size: Optional[List[int]] = None
    ) -> torch.Tensor:
        pass


class BasicFPN(nn.Module):
    def __init__(
        self,
        in_channels: Union[List[int], int],
        scale: int,
        num_inner_channels=256,
        n_layers: Optional[int] = None,
    ) -> None:
        super().__init__()

        assert scale > 0, f"scale {scale} should be > 0"
        if isinstance(in_channels, int):
            assert (
                n_layers is not None and isinstance(n_layers, int) and n_layers > 0
            ), f"if input channels is an int, then we need the num of layers to be given and be > 0. Got {n_layers}"
            in_channels = [in_channels] * n_layers
        assert num_inner_channels > len(
            in_channels
        ), f"need the inner number of channels "
        self._up = self._build_up_sample(
            in_channels=num_inner_channels, layers=len(in_channels), scale=scale
        )

        self._same_num_chans = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=num_inner_channels,
                    kernel_size=1,
                    bias=False,
                )
                for in_channel in in_channels
            ]
        )
        self._output = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=num_inner_channels,
                    out_channels=num_inner_channels // len(in_channels),
                    kernel_size=1,
                    bias=False,
                )
                for _ in in_channels
            ]
        )
        self._upsamplers = self._build_output_same_size(len(in_channels), scale=scale)

        self.reset_params()

    def reset_params(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight.data)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1.0)
                module.bias.data.fill_(0.0)

    def _build_output_same_size(self, num_channels, scale):
        return nn.ModuleList(
            [
                nn.Upsample(scale_factor=int(pow(scale, i)), mode="nearest")
                for i in range(num_channels - 1, 0, -1)
            ]
        )

    def _build_up_sample(self, in_channels: int, layers: int, scale: int):
        convs_up = []
        for _ in range(layers - 1):
            convs_up.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=scale,
                    stride=scale,
                    bias=False,
                )
            )

        return nn.ModuleList(convs_up)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        r_same_num_chans = []

        for i in range(len(x) - 1, -1, -1):
            module: ModuleInterface = self._same_num_chans[i]
            tensor: torch.Tensor = module.forward(x[i])
            r_same_num_chans.append(tensor)
        outputs = [self._upsamplers[0](self._output[0](r_same_num_chans[0]))]
        prev: torch.Tensor = r_same_num_chans[0]
        for i in range(1, len(x) - 1):
            conv2dT: Conv2dTInterface = self._up[i - 1]
            up: ModuleInterface = self._upsamplers[i]
            conv: ModuleInterface = self._output[i]

            out = conv2dT.forward(prev, None) + r_same_num_chans[i]

            outputs.append(up.forward(conv.forward(out)))
            prev = out

        out = self._up[-1](prev) + r_same_num_chans[-1]
        outputs.append(self._output[-1](out))

        return torch.cat(outputs, dim=1)
