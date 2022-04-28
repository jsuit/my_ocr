from matplotlib.pyplot import xlim
from torchvision.ops.deform_conv import DeformConv2d
import torch.nn as nn
import torch


class ModulatedDeformConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(ModulatedDeformConv2d, self).__init__()
        self._deform_conv2d = DeformConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.offset_conv_mask = nn.Conv2d(
            in_channels=in_channels,
            out_channels=groups * 3 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset_mask = self.offset_conv_mask(x)

        off0, off1, mask = torch.chunk(offset_mask, 3, dim=1)

        offset = torch.cat((off0, off1), dim=1)
        return self._deform_conv2d(input=x, offset=offset, mask=self.sigmoid(mask))
