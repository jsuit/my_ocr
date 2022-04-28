from typing import Dict
from omegaconf import DictConfig
import torch.nn as nn
import torch


class NumObjectsPred(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self._config = config
        self._n_obj_pred = self.build_module_module(config)

    @staticmethod
    def build_module_module(config):
        in_channels = config["heads"]["num_input_channels"]
        num_channels = max(in_channels // 2, 1)
        conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=num_channels, kernel_size=3
        )
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(
            in_channels=num_channels, out_channels=128, kernel_size=3, stride=2
        )
        conv2_d = nn.Conv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            dilation=3,
        )
        adaptive_pool_size = config["heads"]["num_objects_regression"]["adaptive_pool"]
        conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2)
        mish_inplace = nn.Mish(inplace=True)
        conv4 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=1)
        num_features = conv4.out_channels * adaptive_pool_size * adaptive_pool_size
        linear1 = nn.Linear(in_features=num_features, out_features=num_features // 2)
        linear2 = nn.Linear(
            in_features=num_features // 2, out_features=num_features // 4
        )
        linear3 = nn.Linear(in_features=num_features // 4, out_features=1024)
        linear4 = nn.Linear(in_features=1024, out_features=1, bias=False)

        return nn.Sequential(
            *[
                conv1,
                relu,
                conv2,
                mish_inplace,
                conv2_d,
                mish_inplace,
                conv3,
                relu,
                conv4,
                nn.AdaptiveMaxPool2d(adaptive_pool_size),
                nn.Flatten(start_dim=1),
                linear1,
                nn.Dropout(p=0.5, inplace=True),
                relu,
                linear2,
                mish_inplace,
                linear3,
                mish_inplace,
                linear4,
            ]
        )

    def forward(self, fpn_output: torch.Tensor) -> Dict[str, torch.Tensor]:

        return {
            "num_objects_regression": torch.exp(self._n_obj_pred(fpn_output)).squeeze(
                dim=-1
            )
        }
