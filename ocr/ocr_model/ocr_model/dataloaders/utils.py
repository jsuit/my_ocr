from omegaconf import DictConfig
import importlib


def load_dataset(config: DictConfig, train: bool):
    class_name = config.data._target_.split(".")[-1]
    module_name = ".".join(config.data._target_.split(".")[:-1])
    return getattr(importlib.import_module(module_name), class_name)(config, train)
