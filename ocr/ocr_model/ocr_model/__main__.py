from typing import Dict
import hydra
from ocr_model.dataloaders.utils import load_dataset
from omegaconf import DictConfig, OmegaConf
from ocr_model.model_defs.adapt_binary import AdaptBinarization
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import os
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@hydra.main(config_path="../configs", config_name="default.yaml")
def main(config: DictConfig):

    loggers_info = config.logger
    loggers = []
    mlfow = None
    for logger_info in loggers_info.values():
        loggers.append(hydra.utils.instantiate(logger_info))
        if logger_info["_target_"] == "pytorch_lightning.loggers.MLFlowLogger":
            mlflow = loggers[-1]

    num_devices = config.trainer.devices
    strategy = None
    if num_devices == -1:
        num_devices = torch.cuda.device_count()
    if num_devices > 1:
        strategy = config.trainer.strategy.multi
    checkpoint = hydra.utils.instantiate(config.callbacks.checkpoint)

    trainer = Trainer(
        logger=loggers,
        accelerator=config.trainer.accelerator,
        devices=num_devices,
        strategy=strategy,
        callbacks=[checkpoint],
        max_epochs=config.optim.stop,
        num_sanity_val_steps=4,
        log_every_n_steps=32,
        gradient_clip_val=1.0,
    )
    keyword_args = {
        "model": config["model"],
        "optim": config["optim"],
        "losses": config["losses"],
    }
    if "image_size" in config["data"]:
        keyword_args.update({"image_size": config["data"]["image_size"]})

    clss = hydra.utils.get_class(config.model._target_)
    model = clss(**keyword_args)

    if mlflow:
        mlflow.log_hyperparams(config)
        mlflow.experiment.log_param(mlflow._run_id, "run_id", mlflow._run_id)
    tds = load_dataset(config, train=True)
    valds = load_dataset(config, train=False)
    if config.data.num_workers == "num_cpus":
        num_workers = min(os.cpu_count(), 32)
    else:
        num_workers = config.data.num_workers
    dls = {
        "train_dataloaders": DataLoader(
            dataset=tds,
            batch_size=config.data.batch_size,
            pin_memory=True,
            shuffle=True,
            collate_fn=tds.batch,
            num_workers=num_workers,
        ),
        "val_dataloaders": DataLoader(
            dataset=valds,
            batch_size=1,
            pin_memory=True,
            shuffle=False,
            num_workers=4,
            collate_fn=valds.batch,
        ),
    }
    trainer.fit(model=model, **dls)


if __name__ == "__main__":
    main()
