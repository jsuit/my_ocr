import hydra
from ocr_model.dataloaders.utils import load_dataset
from omegaconf import DictConfig, OmegaConf
from ocr_model.model_defs.adapt_binary import AdaptBinarization
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import os
from hydra import compose, initialize, initialize_config_dir
from omegaconf import OmegaConf

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# @hydra.main(config_path="../configs", config_name="eval.yaml")
def eval(config: DictConfig):

    import mlflow

    rid = config.get("run_id")
    client = mlflow.tracking.MlflowClient(
        tracking_uri=config.logger.mlflow.tracking_uri
    )

    data = client.get_run(run_id=rid).to_dictionary()

    log_dir = f"{config.work_dir}/{data['data']['params']['log_dir']}"
    old_config = OmegaConf.load(log_dir + "/.hydra/config.yaml")
    exp_name = old_config.exp_name
    config.exp_name = config.exp_name + f"{exp_name}"
    if hasattr(old_config.model, "_target_"):

        model = hydra.utils.get_class(old_config.model._target_)
    else:
        model = AdaptBinarization
    checkpoints = f"{log_dir}/{data['data']['params']['callbacks/checkpoint/dirpath']}"
    config.exp_name = exp_name
    config.logger.mlflow.experiment_name = f"val/{exp_name}"
    files = os.listdir(checkpoints)
    loggers = []
    for logger_info in config.logger.values():
        loggers.append(hydra.utils.instantiate(logger_info))
    if config.data == "old_config":
        config.data = old_config.data

    for f in files:
        checkpoint = f"{checkpoints}{f}"

        model = model.load_from_checkpoint(
            checkpoint_path=checkpoint, map_location="cpu"
        )
        trainer = Trainer(
            logger=loggers,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            max_epochs=1,
        )

        valds = load_dataset(config, train=False)
        if config.data.num_workers == "num_cpus":
            num_workers = min(os.cpu_count(), 32)
        else:
            num_workers = old_config.data.num_workers

        viz_dataloader = DataLoader(
            dataset=valds,
            batch_size=1,
            pin_memory=True,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=valds.batch,
        )

        trainer.test(dataloaders=viz_dataloader, model=model, ckpt_path=checkpoint)


if __name__ == "__main__":
    from pathlib import Path

    cur_dir = Path.cwd()

    home_dir = cur_dir.parent
    config_path = str(home_dir / Path("configs"))

    with initialize_config_dir(config_dir=config_path, job_name="viz"):
        cfg = compose(
            config_name="eval",
            overrides=[
                "run_id=130733e3f3764bfd815091fce665ed0d",
                f"work_dir={str(home_dir)}",
            ],
        )
        print(OmegaConf.to_yaml(cfg))
    eval(cfg)
