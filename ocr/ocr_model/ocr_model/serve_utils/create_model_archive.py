from struct import pack
import hydra
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from model_archiver.model_packaging_utils import ModelExportUtils
from model_archiver.model_packaging import package_model
from pathlib import Path
import os


def archive(config):
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

    model = hydra.utils.get_class(old_config.model._target_)

    checkpoints = f"{log_dir}/{data['data']['params']['callbacks/checkpoint/dirpath']}"
    config.exp_name = exp_name
    config.logger.mlflow.experiment_name = f"val/{exp_name}"
    files = os.listdir(checkpoints)
    file = [f for f in files if not f.endswith(".ts")][0]
    checkpoint = f"{checkpoints}{file}"

    ts_location = checkpoint + ".ts"

    model = model.load_from_checkpoint(checkpoint_path=checkpoint, map_location="cpu")
    model.to_torchscript(file_path=ts_location, method="script")

    cwd = Path.cwd()
    parent = cwd.parent
    target = old_config.model._target_
    target = target.split(".")[1:-1]

    model_file = "/".join(target)
    model_file = parent.joinpath(Path(model_file))
    model_file = str(model_file) + ".py"
    args = dict(
        model_name=data["data"]["params"]["exp_name"],
        serialized_file=ts_location,
        model_file=model_file,
        runtime="python3",
        export_path=str(parent / "serving_models"),
        handler=str(cwd / "obj_detection_with_segs.py"),
        requirements_file=str(parent.parent / "requirements.txt"),
        version=0.01,
        extra_files={},
        force=False,
        archive_format="default",
    )
    args = OmegaConf.create(args)

    manifest = ModelExportUtils.generate_manifest_json(args)
    package_model(args, manifest=manifest)
    print(
        f"saved model archive of model {data['data']['params']['exp_name']} to {parent /'serving_models'}"
    )


if __name__ == "__main__":
    from pathlib import Path

    cur_dir = Path.cwd()

    home_dir = cur_dir.parent.parent
    config_path = str(home_dir / Path("configs"))

    with initialize_config_dir(config_dir=config_path, job_name="viz"):
        cfg = compose(
            config_name="eval",
            overrides=[
                "run_id=225e979d5d22410fb34b556a5b9a175d",
                f"work_dir={str(home_dir)}",
            ],
        )
    archive(cfg)
