import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` above is optional line to make environment more convenient
# should be placed at the top of each entry file
#
# main advantages:
# - allows you to keep all entry files in "src/" without installing project as a package
# - launching python file works no matter where is your current work dir
# - automatically loads environment variables from ".env" if exists
#
# how it works:
# - `setup_root()` above recursively searches for either ".git" or "pyproject.toml" in present
#   and parent dirs, to determine the project root dir
# - adds root dir to the PYTHONPATH (if `pythonpath=True`), so this file can be run from
#   any place without installing project as a package
# - sets PROJECT_ROOT environment variable which is used in "configs/paths/default.yaml"
#   to make all paths always relative to project root
# - loads environment variables from ".env" in root dir (if `dotenv=True`)
#
# you can remove `pyrootutils.setup_root(...)` if you:
# 1. either install project as a package or move each entry file to the project root dir
# 2. remove PROJECT_ROOT variable from paths in "configs/paths/default.yaml"
#
# https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from typing import List, Tuple

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def get_prediction_model(cfg: DictConfig) -> Tuple[LightningModule, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Get Trained Model")
    ckpt_model = model.load_from_checkpoint(
        checkpoint_path=cfg.ckpt_path,
        strict=False, train_loss=None, val_loss=None
    )

    return ckpt_model, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def hydra_prediction_model(cfg: DictConfig):
    return get_prediction_model(cfg)


if __name__ == "__main__":
    hydra_prediction_model()
