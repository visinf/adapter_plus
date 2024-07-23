import os
import hydra
import logging
import warnings

# ignore deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# disables tensorflow warnings regarding GPU support
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

from omegaconf import DictConfig
from hydra.utils import instantiate
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from modeling.adaptermodel import AdapterModel
from data.constants import NUM_CLASSES


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed, workers=True)
    # for VTAB datasets
    tf.random.set_seed(cfg.seed)

    model = AdapterModel(cfg, num_classes=NUM_CLASSES[cfg.data.dataset])

    logger = [
        TensorBoardLogger("logs", name="", version=""),
        CSVLogger("logs", name="", version=""),
    ]

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=10),
    ]
    if cfg.save_checkpoints:
        callbacks.append(ModelCheckpoint(save_last="link"))

    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        strategy="ddp_find_unused_parameters_false",
        accelerator="gpu",
        devices=cfg.gpus,
        precision="bf16",
        max_epochs=cfg.train.epochs,
        gradient_clip_val=cfg.train.gradient_clip_val,
        check_val_every_n_epoch=cfg.val_every_n_epoch,
        enable_checkpointing=cfg.save_checkpoints,
    )

    vit_cfg = model.vit.default_cfg
    data_model = instantiate(cfg.data, mean=vit_cfg["mean"], std=vit_cfg["std"])

    trainer.fit(model, datamodule=data_model)

    metrics = trainer.callback_metrics
    return metrics["val/accuracy"]


if __name__ == "__main__":
    log = logging.getLogger("pytorch_lightning")
    log.propagate = False
    log.setLevel(logging.WARNING)

    main()
