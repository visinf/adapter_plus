import os
import sys
import hydra
import logging
import warnings
import torch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# disables tensorflow warnings regarding GPU support
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from omegaconf import DictConfig
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from modeling.adaptermodel import AdapterModel
from data.constants import NUM_CLASSES


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    model = AdapterModel(cfg, num_classes=NUM_CLASSES[cfg.data.dataset])
    ckpt_path = os.path.join(HydraConfig.get().runtime.output_dir, "logs/checkpoints/last.ckpt")
    state_dict = torch.load(ckpt_path)["state_dict"]
    model.load_state_dict(state_dict, strict=False)

    trainer = Trainer(
        logger=False,
        callbacks=TQDMProgressBar(refresh_rate=10),
        strategy="ddp_find_unused_parameters_false",
        accelerator="gpu",
        devices=cfg.gpus,
    )

    vit_cfg = model.vit.default_cfg
    data_model = instantiate(cfg.data, mean=vit_cfg["mean"], std=vit_cfg["std"])

    trainer.validate(model, datamodule=data_model, verbose=False)

    metrics = trainer.callback_metrics
    acc = metrics["val/accuracy"].item()
    
    print(f"Dataset: {cfg.data.dataset}, seed: {cfg.seed}, accuracy: {acc*100:.2f}")

    if str(HydraConfig.get().mode) == "RunMode.MULTIRUN":    
        with open("../../eval.log", "a") as f:
            f.write(f"{cfg.data.dataset}\t{cfg.seed}\t{acc:.4f}\n")


if __name__ == "__main__":
    log = logging.getLogger("pytorch_lightning")
    log.propagate = False
    log.setLevel(logging.WARNING)

    # disable output file creation for eval 
    sys.argv.append("hydra.output_subdir=null")

    main()
