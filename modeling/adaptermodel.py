from typing import Any, Optional
import torch
import torchmetrics
import torch.nn.functional as F
import timm
import pytorch_lightning as pl
from collections import OrderedDict
from hydra.utils import instantiate
from torch import nn
from torch.optim import AdamW, SGD

from .lr_scheduler import LinearWarmupCosineAnnealingLR
from adapter_plus import Adapter, AdapterBlock


class AdapterModel(pl.LightningModule):
    def __init__(
        self,
        cfg,
        img_size=224,
        num_classes=1000,
    ):
        super().__init__()
        self.cfg = cfg
        self.img_size = img_size
        self.num_classes = num_classes
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.label_smoothing = cfg.train.label_smoothing

        self.vit = timm.create_model(
            cfg.vit.model,
            adapter=True,
            pretrained=True,
            num_classes=num_classes,
            img_size=img_size,
            drop_path_rate=cfg.vit.drop_path,
            adapter_config=cfg.get("adapter", None),
            lora_config=cfg.get("lora", None),
            prompt_config=cfg.get("prompt", None),
        )

        if cfg.get("adapter", None) or cfg.get("lora", None) or cfg.get("prompt", None):
            if not cfg.vit.finetune:
                self.vit.requires_grad_(False)
            self.vit.head.requires_grad_(True)
            for m in self.vit.modules():
                if isinstance(m, Adapter):
                    m.requires_grad_(True)

                # enable fine-tuning of all ViT LayerNorm
                if cfg.train.train_ln:
                    if isinstance(m, nn.LayerNorm):
                        m.requires_grad_(True)

                if cfg.get("prompt", None):
                    if isinstance(m, AdapterBlock):
                        m.prompt.requires_grad_(True)

        if cfg.train.classifier_only:
            self.vit.requires_grad_(False)
            self.vit.head.requires_grad_(True)

        # save hyperparemeters to checkpoints
        self.save_hyperparameters()
        self.trainable_params = {
            "trainable_params": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
        }
        self.save_hyperparameters(self.trainable_params)

    def forward(self, x):
        x = self.vit(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y, label_smoothing=self.label_smoothing)
        self.train_acc(F.softmax(y_hat, -1), y)
        self.log(
            "train/accuracy",
            self.train_acc,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log("train/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val/loss", loss, add_dataloader_idx=False, sync_dist=True)
        self.val_acc(F.softmax(y_hat, -1), y)
        self.log(
            "val/accuracy",
            self.val_acc,
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        self.log(
            "hp_metric", self.val_acc, add_dataloader_idx=False, sync_dist=True
        )

    def configure_optimizers(self):
        parameters = add_weight_decay(
            self.vit, self.cfg.train.optimizer.weight_decay, self.vit.no_weight_decay()
        )

        optimizer = instantiate(self.cfg.train.optimizer, parameters)

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.cfg.train.warmup,
            warmup_start_lr=self.cfg.train.optimizer.lr
            / self.cfg.train.lr_start_factor,
            max_epochs=self.trainer.max_epochs,
            eta_min=self.cfg.train.optimizer.lr / self.cfg.train.lr_factor,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def lr_scheduler_step(self, scheduler, metric: Optional[Any]):
        scheduler.step(epoch=self.current_epoch)

    def on_save_checkpoint(self, checkpoint):
        old_dict = checkpoint["state_dict"]
        new_dict = OrderedDict()
        for k, v in old_dict.items():
            if "adapter" in k or "head" in k:
                new_dict[k] = v
        checkpoint["state_dict"] = new_dict
        assert sum(p.numel() for p in self.parameters() if p.requires_grad) == sum(
            p.numel() for p in new_dict.values()
        )

    def on_load_checkpoint(self, checkpoint):
        old_dict = checkpoint["state_dict"]
        new_dict = OrderedDict()
        for k, v in self.named_parameters():
            if v.requires_grad:
                new_dict[k] = old_dict[k]
            else:
                new_dict[k] = v
        checkpoint["state_dict"] = new_dict


# modified from timm.optim.optim_factory
def add_weight_decay(model, weight_decay=1e-5, skip_list=(), exclude_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad or name in exclude_list:
            continue  # frozen weights
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or name.endswith(".scaling")
            or name in skip_list
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]
