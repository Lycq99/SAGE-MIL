"""Unified SAGE-MIL training entry point."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from sage_mil.model import SemanticTransMIL
from sage_mil.data import FeatureDataModule
from sage_mil.losses import FocalLoss, true_class_margin, verification_hinge, linear_warmup_weight
from sage_mil.optim.lookahead_radam import RAdam, Lookahead


def probabilities(bag, inst, num_classes: int, bag_weight: float = 0.5, temperature: float = 1.0):
    if num_classes <= 2:
        pb = torch.sigmoid(bag / temperature)
        pi = torch.sigmoid(inst / temperature)
    else:
        pb = torch.softmax(bag / temperature, dim=-1)
        pi = torch.softmax(inst / temperature, dim=-1)
    return bag_weight * pb + (1.0 - bag_weight) * pi


def metric_bundle(y, p, num_classes: int, threshold: float = 0.5, multiclass_average: str = "macro"):
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    if num_classes <= 2:
        p1 = p.reshape(-1)
        pred = (p1 >= threshold).astype(int)
        return {
            "auc": roc_auc_score(y, p1),
            "acc": accuracy_score(y, pred),
            "f1": f1_score(y, pred, zero_division=0),
        }
    pred = p.argmax(axis=1)
    return {
        "auc": roc_auc_score(y, p, multi_class="ovr", average=multiclass_average),
        "acc": accuracy_score(y, pred),
        "f1": f1_score(y, pred, average=multiclass_average, zero_division=0),
    }


class LitSAGEMIL(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        m, d, task = cfg["model"], cfg["data"], cfg["task"]
        self.num_classes = int(task["num_classes"])
        self.save_hyperparameters({"config": cfg})
        self.model = SemanticTransMIL(
            input_dim=m["input_dim"],
            model_dim=m["model_dim"],
            num_classes=self.num_classes,
            anchors_path=d["anchors_path"],
            proxy_path=d["proxy_path"],
            proxy_num_prototypes=d.get("proxy_num_prototypes"),
            train_key_patches=m["train_key_patches"],
            lambda_env=m["lambda_env"],
            correction_mask=m["correction_mask"],
            topk_ratio=m["topk_ratio"],
            semantic_heads=m.get("semantic_heads", 1),
            proxy_temperature=m.get("proxy_temperature", 0.10),
            semantic_logit_scale_init=m.get("semantic_logit_scale_init", 1.0),
            sinkhorn_epsilon=m["sinkhorn_epsilon"],
            sinkhorn_iterations=m["sinkhorn_iterations"],
            target_norm_matching=m["target_norm_matching"],
            norm_match_clip=tuple(m["norm_match_clip"]),
        )
        self.criterion = FocalLoss(
            alpha=m.get("focal_alpha", 0.25),
            gamma=m.get("focal_gamma", 1.9),
            num_classes=self.num_classes,
        )
        self.best_val_threshold = 0.5
        self._val = []
        self._test = []

    def training_step(self, batch, batch_idx):
        x, coords, y = batch
        bag, inst, orth, cf, tau, _ = self.model(
            x,
            coords,
            training=True,
            current_epoch=self.current_epoch,
            max_epochs=self.trainer.max_epochs,
        )
        y_flat = y.reshape(-1)
        l_cls = self.criterion(bag, y_flat) + self.criterion(inst, y_flat)
        m_fact = true_class_margin(bag, y_flat, self.num_classes)
        m_cf = true_class_margin(cf, y_flat, self.num_classes)
        l_verify = verification_hinge(m_fact, m_cf, tau, self.cfg["model"]["gamma_margin"])
        w = linear_warmup_weight(
            self.cfg["model"]["lambda_verify"],
            self.current_epoch,
            self.cfg["model"]["verify_warmup_epochs"],
        )
        loss = l_cls + self.cfg["model"]["lambda_orth"] * orth + w * l_verify
        self.log_dict(
            {"train_loss": loss, "train_verify": l_verify, "train_orth": orth},
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def _prob(self, bag, inst):
        m = self.cfg["model"]
        return probabilities(
            bag,
            inst,
            self.num_classes,
            m.get("eval_bag_weight", 0.5),
            m.get("eval_temperature", 1.0),
        )

    def validation_step(self, batch, batch_idx):
        x, c, y = batch
        bag, inst, *_ = self.model(x, c, training=False)
        loss = self.criterion(bag, y.reshape(-1))
        self._val.append((self._prob(bag, inst).detach().cpu(), y.detach().cpu(), loss.detach().cpu()))

    def on_validation_epoch_end(self):
        if not self._val:
            return
        p = torch.cat([a for a, _, _ in self._val]).numpy()
        y = torch.cat([b for _, b, _ in self._val]).numpy().astype(int)
        val_loss = float(torch.stack([c for _, _, c in self._val]).mean())
        if self.num_classes <= 2:
            p1 = p.reshape(-1)
            best = (-1.0, 0.5)
            for th in np.arange(0.10, 0.901, 0.05):
                f = f1_score(y, p1 >= th, zero_division=0)
                if f > best[0]:
                    best = (float(f), float(th))
            self.best_val_threshold = best[1]
            metrics = metric_bundle(y, p1, 2, self.best_val_threshold)
        else:
            avg = self.cfg["task"].get("multiclass_average", "macro")
            metrics = metric_bundle(y, p, self.num_classes, multiclass_average=avg)
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_auc", metrics["auc"], prog_bar=True)
        self.log("val_f1", metrics["f1"])
        self.log("val_acc", metrics["acc"])
        self._val.clear()

    def test_step(self, batch, batch_idx):
        x, c, y = batch
        bag, inst, *_ = self.model(x, c, training=False)
        self._test.append((self._prob(bag, inst).detach().cpu(), y.detach().cpu()))

    def on_test_epoch_end(self):
        if not self._test:
            return
        p = torch.cat([a for a, _ in self._test]).numpy()
        y = torch.cat([b for _, b in self._test]).numpy().astype(int)
        avg = self.cfg["task"].get("multiclass_average", "macro")
        metrics = metric_bundle(y, p, self.num_classes, self.best_val_threshold, avg)
        for k, v in metrics.items():
            self.log(f"test_{k}", v)
        print({**metrics, "threshold": self.best_val_threshold if self.num_classes <= 2 else None})
        self._test.clear()

    def on_save_checkpoint(self, checkpoint):
        checkpoint["best_val_threshold"] = float(self.best_val_threshold)

    def on_load_checkpoint(self, checkpoint):
        self.best_val_threshold = float(checkpoint.get("best_val_threshold", 0.5))

    def configure_optimizers(self):
        o = self.cfg["optimizer"]
        base = RAdam(self.model.parameters(), lr=o["lr"], weight_decay=o["weight_decay"])
        return Lookahead(base, o.get("lookahead_alpha", 0.5), o.get("lookahead_k", 6))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(args.seed if args.seed is not None else cfg["general"].get("seed", 2022))
    pl.seed_everything(seed, workers=True)

    d = cfg["data"]
    dm = FeatureDataModule(
        d["manifest"],
        d.get("max_patches", -1),
        d.get("num_workers", 8),
        d.get("feature_root"),
    )
    model = LitSAGEMIL(cfg)

    out = Path(cfg["general"]["output_dir"]) / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            dirpath=out,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best-{epoch:02d}-{val_loss:.4f}",
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=cfg["general"]["patience"],
        ),
    ]
    trainer = Trainer(
        max_epochs=cfg["general"]["epochs"],
        callbacks=callbacks,
        logger=CSVLogger(str(out), name="logs"),
        precision=cfg["general"]["precision"],
        accumulate_grad_batches=cfg["general"]["gradient_accumulation"],
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
