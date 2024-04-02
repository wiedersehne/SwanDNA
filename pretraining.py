from random import random as rand
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
from functools import lru_cache
import pytorch_lightning as pl
from torch import nn, optim
from transformers import get_cosine_schedule_with_warmup
from pretraining_models import Model4Pretrain
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging, TQDMProgressBar
pl.seed_everything(42)


def pretrain_loss(loss, preds, labels, masks):
    masks_new = masks.repeat(5, 1, 1)#.reshape(preds.shape)
    masks_new = torch.reshape(masks_new, preds.shape)

    labels = labels[masks_new == 1]
    preds = preds[masks_new == 1]
    print(len(labels), len(preds))

    return loss(preds.float(), labels.float())


class DatasetCreator(Dataset):
    """
    Class to construct a dataset for training/inference
    """
    def __init__(self, genes, masked_genes, masks):
        self.genes = genes
        self.masked_genes = masked_genes
        self.masks = masks

    def __getitem__(self, index):
        return (self.genes[index], self.masked_genes[index], self.masks[index])

    def __len__(self):
        return len(self.genes)


class LightningWrapper(pl.LightningModule):
    def __init__(self, model, cfg, snapshot_path, train_set, val_set, loss):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model_config = self.hparams.DNASwan
        self.batch_size = self.hparams.training.batch_size
        self.length = self.hparams.training.max_len
        self.dim = self.model_config.embedding_size
        self.model = model(**self.model_config).apply(self._init_weights)
        self.save_every = self.hparams.training.save_every
        self.snapshot_path = snapshot_path
        self.train_set = train_set
        self.val_set = val_set
        self.loss = loss

        print(self.model)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.model(x)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def training_step(self, batch, batch_idx):
        gene, masked_gene, masks = batch
        logits = torch.squeeze(self.model(masked_gene))
        loss = pretrain_loss(self.loss, logits, gene, masks)
        self.log('train_loss', loss, sync_dist=True)
        if self.global_rank == 0 and self.global_step % self.save_every == 0:
            l = float("{:.4f}".format(loss.item()))
            self._save_snapshot(l)
        return {"loss":loss}

    def validation_step(self, batch, batch_idx):
        gene, masked_gene, masks = batch
        # print(gene.shape, masked_gene.shape, masks.shape)
        logits = torch.squeeze(self.model(masked_gene))
        # print(logits.shape)
        loss = pretrain_loss(self.loss, logits, gene, masks)

        return {"loss":loss}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log('val_loss', val_loss, sync_dist=True)
        # self._save_snapshot()

    def _save_snapshot(self, loss):
        snapshot = {
            "MODEL_STATE": self.model.state_dict(),
            "EPOCHS_RUN": self.current_epoch ,
        }
        torch.save(snapshot, f"{self.snapshot_path}/model_{self.current_epoch}_{self.length}_{self.dim}_{loss}.pt")
        print(f"Epoch {self.current_epoch } | Training snapshot saved at {self.snapshot_path}")

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:0"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            num_workers=1,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            batch_size=self.batch_size
            )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            num_workers=1,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
            batch_size=self.batch_size
            )

    @lru_cache
    def total_steps(self):
        l = len(self.trainer._data_connector._train_dataloader_source.dataloader())
        print('Num devices', self.trainer.num_devices)
        max_epochs = self.trainer.max_epochs
        accum_batches = self.trainer.accumulate_grad_batches
        manual_total_steps = (l // accum_batches * max_epochs)/self.trainer.num_devices
        print('MANUAL Total steps', manual_total_steps)
        return manual_total_steps

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.training.learning_rate,
            weight_decay=self.hparams.training.weight_decay
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.total_steps()*0.3,
                    num_training_steps=self.total_steps(),
                    num_cycles=self.hparams.training.n_cycles
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]


def pretrain_main(cfg):
    """
    # 1. Load data for pretraining
    """
    genes_train = torch.load(f"./data/pretrain/gene_train_{cfg.Pretraining.training.max_len}_200k.pt")
    masked_genes_train = torch.load(f"./data/pretrain/masked_train_{cfg.Pretraining.training.max_len}_200k.pt")
    masks_train = torch.load(f"./data/pretrain/mask_train_{cfg.Pretraining.training.max_len}_200k.pt")

    genes_val = torch.load(f"./data/pretrain/gene_val_{cfg.Pretraining.training.max_len}.pt")
    masked_genes_val = torch.load(f"./data/pretrain/masked_val_{cfg.Pretraining.training.max_len}.pt")
    masks_val = torch.load(f"./data/pretrain/mask_val_{cfg.Pretraining.training.max_len}.pt")
    
    # print(genes_hg38.shape, masked_genes_hg38.shape, masks_hg38.shape)
    # print(genes_hg38[0:10], masked_genes_hg38[0:10], masks_hg38[0:10])

    train_set =   DatasetCreator(genes_train, masked_genes_train, masks_train)
    val_set =   DatasetCreator(genes_val, masked_genes_val, masks_val)

    """
    # 2. Prepare model
    """

    ddp = DDPStrategy(process_group_backend="nccl", find_unused_parameters=True)
    # profiler = SimpleProfiler()
    snapshot_path = "./pretrained_models/"

    # loss = nn.CrossEntropyLoss(reduce="sum")
    loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
    # loss = nn.MSELoss(reduce="sum")
    model = LightningWrapper(Model4Pretrain, cfg.Pretraining, snapshot_path, train_set, val_set, loss)
    print(model)
    summary = ModelSummary(model, max_depth=-1)
    wandb_logger = WandbLogger(dir="./wandb/", project="VE_Pretrain", entity='', name=f'Pretraining_{cfg.Pretraining.training.max_len}_{cfg.Pretraining.DNASwan.embedding_size}_{cfg.Pretraining.DNASwan.hidden_size}')
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks_for_trainer = [TQDMProgressBar(refresh_rate=10), lr_monitor, checkpoint_callback]
    
    """
    # 3. init trainer
    """

    print(summary)
    trainer = pl.Trainer(
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        accelerator='gpu',
        strategy=ddp,
        devices=[0, 1],
        max_epochs=cfg.Pretraining.training.n_epochs,
        gradient_clip_val=0.5,
        num_sanity_val_steps=0,
        precision=16,
        logger=wandb_logger,
        callbacks=callbacks_for_trainer
    )
    trainer.fit(model)


if __name__ == "__main__":
    cfg = OmegaConf.load('./config/config_gb.yaml') #for ve pretraining, chenge it to config.yaml
    OmegaConf.set_struct(cfg, False)
    pretrain_main(cfg)
