import torch
import argparse
from torch import nn, optim
from functools import lru_cache
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from torchmetrics import AUROC, AveragePrecision

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar

from data_utils import plant_Dataset
from data.plant_generate.plants import plant_feature, plant_bed
from models.SwanDNA import Plant_Classifier
from models.deeperdeepsea import DeeperDeepSEA
from models.x_formers import FormerClassifier


class LightningWrapper(pl.LightningModule):
    def __init__(self, m_name, model, cfg, snapshot_path, train_set, val_set, pretrained, loss, file_name, n_feature):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.batch_size = self.hparams.training.batch_size
        length = 1000  # cfg.training.length
        
        if m_name == 'swan':
            self.model_config = self.hparams.SwanDNA
            self.model = model(**self.model_config, output_size=n_feature)
        elif m_name == 'deepsea':
            self.model = model(sequence_length=length, n_targets=n_feature)
        else:
            self.model_config = self.hparams.xformer
            self.model = model(name=m_name, **self.model_config, max_seq_len=length, output_size=n_feature)

        for k, v in self.model.state_dict().items():
            print(k, v.dtype)

        self.snapshot_path = snapshot_path
        self.train_set = train_set
        self.val_set = val_set
        self.loss = loss
        self.file_name = file_name

        # print(self.model)
        self.train_auroc = AUROC(task="multilabel", num_labels=n_feature, average='none')
        self.train_ap = AveragePrecision(task="multilabel", num_labels=n_feature, average='none')
        self.eval_auroc = AUROC(task="multilabel", num_labels=n_feature, average='none')
        self.eval_ap = AveragePrecision(task="multilabel", num_labels=n_feature, average='none')
        self.best_roc = 0

        if pretrained:
            pretrained = torch.load(self.snapshot_path + '_E' + str(self.hparams.training.pre_load) + '.pt', map_location='cpu')
            pretrained = pretrained["MODEL_STATE"]

            from collections import OrderedDict
            new_state_dict = OrderedDict()

            for k, v in pretrained.items():
                if k.startswith('encoder') or k.startswith('embedding'):
                    new_state_dict[k] = v

            net_dict = self.model.state_dict()
            pretrained_cm = {k: v for k, v in new_state_dict.items() if k in net_dict}

            net_dict.update(pretrained_cm)
            self.model.load_state_dict(net_dict)
            for k, v in self.model.state_dict().items():
                print(k, v.shape)
            print(self.file_name)
            print("*************pretrained model loaded***************")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        seq, label = batch
        output = self.model(seq)
        train_loss = self.loss(output, label)
        self.train_auroc.update(output, label.int())
        self.train_ap.update(output, label.int())
        return {"loss": train_loss}

    def training_epoch_end(self, outputs):
        train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        train_auroc_all = self.train_auroc.compute()
        train_ap_all = self.train_ap.compute()
        train_roc = train_auroc_all.mean()
        train_ap = train_ap_all.mean()

        self.log('train_loss', train_loss, sync_dist=True)
        self.log('train_roc', train_roc, sync_dist=True)
        self.log('train_ap', train_ap, sync_dist=True)
        self.train_auroc.reset()
        self.train_ap.reset()

    def validation_step(self, batch, batch_idx):
        seq, label = batch
        output = self.model(seq)
        val_loss = self.loss(output, label)
        self.eval_auroc.update(output, label.int())
        self.eval_ap.update(output, label.int())
        return {"loss": val_loss}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["loss"] for x in outputs]).mean()
        eval_auroc_all = self.eval_auroc.compute()
        eval_ap_all = self.eval_ap.compute()
        val_roc = eval_auroc_all.mean()
        val_ap = eval_ap_all.mean()

        if self.best_roc <= val_roc:
            self.best_roc = val_roc
        self.log('val_loss', val_loss, sync_dist=True)
        self.log("val_roc", val_roc, sync_dist=True)
        self.log("val_ap", val_ap, sync_dist=True)
        self.log("best_roc", self.best_roc, sync_dist=True)
        self.eval_auroc.reset()
        self.eval_ap.reset()

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            num_workers=16,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            batch_size=self.batch_size
            )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            num_workers=16,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            batch_size=self.batch_size
            )

    @lru_cache
    def total_steps(self):
        len_tr = len(self.trainer._data_connector._train_dataloader_source.dataloader())
        print('Num devices', self.trainer.num_devices)
        max_epochs = self.trainer.max_epochs
        accum_batches = self.trainer.accumulate_grad_batches
        manual_total_steps = (len_tr // accum_batches * max_epochs)/self.trainer.num_devices
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
                    num_warmup_steps=int(self.total_steps()*self.hparams.training.warm),
                    num_training_steps=self.total_steps(),
                    num_cycles=0.5
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]


def classify_main(cfg, m_name, plant, num_train, n_epochs):
    pretrained = cfg.training.pretrained
    length = 1000  # cfg.training.length
    file_name = m_name + '_L' + str(length)

    pre_seq_len = cfg.Pretrain.pre_seq_len

    if m_name == 'swan':
        save_name = 'SwanDNA_plant_L' + str(pre_seq_len)
        this_model = Plant_Classifier
    else:
        save_name = None
        if m_name == 'deepsea':
            this_model = DeeperDeepSEA
        else:
            this_model = FormerClassifier
    file_name = file_name + '_' + str(cfg.training.learning_rate) + '_' + str(cfg.training.weight_decay) + '_' + str(cfg.training.warm)
    if pretrained:
        # save_dir = './pre_' + str(pre_seq_len) + '_tr' + str(cfg.Pretrain.pre_num * 7) + '/'
        save_dir = './Pretrained_models/'
        snapshot_path = save_dir + save_name
        file_name = "SwanDNA_plant_72_192.pt"
    else:
        snapshot_path = None

    # data
    fasta_path, bed_path, features_path, n_feature = plant_feature(plant)
    data_dir = '../plant_pt/'
    _, _, _, _, num_eval = plant_bed(plant)

    tr_name = plant + str(length) + '_tr' + str(num_train)
    te_name = plant + str(length) + '_te' + str(num_eval)

    train_sequences = torch.load(data_dir + tr_name + '_seq.pt')
    train_targets = torch.load(data_dir + tr_name + '_target.pt')
    test_sequences = torch.load(data_dir + te_name + '_seq.pt')
    test_targets = torch.load(data_dir + te_name + '_target.pt')
    print(train_sequences.shape, train_targets.shape)
    print(test_sequences.shape, test_targets.shape)

    train_set = plant_Dataset(train_sequences, train_targets)
    val_set = plant_Dataset(test_sequences, test_targets)

    loss = nn.BCEWithLogitsLoss()

    model = LightningWrapper(m_name, this_model, cfg, snapshot_path, train_set, val_set, pretrained, loss, file_name, n_feature)
    summary = ModelSummary(model, max_depth=2)
    print(summary)

    wandb_logger = WandbLogger(dir="./wandb/", project=plant + str(num_train), entity='', name=f'{file_name}')
    # checkpoint_callback = ModelCheckpoint(monitor="val_roc", mode="max", save_top_k=1)
    checkpoint_callback = ModelCheckpoint(monitor="val_roc", mode="max")

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks_for_trainer = [TQDMProgressBar(refresh_rate=10), lr_monitor, checkpoint_callback]

    ddp = DDPStrategy(process_group_backend="nccl", find_unused_parameters=True)
    trainer = pl.Trainer(
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        accelerator='gpu',
        strategy=ddp,
        devices=-1,
        max_epochs=n_epochs,
        gradient_clip_val=1,
        num_sanity_val_steps=0,
        precision=16,
        logger=wandb_logger,
        callbacks=callbacks_for_trainer,
    )
    trainer.fit(model)


if __name__ == "__main__":
    this_cfg = OmegaConf.load('config/config_plant.yaml')
    OmegaConf.set_struct(this_cfg, False)

    parser = argparse.ArgumentParser(description='experiment')
    parser.add_argument('-m_name', type=str, default='swan')
    parser.add_argument('-plant', type=str, default='ar')  # ar bd mh sb si zm zs
    parser.add_argument('-num_train', type=int, default=500)
    parser.add_argument('-n_epochs', type=int, default=20)
    parser.add_argument('-seed', type=int, default=42)
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    classify_main(this_cfg, args.m_name, args.plant, args.num_train, args.n_epochs)
