import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from augment import RandomInversion, RandomTranslocation, RandomInsertion, RandomMutation
import pytorch_lightning as pl
import torch.nn as nn
from functools import lru_cache
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import Accuracy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from models.SwanDNA import SwanDNANetwork
from transformers import get_cosine_schedule_with_warmup
import random
import numpy as np
from pytorch_metric_learning.losses import NTXentLoss

def random_nucleotide_replacement_one_hot(sequence, p=0.1):
    nucleotides = np.eye(5)
    for i in range(sequence.shape[0]):
        if random.random() < p:
            sequence[i] = random.choice(nucleotides)
    return sequence

def shuffle_sequence_one_hot(sequence):
    indices = np.arange(sequence.shape[0])
    np.random.shuffle(indices)
    return sequence[indices]

def reverse_complement_one_hot(sequence):
    complement = np.array([4, 3, 2, 1, 0])
    return sequence[::-1][:, complement]

def dna_transform(sequence):
    sequence = random_nucleotide_replacement_one_hot(sequence, p=0.1)
    sequence = shuffle_sequence_one_hot(sequence)
    if random.random() < 0.5:
        sequence = reverse_complement_one_hot(sequence)
    return sequence
    
class DNASequenceDataset(Dataset):
    def __init__(self, sequences, transform):
        self.sequences = sequences
        self.transform = transform

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        seq1 = self.transform(sequence)
        seq2 = self.transform(sequence)
        return seq1, seq2

class SimCLRModel(pl.LightningModule):
    def __init__(self, config, train_set, val_set):
        super(SimCLRModel, self).__init__()
        self.save_hyperparameters(config.Pretraining)
        print(self.hparams)
        training_config = config.Pretraining.training
        model_config = config.Pretraining.SwanDNA
         
        self.encoder1 = SwanDNANetwork(
                        model_config.max_len,
                        model_config.embedding_size,
                        model_config.group_size,
                        model_config.hidden_size,
                        model_config.mlp_dropout,
                        model_config.layer_dropout,
                        model_config.prenorm,
                        model_config.norm,
                        model_config.block_num
                    )

        self.encoder2 = SwanDNANetwork(
                        model_config.max_len,
                        model_config.embedding_size,
                        model_config.group_size,
                        model_config.hidden_size,
                        model_config.mlp_dropout,
                        model_config.layer_dropout,
                        model_config.prenorm,
                        model_config.norm,
                        model_config.block_num
                    )

        self.temperature = training_config.temperature
        self.batch_size = training_config.batch_size
        self.learning_rate = training_config.learning_rate
        self.train_set = train_set
        self.val_set = val_set
        # self.loss = NTXentLoss(self.temperature)

    def nt_xent_loss(self, out_1, out_2):
        out_1, out_2 = torch.mean(out_1, 1).squeeze(1), torch.mean(out_2, 1).squeeze(1)
        batch_size = out_1.shape[0]
        out = torch.cat([out_1, out_2], dim=0)
        similarity_matrix = F.cosine_similarity(out.unsqueeze(1), out.unsqueeze(0), dim=2)

        # print(similarity_matrix.shape)
        
        # Create the mask to ignore self-similarities
        mask = torch.eye(batch_size * 2, dtype=torch.bool).to(out.device)
        
        # Remove self-similarities from the similarity matrix
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
        
        # Temperature scaling
        logits = similarity_matrix / self.temperature
        
        # Targets are the indices of the positive pairs (45670123 because positive pairs are (0,4) (1,5) (2,6), (3,7))
        targets = torch.cat([torch.arange(batch_size, batch_size * 2), torch.arange(batch_size)], dim=0).to(out.device)
        
        # CrossEntropyLoss expects class indices, so no need to create a one-hot encoded labels
        criterion = nn.CrossEntropyLoss()
        
        # print(logits.shape, targets.shape)
        loss = criterion(logits, targets)
    
        return loss

    def forward(self, x):
        return self.encoder1(x)

    def training_step(self, batch, batch_idx):
        seq1, seq2 = batch
        out1 = self.encoder1(seq1)
        out2 = self.encoder2(seq2)
        loss = self.nt_xent_loss(out1, out2)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        seq1, seq2 = batch
        out1 = self.encoder1(seq1)
        out2 = self.encoder2(seq2)
        loss = self.nt_xent_loss(out1, out2)
        self.log('val_loss', loss)
        return loss

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

def training(cfg):
    #"1. load data":
    genes_hg38_30k = torch.load(f"./data/gene_train_{cfg.Pretraining.training.max_len}_30k.pt").numpy()
    genes_hg38_20k = torch.load(f"./data/gene_train_{cfg.Pretraining.training.max_len}_20k.pt").numpy()
    genes_val = torch.load(f"./data/gene_valid_{cfg.Pretraining.training.max_len}_100k.pt").numpy()

    genes_hg38 = np.concatenate((genes_hg38_30k, genes_hg38_20k), 0)

    train_set = DNASequenceDataset(genes_hg38, dna_transform)
    val_set = DNASequenceDataset(genes_val, dna_transform)

    #"2. create model"
    model = SimCLRModel(cfg, train_set, val_set)

    #"3. logging and checkpoints"

    wandb_logger = WandbLogger(project=cfg.experiment.project_name,
                               name=cfg.experiment.run_id,
                               job_type='train',
                               save_dir=cfg.experiment.output_dir
                               )

    # Make all callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.experiment.output_dir}checkpoints",
        filename=cfg.experiment.run_id,
        verbose=cfg.experiment.verbose,
        monitor="train_loss",
    )

    #"4. training"

    trainer = pl.Trainer(
        check_val_every_n_epoch=5,
        enable_progress_bar=True,
        accelerator='gpu',
        strategy="ddp",
        devices=[0, 1],
        max_epochs=cfg.Pretraining.training.n_epochs,
        gradient_clip_val=0.5,
        num_sanity_val_steps=2,
        precision=16,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=5
    )
    trainer.fit(model)

if __name__ == "__main__":
    cfg = OmegaConf.load('./config/config_simclr.yaml')
    OmegaConf.set_struct(cfg, False)
    training(cfg)

