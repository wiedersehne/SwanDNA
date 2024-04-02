from DNASWAN import DNASwanEncoder
import torch.nn as nn
import math
import numpy as np

class Model4Pretrain(nn.Module):
    """
    DNASwan Model for Pretrain : Masked LM
    With one DNASwan encoder and one DNASwan decoder.
    """
    def __init__(self, input_size, max_len, embedding_size, group_size, hidden_size, mlp_dropout, layer_dropout, prenorm, norm):
        super().__init__()
        self.max_n_layers = math.ceil(np.log2(max_len))
        self.embedding_size = (self.max_n_layers+1) * group_size
        self.embedding = nn.Linear(
                    input_size,
                    self.embedding_size
            )
        self.encoder = DNASwanEncoder(
                max_len,
                self.embedding_size,
                group_size,
                hidden_size,
                mlp_dropout,
                layer_dropout,
                prenorm,
                norm
            )
        self.decoder = DNASwanEncoder(
                max_len,
                self.embedding_size,
                group_size,
                hidden_size,
                mlp_dropout,
                layer_dropout,
                prenorm,
                norm
            )

        self.linear = nn.Linear(self.embedding_size, input_size)

    def forward(self, input_seq):
        input_seq = input_seq.float()
        h = self.embedding(input_seq)
        # encoder
        h = self.encoder(h)
        # decoder
        h = self.linear(self.decoder(h))

        return h