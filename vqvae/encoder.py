import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from vqvae.transformer import Transformer


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=2):
        assert n_layers >= 2
        super(MLP, self).__init__()
        self.map = nn.ModuleList([nn.Linear(in_dim, hidden_dim)])
        for _ in range(0, n_layers - 2):
            self.map.append(nn.Linear(hidden_dim, hidden_dim))
        self.map.append(nn.Linear(hidden_dim, out_dim))
        self.act = nn.ReLU()
        self.n_layers = n_layers

    def forward(self, x):
        for i in range(self.n_layers - 1):
            x = self.act(self.map[i](x))
        x = self.map[-1](x)
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.atom_emb = MLP(109, config.model.hidden_dim, config.model.hidden_dim)
        self.fuse_trans = Transformer(
            config.model.hidden_dim,
            config.model.fuse_layer,
            config.model.attn_head,
            config.model.dropout,
            cls_token=False,
        )

    def forward(self, x, batch):
        atom_emb = self.atom_emb(x)
        batch_atom_emb, batch_mask = to_dense_batch(atom_emb, batch)
        h = self.fuse_trans(batch_atom_emb, batch_mask)
        return h[batch_mask]
