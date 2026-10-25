import torch.nn as nn
from vqvae.transformer import Transformer
from torch_scatter import scatter_mean


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.trans = Transformer(config.model.hidden_dim, config.model.neigh_layer, config.model.attn_head, config.model.dropout, cls_token=False)
        self.atom_pred = nn.Linear(config.model.hidden_dim, 100)
        self.pos_pred = nn.Linear(config.model.hidden_dim, 3)
        self.matrix_pred = nn.Linear(config.model.hidden_dim, 6)

    def forward(self, h, mask, batch):
        h = self.trans(h, mask)
        h = h[mask]
        h_global = scatter_mean(h, batch, dim=0)

        pred_atom = self.atom_pred(h)
        pred_pos = self.pos_pred(h)
        pred_matrix = self.matrix_pred(h_global)
        return pred_atom, pred_pos, pred_matrix
