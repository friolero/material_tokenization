import torch
import torch.nn as nn
import math

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=2):
        assert n_layers >= 2
        super(MLP, self).__init__()
        self.map = nn.ModuleList([nn.Linear(in_dim, hidden_dim)])
        for _ in range(0, n_layers-2):
            self.map.append(nn.Linear(hidden_dim, hidden_dim))
        self.map.append(nn.Linear(hidden_dim, out_dim))
        self.act = nn.ReLU()
        self.n_layers = n_layers
    
    def forward(self, x):
        for i in range(self.n_layers-1):
            x = self.act(self.map[i](x))
        x = self.map[-1](x)
        return x

class MHA(nn.Module):
    def __init__(self, attn_head, dim, dropout):
        super(MHA, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.dim = dim
        self.attn_head = attn_head
        self.softmax = nn.Softmax(dim=-1)
        self.WO = nn.Linear(dim, dim)
    
    def forward(self, q, k, v, mask):
        b, lq, _ = q.shape
        b, kq, _ = k.shape
        b, vq, _ = v.shape

        dim_head = self.dim // self.attn_head
        q = q.reshape(b, lq, self.attn_head, dim_head).transpose(1, 2)
        k = k.reshape(b, kq, self.attn_head, dim_head).transpose(1, 2)
        v = v.reshape(b, vq, self.attn_head, dim_head).transpose(1, 2)
        
        mask = mask.unsqueeze(1).unsqueeze(-1).float()
        mask = mask @ mask.transpose(-1, -2)

        attn_scores = q @ k.transpose(2, 3) / math.sqrt(dim_head)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-1e3'))
        attn = self.softmax(attn_scores)
        attn = self.dropout(attn)
        
        attn_out = (attn @ v).transpose(1, 2).reshape(b, lq, self.dim)
        attn_out = self.dropout(attn_out)
        attn_out = self.WO(attn_out)
        return attn_out

class Decoder_layer(nn.Module):
    def __init__(self, dim, attn_head, dropout):
        assert dim % attn_head == 0
        super(Decoder_layer, self).__init__()
        self.LN_MHA_SA = nn.LayerNorm(dim)
        self.LN_MLP = nn.LayerNorm(dim)
        
        self.qkv = nn.Linear(dim, 3*dim)
        self.mha_sa = MHA(attn_head, dim, dropout)
        self.MLP = nn.Sequential(nn.Linear(dim, 4*dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(4*dim, dim))
    
    def forward(self, h, mask):
        ## self_attention
        h[~mask] = 0.
        h_ini = h
        h = self.LN_MHA_SA(h)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        sa_attn_out = self.mha_sa(q, k, v, mask)
        sa_attn_out[~mask] = 0.
        h = h_ini + sa_attn_out

        ## add & norm
        h = h + self.MLP(self.LN_MLP(h))
        h[~mask] = 0.
        return h
    
class Transformer(nn.Module):
    def __init__(self, hidden_dim, layers, attn_head, dropout, cls_token=False):
        super(Transformer, self).__init__()

        self.pe_emb = nn.Parameter(torch.zeros(500, hidden_dim))
        nn.init.trunc_normal_(self.pe_emb, std=0.02)

        if cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        self.block = nn.ModuleList()
        for _ in range(layers):
            self.block.append(Decoder_layer(hidden_dim, attn_head, dropout))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, mask):
        b, l, _ = h.shape

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(b, 1, -1)
            h = torch.cat([cls_tokens, h], 1)
            mask = torch.cat([torch.BoolTensor([[True]]).to(mask.device).repeat(b, 1), mask], 1)
            l = l+1
            
        h = h + self.pe_emb[:l].unsqueeze(0).repeat(b, 1, 1)

        h[~mask] = 0.

        for i in range(len(self.block)):
            h = self.block[i](h, mask)
        h = self.norm(h)
        h[~mask] = 0.
        return h