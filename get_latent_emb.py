"""Encode the training set with a saved auto-encoder (ae_<epoch>.pt) for k-means codebook initialization.

Writes ae_emb_<epoch>.pt next to the checkpoint. Usage:
    python get_latent_emb.py --ckpt logs/mp_20/<run>/saved_model/ae_500.pt --conf_new model.flow=True
"""

import argparse

import torch
from torch.utils.data import DataLoader

from config import DictAction, parse_cfg
from dataloader import CrystalDataset, collate
from utils import dict2namespace, get_scaler
from vqvae.quantize import ae_derived_path
from vqvae.vqvae import VQVAE

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dataset", type=str, default="mp_20")
parser.add_argument("--ckpt", type=str, required=True, help="Auto-encoder checkpoint ae_<epoch>.pt")
parser.add_argument("--conf_new", nargs="+", action=DictAction, help="Config overrides of the training run")
args = parser.parse_args()

config = dict2namespace(parse_cfg(f"configs/{args.dataset}.yml", args.conf_new))
config.device = torch.device("cuda")

model = VQVAE(config).cuda()
state_dict = torch.load(args.ckpt)["model_state_dict"]
model.load_state_dict({k.replace("module.", "", 1): v for k, v in state_dict.items()})
model.eval()

train = CrystalDataset(args.dataset, config=config)
train.matrix_scaler = get_scaler(args)
loader = DataLoader(train, shuffle=True, batch_size=500, collate_fn=collate)

# The FlowQuantizer of an auto-encoder checkpoint is not initialized yet, so the "quantized" latents
# returned here are the raw encoder latents.
with torch.no_grad():
    all_embs, _ = model.get_emb(loader)
out_path = ae_derived_path(args.ckpt, "ae_emb_")
torch.save(all_embs, out_path)
print(f"==> Saved {tuple(all_embs.shape)} latents to {out_path}")
