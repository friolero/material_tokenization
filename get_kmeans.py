"""K-means (faiss) on the latents written by get_latent_emb.py; the centroids initialize the baselines.

Writes ae_centroids_<K>_<epoch>.pt next to the checkpoint. Usage:
    python get_kmeans.py --ckpt logs/mp_20/<run>/saved_model/ae_500.pt
"""

import argparse

import faiss
import faiss.contrib.torch_utils  # noqa: F401  (lets faiss consume torch tensors)
import numpy as np
import torch

from vqvae.quantize import ae_derived_path

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--ckpt", type=str, required=True, help="Auto-encoder checkpoint ae_<epoch>.pt")
parser.add_argument("--code_num", type=int, nargs="+", default=[8192], help="Codebook size(s)")
parser.add_argument("--niter", type=int, default=30)
args = parser.parse_args()

all_embs = torch.load(ae_derived_path(args.ckpt, "ae_emb_"))
for k in args.code_num:
    kmeans = faiss.Kmeans(d=all_embs.shape[1], k=k, niter=args.niter, verbose=False)
    kmeans.train(all_embs)
    dists, _ = kmeans.index.search(all_embs, 1)  # squared L2 distances
    out_path = ae_derived_path(args.ckpt, f"ae_centroids_{k}_")
    torch.save(torch.from_numpy(kmeans.centroids).float(), out_path)
    print(f"==> K={k}: mean distance to centroid {np.sqrt(dists).mean():.4f}, saved to {out_path}")
