import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm
from val_eval import CrystalReconstructionEvaluator

from vqvae.decoder import Decoder
from vqvae.encoder import Encoder
from vqvae.lfq import LFQ
from vqvae.ot import VectorQuantizerSinkhorn
from vqvae.quantize import FlowQuantizer, VectorQuantizer, kmeans_centroids_path


class VQVAE(nn.Module):
    def __init__(self, config):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.device = config.device

        self.vae = config.model.vae
        self.flow = config.model.flow
        self.optvq = config.model.optvq
        self.lfq = config.model.lfq
        assert (
            (self.flow and (not self.optvq) and (not self.lfq))
            or ((not self.flow) and self.optvq and (not self.lfq))
            or ((not self.flow) and (not self.optvq) and self.lfq)
            or (
                self.vae
                and (not self.flow)
                and (not self.optvq)
                and (not self.lfq)
            )
            or (
                (not self.vae)
                and (not self.flow)
                and (not self.optvq)
                and (not self.lfq)
            )
        )
        if self.optvq:
            self.quant_conv = nn.Linear(
                config.model.hidden_dim, config.model.latent_dim
            )
            # parameters according to the original paper
            self.quantize = VectorQuantizerSinkhorn(
                epsilon=config.model.optvq_eps,
                n_iters=5,
                normalize_mode="all",
                use_prob=False,
                x_dim=config.model.latent_dim,
                k_dim=config.model.code_num,
                beta=0.25,
                use_norm=False,
                use_proj=False,
                fix_codes=False,
                loss_q_type="l2",
                num_head=1,
                kmeans_centroids=(
                    kmeans_centroids_path(config)
                    if config.model.kmeans_init
                    else None
                ),
            )
            # NOTE: quant_conv is (re-)created here a second time; kept so that parameter
            # initialization matches the OptVQ runs reported in the paper.
            self.quant_conv = nn.Linear(
                config.model.hidden_dim, config.model.latent_dim
            )
        elif self.lfq:
            self.quantize = LFQ(
                dim=config.model.latent_dim,
                codebook_size=config.model.code_num,
                entropy_loss_weight=0.02,
                commitment_loss_weight=0.25,
            )
            self.quant_conv = nn.Linear(
                config.model.hidden_dim, config.model.latent_dim
            )
        elif self.flow:
            self.quantize = FlowQuantizer(config)
            self.quant_conv = nn.Linear(
                config.model.hidden_dim, config.model.latent_dim
            )
        elif self.vae:
            self.quant_conv = nn.Linear(
                config.model.hidden_dim, 2 * config.model.latent_dim
            )
        else:
            self.quantize = VectorQuantizer(config)
            self.quant_conv = nn.Linear(
                config.model.hidden_dim, config.model.latent_dim
            )
        self.post_quant_conv = nn.Linear(
            config.model.latent_dim, config.model.hidden_dim
        )
        self.val_reconstruction_evaluators = CrystalReconstructionEvaluator()
        self.config = config

    def to_device(self, data):
        return [i.to(self.device) for i in data]

    def encode(self, atom_input, batch, return_inds=False):
        h = self.encoder(atom_input, batch)
        h = self.quant_conv(h)
        if self.vae:
            posterior = DiagonalGaussianDistribution(h)
            return posterior
        else:
            if return_inds:
                q_inds, quant = self.quantize.quantize(h)
                return q_inds, quant
            else:
                quant, codebook_loss = self.quantize(h)
                return quant, codebook_loss

    def decode(self, quant, batch):
        quant = self.post_quant_conv(quant)
        h, mask = to_dense_batch(quant, batch)
        pred_atom, pred_pos, pred_matrix = self.decoder(h, mask, batch)
        return pred_atom, pred_pos, pred_matrix

    def forward(
        self, data, train_codebook=False, encode_only=False, train_prior=False
    ):
        matrix, cart_coords, atomic_numbers, num_atoms, _, _ = self.to_device(
            data
        )
        bs = matrix.shape[0]
        batch = torch.repeat_interleave(
            torch.arange(bs).to(self.device), num_atoms
        )
        atom_input = torch.cat(
            [
                cart_coords,
                F.one_hot(atomic_numbers, 100).float(),
                matrix[batch],
            ],
            -1,
        )

        if train_codebook:
            with torch.no_grad():
                z_e = self.quant_conv(self.encoder(atom_input, batch))
                z_e = z_e.reshape(-1, z_e.shape[-1])
            if encode_only:
                return z_e
            elif train_prior:
                zq_inds, zq = self.quantize.quantize(z_e)
                return (
                    (self.quantize.codebook(zq_inds) - z_e.detach())
                    .square()
                    .mean()
                )
            else:
                codebook_fm_loss, _ = self.quantize.loss(z_e)
                return codebook_fm_loss
        else:
            if self.vae:
                posterior = self.encode(atom_input, batch)
                quant = posterior.sample()
                kl_loss = posterior.kl().mean()
            else:
                quant, codebook_loss = self.encode(atom_input, batch)

            pred_atom, pred_pos, pred_matrix = self.decode(quant, batch)

            # loss
            atom_loss = F.cross_entropy(pred_atom, atomic_numbers)
            pos_loss = F.mse_loss(pred_pos, cart_coords)
            matrix_loss = F.mse_loss(pred_matrix, matrix)

            if self.vae:
                return (
                    atom_loss,
                    pos_loss,
                    matrix_loss,
                    kl_loss,
                    atom_loss + pos_loss + matrix_loss + kl_loss * 0.00001,
                )
            else:
                return (
                    atom_loss,
                    pos_loss,
                    matrix_loss,
                    codebook_loss,
                    atom_loss
                    + pos_loss
                    + matrix_loss
                    + codebook_loss * self.config.model.code_weight,
                )

    def eval_val(self, val_loader, matrix_scaler, get_fps=False):
        self.val_reconstruction_evaluators.clear()
        self.val_reconstruction_evaluators.device = self.device

        used = torch.empty(0)
        for i, data in tqdm(enumerate(val_loader)):
            matrix, cart_coords, atomic_numbers, num_atoms, _, _ = (
                self.to_device(data)
            )
            bs = matrix.shape[0]
            batch = torch.repeat_interleave(
                torch.arange(bs).to(self.device), num_atoms
            )
            atom_input = torch.cat(
                [
                    cart_coords,
                    F.one_hot(atomic_numbers, 100).float(),
                    matrix[batch],
                ],
                -1,
            )

            if self.vae:
                posterior = self.encode(atom_input, batch)
                quant = posterior.sample()
            else:
                q_inds, quant = self.encode(atom_input, batch, return_inds=True)
                used = torch.cat(
                    [used, q_inds.reshape(-1).cpu().unique()]
                ).unique()
                # print("    ==> Used", used.shape[0], "Total", self.quantize.n_e)
            pred_atom, pred_pos, pred_matrix = self.decode(quant, batch)

            pred_atom = pred_atom.argmax(-1)
            pred_atom[pred_atom == 0] = 1
            pred_matrix = matrix_scaler.inverse_transform(
                pred_matrix
            ) * num_atoms.reshape(-1, 1).float() ** (1 / 3)
            pred_pos = matrix_scaler.inverse_transform(pred_pos) * num_atoms[
                batch
            ].reshape(-1, 1).float() ** (1 / 3)

            matrix = matrix_scaler.inverse_transform(
                matrix
            ) * num_atoms.reshape(-1, 1).float() ** (1 / 3)
            cart_coords = matrix_scaler.inverse_transform(
                cart_coords
            ) * num_atoms[batch].reshape(-1, 1).float() ** (1 / 3)

            start_id = 0
            for idx_in_batch, num in enumerate(num_atoms):
                _atom_types = atomic_numbers[start_id : start_id + num]
                _cart_coords = cart_coords[start_id : start_id + num]
                _matrix = matrix[idx_in_batch]

                self.val_reconstruction_evaluators.append_gt_array(
                    {
                        "atom_types": _atom_types.detach().cpu().numpy(),
                        "cart_coords": _cart_coords.detach().cpu().numpy(),
                        "matrices": vector2matrix(
                            _matrix.detach().cpu()
                        ).numpy(),
                    }
                )
                start_id = start_id + num

            start_id = 0
            for idx_in_batch, num in enumerate(num_atoms):
                _atom_types = pred_atom[start_id : start_id + num]
                _cart_coords = pred_pos[start_id : start_id + num]
                _matrix = pred_matrix[idx_in_batch]

                self.val_reconstruction_evaluators.append_pred_array(
                    {
                        "atom_types": _atom_types.detach().cpu().numpy(),
                        "cart_coords": _cart_coords.detach().cpu().numpy(),
                        "matrices": vector2matrix(
                            _matrix.detach().cpu()
                        ).numpy(),
                    }
                )
                start_id = start_id + num

        if not self.vae:
            print("    ==> Used", used.shape[0], "Total", self.quantize.n_e)
        results = self.val_reconstruction_evaluators.get_metrics(cart=True)
        return results["match_rate"]

    def get_emb(self, loader):
        all_embs = []
        all_nums = []
        for i, data in tqdm(enumerate(loader)):
            matrix, cart_coords, atomic_numbers, num_atoms, _, _ = (
                self.to_device(data)
            )
            bs = matrix.shape[0]
            batch = torch.repeat_interleave(
                torch.arange(bs).to(self.device), num_atoms
            )
            atom_input = torch.cat(
                [
                    cart_coords,
                    F.one_hot(atomic_numbers, 100).float(),
                    matrix[batch],
                ],
                -1,
            )

            if self.vae:
                posterior = self.encode(atom_input, batch)
                quant = posterior.sample()
            else:
                quant, _ = self.encode(atom_input, batch)

            all_embs.append(quant.detach().cpu())
            all_nums.append(num_atoms.detach().cpu())
        all_embs = torch.cat(all_embs, 0)
        all_nums = torch.cat(all_nums, 0)
        return all_embs, all_nums


def vector2matrix(vector):
    A = torch.zeros(3, 3, dtype=torch.float32, device=vector.device)
    A[0, 0] = vector[0]
    A[0, 1] = A[1, 0] = vector[1]
    A[0, 2] = A[2, 0] = vector[2]
    A[1, 1] = vector[3]
    A[1, 2] = A[2, 1] = vector[4]
    A[2, 2] = vector[5]
    return A


class DiagonalGaussianDistribution:
    """Diagonal Gaussian distribution with mean and logvar parameters.

    Adapted from: https://github.com/CompVis/latent-diffusion, with modifications for our tensors,
    which are of shape (N, d) instead of (B, H, W, d) for 2D images.
    """

    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(
            parameters, 2, dim=-1
        )  # split along channel dim
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=-1,
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=-1,
                )

    def mode(self):
        return self.mean

    def __repr__(self):
        return f"DiagonalGaussianDistribution(mean={self.mean}, logvar={self.logvar})"
