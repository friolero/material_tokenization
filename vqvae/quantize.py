import math
import os

import torch
import torch.nn as nn
from einops import rearrange
from torch.distributions.normal import Normal
from zuko.utils import odeint


def ae_derived_path(ae_ckpt, prefix):
    """``.../ae_<epoch>.pt`` -> ``.../<prefix><epoch>.pt`` (latents / k-means centroids of that auto-encoder)."""
    dirname, basename = os.path.split(ae_ckpt)
    assert basename.startswith("ae_"), f"Expected an auto-encoder checkpoint 'ae_<epoch>.pt', got {ae_ckpt}"
    return os.path.join(dirname, prefix + basename[len("ae_") :])


def kmeans_centroids_path(config):
    """K-means centroids written by ``get_kmeans.py`` for the auto-encoder in ``config.model.ckpt``."""
    if not config.model.ckpt:
        raise ValueError("model.kmeans_init=True requires model.ckpt=<path to ae_<epoch>.pt>")
    path = ae_derived_path(config.model.ckpt, f"ae_centroids_{config.model.code_num}_")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found: run get_latent_emb.py and get_kmeans.py first (see README)")
    return path


def find_nearest(query, dictionary):
    dists = ((query.unsqueeze(1) - dictionary.unsqueeze(0)) ** 2).sum(-1)
    min_inds = dists.min(1)[1]
    return min_inds, dictionary[min_inds]


class VectorQuantizer(nn.Module):
    def __init__(self, config):
        super(VectorQuantizer, self).__init__()
        self.n_e = config.model.code_num
        self.e_dim = config.model.latent_dim
        self.beta = 0.25
        self.embed = nn.Embedding(self.n_e, self.e_dim)
        if config.model.kmeans_init:
            centroids_path = kmeans_centroids_path(config)
            self.embed.weight.data.copy_(torch.load(centroids_path))
            print(f"==> Initialized the codebook from {centroids_path}")

    def forward(self, z):
        d = (
            torch.sum(z**2, dim=1, keepdim=True)
            + torch.sum(self.embed.weight**2, dim=1)
            - 2
            * torch.einsum(
                "bd,dn->bn", z, rearrange(self.embed.weight, "n d -> d n")
            )
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embed(min_encoding_indices)
        codebook_loss = torch.mean(
            (z_q.detach() - z) ** 2
        ) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        z_q = z + (z_q - z).detach()
        return z_q, codebook_loss

    def quantize(self, z):
        d = (
            torch.sum(z**2, dim=1, keepdim=True)
            + torch.sum(self.embed.weight**2, dim=1)
            - 2
            * torch.einsum(
                "bd,dn->bn", z, rearrange(self.embed.weight, "n d -> d n")
            )
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embed(min_encoding_indices)
        z_q = z + (z_q - z).detach()
        return min_encoding_indices, z_q


class IndependentGaussianDist:
    def __init__(self, sample_dim, loc=None, std=None):

        self.sample_dim = sample_dim

        if loc is None:
            self._loc = torch.zeros(self.sample_dim)
        else:
            assert loc.shape[-1] == self.sample_dim, "Wrong loc dimension."
            self._loc = loc

        if std is None:
            self._std = torch.ones(sample_dim)
        else:
            assert std.shape[-1] == self.sample_dim, "Wrong var dimension."
            self._std = std

        self._dist = [
            Normal(self._loc[i], self._std[i]) for i in range(self.sample_dim)
        ]

    def icdf(self, cdfs):
        """Map (N, sample_dim) values in (0, 1) to prior samples, dimension by dimension."""
        return torch.cat(
            [
                self._dist[i].icdf(cdfs[:, i]).reshape(-1, 1)
                for i in range(self.sample_dim)
            ],
            dim=1,
        )


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class MLP(nn.Module):
    """
    MLP Model
    """

    def __init__(self, x_dim, c_dim=None, h_dim=256, n_hidden=3):

        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim

        self.x_mlp = nn.Linear(x_dim, h_dim)
        self.t_enc = SinusoidalPosEmb(h_dim)
        if c_dim is not None:
            self.c_mlp = nn.Linear(self.c_dim, h_dim)

        self.mid_layer = nn.ModuleList()
        for in_dim, out_dim in zip([h_dim] * n_hidden, [h_dim] * n_hidden):
            self.mid_layer.extend([nn.Linear(in_dim, out_dim), nn.ELU()])
        self.final_layer = nn.Linear(h_dim, self.x_dim)

    def forward(self, x, t, c=None):
        t_emb = self.t_enc(t)
        h = self.x_mlp(x) + t_emb
        if c is not None:
            h += self.c_mlp(c)
        for layer in self.mid_layer:
            h = layer(h)
        output = self.final_layer(h)
        return output


class FlowMatchingLoss(nn.Module):
    """Conditional flow matching loss with a linear interpolation path."""

    def __init__(self, sigma_min=1e-4):
        super().__init__()
        self.sigma_min = sigma_min

    def forward(self, flow, x, c=None):
        t = torch.rand_like(x[..., 0, None])
        z = torch.randn_like(x)
        psi_t = (1 - (1 - self.sigma_min) * t) * z + t * x
        d_psi = x - (1 - self.sigma_min) * z
        pred_psi = flow.net(psi_t, t.squeeze(-1), c=c)
        return (pred_psi - d_psi.detach()).square().mean()


class ZukoPFlow(nn.Module):
    """Probability-flow ODE solved with zuko's adaptive ``odeint``. t=0 is the prior, t=1 the data."""

    def __init__(self, net, prior_dist, t_limit=None):
        super().__init__()
        self.prior_dist = prior_dist
        self.t_limit = t_limit
        self.net = net

    def forward(self, t, x):
        if len(t.shape) == 0:
            t = t.unsqueeze(0).repeat(x.shape[0]).to(x)
        if self.t_limit is not None:
            t = torch.clip(t, self.t_limit[0], self.t_limit[1])
        return self.net(x, t)

    def encode(self, tgt_x):
        """Data -> prior."""
        return odeint(self, tgt_x, 1.0, 0.0, phi=self.net.parameters())

    def decode(self, src_x):
        """Prior -> data."""
        return odeint(self, src_x, 0.0, 1.0, phi=self.net.parameters())


class FlowQuantizer(nn.Module):
    """FlowVQ codebook.

    A flow-matching model maps a standard Gaussian onto the encoder latent distribution. The codebook
    is obtained by pushing a scrambled Sobol (quasi-random) grid of the Gaussian prior through the flow,
    optionally followed by Lloyd iterations on dense flow samples (``self_correction``).
    """

    def __init__(self, config):
        super().__init__()

        self.k_dim = config.model.code_num
        self.x_dim = config.model.latent_dim
        self.beta = 0.25

        self.prior_dist = IndependentGaussianDist(
            self.x_dim, loc=torch.zeros(self.x_dim), std=torch.ones(self.x_dim)
        )
        net = MLP(x_dim=self.x_dim)  # 3 hidden layers of width 256
        self.flow = ZukoPFlow(net, self.prior_dist, t_limit=(0, 1))

        self.codebook = nn.Embedding(self.k_dim, self.x_dim)
        # NOTE: the scrambled Sobol engine draws from the global torch RNG. Keep the draw order below
        # (prior codebook first, then the dense samples) to reproduce the paper's codebooks.
        self.sobol = torch.quasirandom.SobolEngine(
            dimension=self.prior_dist.sample_dim, scramble=True
        )
        cdf_vals = self.sobol.draw(self.k_dim)
        self.register_buffer("prior_codebook", self.prior_dist.icdf(cdf_vals))
        eps = 5e-3
        self.sample_ratio = 50
        self.cdf_vals = self.sobol.draw(self.sample_ratio * self.k_dim).clamp(
            eps, 1 - eps
        )
        self.initialized = False
        self.self_correction = config.model.self_correction
        self.online_correction = config.model.online_correction

    @property
    def n_e(self):
        return self.k_dim

    @property
    def e_dim(self):
        return self.x_dim

    @property
    def embed(self):
        return self.codebook

    @property
    def vocab_t(self):
        with torch.no_grad():
            posterior_codebook = self.flow.decode(self.prior_codebook)
        return posterior_codebook

    def set_codebook(self, codebook_vectors):
        with torch.no_grad():
            self.codebook.weight.copy_(codebook_vectors)

    def update_codebook(self, optim_iter=0):
        """Reset the codebook to the flow push-forward of the prior codebook, then (if
        ``self_correction``) run ``optim_iter`` Lloyd iterations against dense flow samples."""
        self.set_codebook(self.vocab_t)
        self.initialized = True
        if self.self_correction:
            if optim_iter > 0:
                with torch.no_grad():
                    centroids = self.codebook.weight.data.clone()
                    sample_priors = self.prior_dist.icdf(self.cdf_vals).to(
                        centroids.device
                    )
                    samples = [
                        self.flow.decode(
                            sample_priors[i * self.k_dim : (i + 1) * self.k_dim]
                        ).detach()
                        for i in range(self.sample_ratio)
                    ]
                    samples = torch.cat(samples, 0)
                    for n_iter in range(optim_iter):
                        zq_inds = torch.cat(
                            [
                                find_nearest(
                                    samples[i * self.k_dim : (i + 1) * self.k_dim],
                                    centroids,
                                )[0]
                                for i in range(self.sample_ratio)
                            ],
                            0,
                        )
                        for i in range(self.k_dim):
                            if (zq_inds == i).sum() > 0:
                                centroids[i] = samples[zq_inds == i].mean(0)
                            else:
                                print(
                                    f"==> Lloyd iter {n_iter}: code {i} has no samples, keeping it unchanged"
                                )
                    self.set_codebook(centroids)

    def quantize(self, z):
        """Nearest-code lookup. Before the codebook is initialized this is the identity (plain AE)."""
        if self.initialized:
            z_flattened = z.reshape(-1, self.x_dim)

            d = (
                torch.sum(z_flattened**2, dim=1, keepdim=True)
                + torch.sum(self.codebook.weight**2, dim=1)
                - 2
                * torch.einsum(
                    "bd,dn->bn",
                    z_flattened,
                    rearrange(self.codebook.weight, "n d -> d n"),
                )
            )
            min_encoding_indices = torch.argmin(d, dim=1)
            z_q = self.codebook(min_encoding_indices).reshape(z.shape)
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[:-1])
        else:
            min_encoding_indices = torch.ones(z.shape[:-1]).long().to(z.device)
            z_q = z.clone()
        return min_encoding_indices, z_q

    def loss(self, ze):
        assert (
            ze.ndim == 2 and ze.shape[1] == self.x_dim
        ), "Wrong input dimension, should be (N, d)"
        loss = FlowMatchingLoss()(self.flow, ze)
        log = {"latent_fm_loss": loss}
        return loss, log

    def forward(self, z):
        zq_inds, z_q = self.quantize(z)

        if self.online_correction:
            # With the additional codebook loss the matching is more precise and the performance is 92% vs 90% without
            codebook_loss = torch.mean(
                (z_q - z.detach()) ** 2
            ) + self.beta * torch.mean((z_q.detach() - z) ** 2)
        else:
            # The default loss with commitment loss only
            codebook_loss = self.beta * torch.mean((z_q.detach() - z) ** 2)

        z_q = z + (z_q - z).detach()
        return z_q, codebook_loss
