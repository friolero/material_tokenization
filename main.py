"""Train a crystal (MP-20) tokenizer: FlowVQ or one of the VQ-VAE / LFQ / OptVQ baselines.

See README.md for the commands used in the paper.
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from sklearn import manifold
from torch.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import wandb
from dataloader import CrystalDataset, collate
from utils import get_scaler, parse_args_and_config, set_seed
from vqvae.quantize import find_nearest
from vqvae.vqvae import VQVAE

warnings.filterwarnings("ignore")

FM_PATIENCE = 5  # codebook-initialization rounds without improvement before stopping


def eval_qloss(data, batch_size, codebook):
    """Mean L1 quantization error of ``data`` (N, D); the last incomplete batch is skipped."""
    n_batch = len(data) // batch_size
    qlosses = []
    for batch_idx in range(n_batch):
        z_e = data[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        qlosses.append((find_nearest(z_e, codebook)[1] - z_e).abs().mean())
    return sum(qlosses) / len(qlosses)


def get_vis_embedding(raw_embedding, codebook_embedding, method="tsne"):
    assert method in [
        "tsne",
        "isomap",
    ], "Incompatible dimension reducing method."
    if raw_embedding.shape[1] > 3:
        combined = np.vstack([raw_embedding, codebook_embedding])
        if method == "isomap":
            dim_reductor = manifold.Isomap(n_neighbors=12, n_components=3, p=1)
        else:
            dim_reductor = manifold.TSNE(n_components=3)
        reduced = dim_reductor.fit_transform(combined)
        return (
            reduced[: raw_embedding.shape[0]],
            reduced[raw_embedding.shape[0] :],
        )
    else:
        return raw_embedding, codebook_embedding


def vis_3d_codebook(
    raw_zes, raw_vocab_t, epoch, n_vis=1000, logger=None, name="codebook"
):

    raw_zes = raw_zes.flatten(0, -2)
    if raw_vocab_t is None:
        raw_vocab_t = raw_zes

    ze_vis = raw_zes[torch.randperm(raw_zes.size(0))][:n_vis].cpu().numpy()
    vocab_vis = (
        raw_vocab_t[torch.randperm(raw_vocab_t.size(0))][:n_vis].cpu().numpy()
    )

    print("==> Reduce dimension for visualization")
    ze_vis, vocab_vis = get_vis_embedding(ze_vis, vocab_vis)
    print("==> Done")

    x_min, y_min, z_min = ze_vis.min(axis=0)
    x_max, y_max, z_max = ze_vis.max(axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.scatter(ze_vis[:, 0], ze_vis[:, 1], ze_vis[:, 2], marker="o")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    ax = fig.add_subplot(1, 3, 2, projection="3d")
    ax.scatter(vocab_vis[:, 0], vocab_vis[:, 1], vocab_vis[:, 2], marker="^")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    ax = fig.add_subplot(1, 3, 3, projection="3d")
    ax.scatter(ze_vis[:, 0], ze_vis[:, 1], ze_vis[:, 2], marker="o")
    ax.scatter(vocab_vis[:, 0], vocab_vis[:, 1], vocab_vis[:, 2], marker="^")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    wandb.log({name: wandb.Image(fig)}, step=epoch)
    plt.close(fig)


def load_model(ckpt, config, rank=None):
    checkpoint = torch.load(ckpt)
    model = VQVAE(config)
    new_state_dict = {
        k.replace("module.", "", 1): v
        for k, v in checkpoint["model_state_dict"].items()
    }
    model.load_state_dict(new_state_dict)
    return model.cuda() if rank is None else model.to(rank)


def save_checkpoint(
    path, epoch, model, optimizer, scheduler, val_match_rate, initialized=None
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_match_rate": val_match_rate,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }
    if initialized is not None:
        checkpoint["initialized"] = initialized
    torch.save(checkpoint, path)


@torch.no_grad()
def encode_latents(model, loader):
    """Encoder latents (before quantization) of every atom in ``loader``, as an (N, D) tensor."""
    zes = [
        model.module(data, train_codebook=True, encode_only=True).flatten(0, -2)
        for data in loader
    ]
    return torch.cat(zes, 0)


def load_ckpt(model, ckpt, device, ddp_model=False):
    checkpoint = torch.load(ckpt, map_location=device)
    start_epoch = checkpoint["epoch"]
    val_match_rate = checkpoint["val_match_rate"]
    try:
        if not ddp_model:
            new_state_dict = {
                k.replace("module.", "", 1): v
                for k, v in checkpoint["model_state_dict"].items()
            }
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded whole model from checkpoint {}".format(ckpt))
        if "initialized" in checkpoint:
            module = model.module if ddp_model else model
            module.quantize.initialized = checkpoint["initialized"]
    except RuntimeError:
        # Checkpoint of a model with a different quantizer (e.g. the FlowVQ auto-encoder used to
        # initialize a baseline): load the encoder & decoder only.
        if not ddp_model:
            new_state_dict = {
                k.replace("module.", "", 1): v
                for k, v in checkpoint["model_state_dict"].items()
                if "quantize" not in k
            }
            model.load_state_dict(new_state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print("Loaded encoder & decoder only from checkpoint {}".format(ckpt))
    return model, start_epoch, val_match_rate


def train(rank, world_size, args, config):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    # Seed inside the spawned worker: seeding the parent process does not carry over to mp.spawn.
    if args.use_seed:
        set_seed(args.seed)

    # initialize
    dataset = CrystalDataset(args.dataset, config=config)
    all_scaled_matrix = dataset.data["matrix"]
    dataset.matrix_scaler = get_scaler(args, all_scaled_matrix)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=config.training.batch_size,
        collate_fn=collate,
    )
    fm_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=config.training.batch_size,
        collate_fn=collate,
    )

    if rank == 0:
        val = CrystalDataset(args.dataset, config=config, val=True)
        val.matrix_scaler = get_scaler(args)
        val_loader = DataLoader(
            val,
            shuffle=False,
            batch_size=config.training.batch_size,
            collate_fn=collate,
        )
        fm_val_loader = DataLoader(
            val,
            shuffle=False,
            batch_size=config.training.batch_size,
            collate_fn=collate,
        )

    device = torch.device(f"cuda:{rank}")
    model = VQVAE(config).to(device)
    print(
        "Model parameters: {}".format(
            sum(p.numel() for p in model.parameters())
        )
    )

    start_epoch = -1
    best_val_match_rate = 0.0
    val_match_rate = 0.0
    if os.path.isfile(config.model.ckpt):
        model, start_epoch, val_match_rate = load_ckpt(
            model, config.model.ckpt, device
        )
    print("start at ", start_epoch)

    model = DDP(
        model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=True,
    )
    wandb_run_name = args.wandb_run_name
    if rank == 0:
        wandb.init(
            project=config.wandb.project,
            config=config,
            name=wandb_run_name,
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=0.0,
        betas=(0.9, 0.99),
        amsgrad=False,
    )
    if config.model.flow:
        fm_optimizer = torch.optim.Adam(
            model.module.quantize.parameters(),
            lr=config.training.lr,
            weight_decay=0.0,
            betas=(0.9, 0.99),
            amsgrad=False,
        )

    if config.use_gradscalar:
        scaler = GradScaler(enabled=True)
    if config.use_schedule:
        scheduler = ReduceLROnPlateau(
            optimizer, factor=0.6, patience=70, min_lr=1e-6
        )
    else:
        scheduler = None

    for epoch in range(
        start_epoch + 1, start_epoch + config.training.epoch + 1
    ):
        print(f"==> Epoch {epoch}")
        curr_loss = []
        curr_a_loss, curr_p_loss, curr_m_loss, curr_c_loss = [], [], [], []
        curr_fm_loss = []
        sampler.set_epoch(epoch)
        model.train()
        if not model.module.vae:
            model.module.quantize.eval()
        for i, data in enumerate(tqdm(loader)):
            optimizer.zero_grad()
            atom_loss_, pos_loss_, matrix_loss_, codebook_loss_, loss_ = model(
                data
            )
            loss, atom_loss, pos_loss, matrix_loss, codebook_loss = (
                loss_.mean(),
                atom_loss_.mean(),
                pos_loss_.mean(),
                matrix_loss_.mean(),
                codebook_loss_.mean(),
            )
            if config.use_gradscalar:
                # NOTE: the forward pass runs in full precision; only the backward call sits in the
                # autocast context (kept as in the paper runs).
                with autocast(device_type="cuda"):
                    scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            curr_loss.append(loss.detach().item())
            curr_a_loss.append(atom_loss.detach().item())
            curr_p_loss.append(pos_loss.detach().item())
            curr_m_loss.append(matrix_loss.detach().item())
            curr_c_loss.append(codebook_loss.detach().item())

            if config.model.flow and (epoch >= config.model.flow_start_epoch):

                if not model.module.quantize.initialized:
                    # ---------- FlowVQ codebook initialization (runs once) ----------
                    save_path = os.path.join(args.log, "saved_model")
                    os.makedirs(save_path, exist_ok=True)
                    ae_path = os.path.join(save_path, f"ae_{epoch - 1}.pt")
                    if not os.path.isfile(ae_path):
                        # Auto-encoder before quantization; also used to initialize the baselines.
                        save_checkpoint(
                            ae_path,
                            epoch - 1,
                            model,
                            optimizer,
                            scheduler,
                            val_match_rate,
                            initialized=model.module.quantize.initialized,
                        )
                    val_match_rate = 0.0

                    zes = encode_latents(model, fm_val_loader)

                    if not os.path.isfile(config.model.flow_init_ckpt):
                        # 1. Fit the flow until its push-forward codebook stops improving the
                        #    validation quantization error.
                        # NOTE: the main `optimizer` (over all parameters) is used here, as in the
                        # paper experiments; only the flow receives gradients.
                        print("== Training fm...")
                        last_sobol_qloss = None
                        while True:
                            curr_fm_loss = []
                            for fm_data in tqdm(fm_loader):
                                optimizer.zero_grad()
                                fm_loss = model(fm_data, train_codebook=True)
                                fm_loss.backward()
                                optimizer.step()
                                curr_fm_loss.append(fm_loss.detach().item())

                            model.module.quantize.update_codebook()
                            sobol_qloss = eval_qloss(
                                zes,
                                config.training.batch_size,
                                model.module.quantize.codebook.weight,
                            )
                            print(f"Current sobol q_loss = {sobol_qloss}")
                            if last_sobol_qloss is None:
                                last_sobol_qloss = sobol_qloss
                                patience = 0
                            elif sobol_qloss >= (last_sobol_qloss - 1e-3):
                                patience += 1
                                if patience >= FM_PATIENCE:
                                    break
                            else:
                                patience = 0
                                last_sobol_qloss = sobol_qloss
                                save_checkpoint(
                                    os.path.join(save_path, "init_flow.pt"),
                                    epoch,
                                    model,
                                    optimizer,
                                    scheduler,
                                    val_match_rate,
                                    initialized=True,
                                )
                            print(
                                f"==> Last sobol qloss: {last_sobol_qloss}, patience = {patience}"
                            )

                        if rank == 0:
                            curr_fm_loss = sum(curr_fm_loss) / len(curr_fm_loss)
                            wandb.log(
                                {"train/epoch_fm_loss": curr_fm_loss},
                                step=epoch,
                            )
                        model.module.quantize.update_codebook()
                        vis_3d_codebook(
                            zes, model.module.quantize.embed.weight.data, epoch
                        )

                        if config.model.self_correction:
                            # 2. Move the codebook vectors towards the Voronoi centers of the
                            #    training latents, with the same stopping rule.
                            print("==> Optimizing towards the voronoi center")

                            # NOTE: `patience` is deliberately NOT reset here: it carries over from
                            # step 1 (== FM_PATIENCE), so this loop stops at the first round without
                            # improvement unless an improvement happened before. This matches the
                            # paper runs.
                            patience = FM_PATIENCE
                            last_sobol_qloss = None
                            while True:
                                for fm_data in tqdm(fm_loader):
                                    optimizer.zero_grad()
                                    codebook_loss = model(
                                        fm_data,
                                        train_codebook=True,
                                        train_prior=True,
                                    )
                                    codebook_loss.backward()
                                    optimizer.step()

                                sobol_qloss = eval_qloss(
                                    zes,
                                    config.training.batch_size,
                                    model.module.quantize.codebook.weight,
                                )
                                print(f"Current sobol q_loss = {sobol_qloss}")
                                if last_sobol_qloss is None:
                                    last_sobol_qloss = sobol_qloss
                                elif sobol_qloss >= (last_sobol_qloss - 1e-3):
                                    patience += 1
                                    if patience >= FM_PATIENCE:
                                        break
                                else:
                                    patience = 0
                                    last_sobol_qloss = sobol_qloss
                                    save_checkpoint(
                                        os.path.join(
                                            save_path, "optim_init_flow.pt"
                                        ),
                                        epoch,
                                        model,
                                        optimizer,
                                        scheduler,
                                        val_match_rate,
                                        initialized=True,
                                    )

                                print(
                                    f"==> Last sobol qloss: {last_sobol_qloss}, patience = {patience}"
                                )
                            # 3. Store the flow inverse of the refined codebook as the prior codebook.
                            print("==> Updating the prior codebook")
                            model.module.quantize.prior_codebook = (
                                model.module.quantize.flow.encode(
                                    model.module.quantize.codebook.weight.data
                                )
                            )
                    else:
                        # Start from a previously initialized codebook (init_flow.pt / optim_init_flow.pt).
                        model, _, val_match_rate = load_ckpt(
                            model,
                            config.model.flow_init_ckpt,
                            device,
                            ddp_model=True,
                        )
                        sobol_qloss = eval_qloss(
                            zes,
                            config.training.batch_size,
                            model.module.quantize.codebook.weight,
                        )
                        print(f"Starting sobol q_loss = {sobol_qloss}")
                    model.module.quantize.initialized = True
                    vis_3d_codebook(
                        zes, model.module.quantize.embed.weight.data, epoch
                    )
                    curr_fm_loss = []
                else:
                    # ---------- online codebook updates (every flow_update_every epochs) ----------
                    if (
                        ((epoch - start_epoch - 1) > 0)
                        and (
                            (epoch - start_epoch - 1)
                            % config.model.flow_update_every
                            == 0
                        )
                        and (i == 0)
                    ):
                        model.module.quantize.update_codebook(optim_iter=5)
                        zes = encode_latents(model, fm_val_loader)
                        if (
                            epoch - start_epoch - 1
                        ) % config.model.flow_vis_every == 0:
                            with torch.no_grad():
                                vis_3d_codebook(
                                    zes,
                                    model.module.quantize.embed.weight.data,
                                    epoch,
                                )

                    # one flow-matching step per training batch
                    fm_optimizer.zero_grad()
                    fm_loss = model(data, train_codebook=True)
                    fm_loss.backward()
                    fm_optimizer.step()
                    curr_fm_loss.append(fm_loss.detach().item())

        curr_loss = sum(curr_loss) / len(curr_loss)
        curr_a_loss = sum(curr_a_loss) / len(curr_a_loss)
        curr_p_loss = sum(curr_p_loss) / len(curr_p_loss)
        curr_m_loss = sum(curr_m_loss) / len(curr_m_loss)
        curr_c_loss = sum(curr_c_loss) / len(curr_c_loss)
        if config.model.flow and model.module.quantize.initialized:
            curr_fm_loss = sum(curr_fm_loss) / len(curr_fm_loss)
        else:
            curr_fm_loss = 0.0
        if rank == 0:
            wandb.log(
                {
                    "epoch": epoch,
                    "epoch_loss": curr_loss,
                    "epoch_a_loss": curr_a_loss,
                    "epoch_p_loss": curr_p_loss,
                    "epoch_m_loss": curr_m_loss,
                    "epoch_c_loss": curr_c_loss,
                    "epoch_fm_loss": curr_fm_loss,
                },
                step=epoch,
            )
            print(
                {
                    "epoch": epoch,
                    "epoch_loss": curr_loss,
                    "epoch_a_loss": curr_a_loss,
                    "epoch_p_loss": curr_p_loss,
                    "epoch_m_loss": curr_m_loss,
                    "epoch_c_loss": curr_c_loss,
                    "epoch_fm_loss": curr_fm_loss,
                }
            )

            if config.use_schedule:
                scheduler.step(curr_loss)
                wandb.log({"lr": scheduler.get_last_lr()[0]}, step=epoch)

            if epoch % config.training.check == 0:
                model.eval()
                with torch.no_grad():
                    val_match_rate = (
                        model.module.eval_val(val_loader, dataset.matrix_scaler)
                        .mean()
                        .cpu()
                        .item()
                    )
                wandb.log({"val_match_rate": val_match_rate}, step=epoch)

                if best_val_match_rate < val_match_rate:
                    best_val_match_rate = val_match_rate
                    save_path = os.path.join(args.log, "saved_model")
                    os.makedirs(save_path, exist_ok=True)
                    save_checkpoint(
                        os.path.join(save_path, f"model_{epoch}.pt"),
                        epoch,
                        model,
                        optimizer,
                        scheduler,
                        val_match_rate,
                        initialized=getattr(
                            getattr(model.module, "quantize", None),
                            "initialized",
                            None,
                        ),
                    )
    args.logger.info("training completed")
    dist.destroy_process_group()
    if rank == 0:
        wandb.finish()


if __name__ == "__main__":
    args, config = parse_args_and_config()
    world_size = torch.cuda.device_count()
    mp.spawn(
        train, args=(world_size, args, config), nprocs=world_size, join=True
    )
