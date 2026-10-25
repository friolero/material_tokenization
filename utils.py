import argparse
import datetime
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import DictAction, parse_cfg
from dataloader import CrystalDataset

EPSILON = 1e-8


def get_logger(level=logging.INFO, path="log.log") -> logging.Logger:
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(path)
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)
    return logger


class StandardScalerTorch_cell(object):
    def __init__(self, min=None, max=None):
        self.min = min
        self.max = max

    def fit(self, X):
        X = X.float()
        self.min = torch.min(X)
        self.max = torch.max(X)

    def transform(self, X):
        X = X.float()
        new_X = (X - self.min) / (self.max - self.min)
        return new_X

    def inverse_transform(self, X):
        X = X.float()
        new_X = X * (self.max - self.min) + self.min
        return new_X

    def pack(self):
        return torch.tensor([self.min, self.max])

    def load(self, re):
        self.min, self.max = re


class StandardScaler:
    def __init__(self, means=None, stds=None, replace_nan_token=None):
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X):
        X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.means = np.where(
            np.isnan(self.means), np.zeros(self.means.shape), self.means
        )
        self.stds = np.where(
            np.isnan(self.stds), np.ones(self.stds.shape), self.stds
        )
        self.stds = np.where(
            self.stds == 0, np.ones(self.stds.shape), self.stds
        )

        return self

    def transform(self, X):
        X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan),
            self.replace_nan_token,
            transformed_with_nan,
        )

        return transformed_with_none

    def inverse_transform(self, X):
        X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan),
            self.replace_nan_token,
            transformed_with_nan,
        )

        return transformed_with_none


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument("--dataset", type=str, default="mp_20")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--port", type=str, default="11111")
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument(
        "--use_seed",
        action="store_true",
        help="Seed torch/numpy with --seed in every training process",
    )
    parser.add_argument("--conf_new", nargs="+", action=DictAction)

    parser.add_argument(
        "--ctn",
        action="store_true",
        help="Continue to train, restart from the last better epoch saved in saved_time",
    )
    parser.add_argument("--saved_time", type=str)
    parser.add_argument("--eval_epoch", type=str)
    args = parser.parse_args()
    if args.ctn:
        args.start_date = args.saved_time
    else:
        now = datetime.datetime.now()
        start_date = now.strftime("%Y-%m-%d %H:%M:%S")
        args.start_date = start_date

    args.cfg = os.path.join("configs", args.dataset + ".yml")
    config = parse_cfg(args.cfg, args.conf_new)
    new_config = dict2namespace(config)

    if new_config.model.optvq:
        args.wandb_run_name = "optvq"
    elif new_config.model.lfq:
        args.wandb_run_name = "lfq"
    elif new_config.model.flow:
        args.wandb_run_name = "flow"
    elif new_config.model.vae:
        args.wandb_run_name = "vae"
    else:
        args.wandb_run_name = "vqvae"
    if (new_config.model.kmeans_init) and (
        args.wandb_run_name == "optvq" or args.wandb_run_name == "vqvae"
    ):
        args.wandb_run_name += "_kmeans_init"
    if args.wandb_run_name.startswith("optvq"):
        args.wandb_run_name += f"_eps{new_config.model.optvq_eps}"
    args.wandb_run_name += (
        f"_K{new_config.model.code_num}Z{new_config.model.latent_dim}"
    )
    if new_config.model.flow:
        args.wandb_run_name += f"E{new_config.model.flow_start_epoch}F{new_config.model.flow_update_every}e_sc{new_config.model.self_correction}_oc{new_config.model.online_correction}"
    args.log = os.path.join("logs", args.dataset, args.wandb_run_name)
    os.makedirs(args.log, exist_ok=True)

    #### setup logger ####
    logger = get_logger(path=os.path.join(args.log, "running.log"))
    args.logger = logger

    device = torch.device("cuda")
    logger.info("Using device: {}".format(device))
    new_config.device = device

    torch.backends.cudnn.benchmark = True

    return args, new_config


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def get_scaler(args, scaled_matrix=None):
    matrix_scaler_path = os.path.join("data", args.dataset, "matrix_scaler.pt")
    matrix_scaler = StandardScalerTorch_cell()
    if not os.path.exists(matrix_scaler_path):
        matrix_scaler.fit(scaled_matrix)
        torch.save(matrix_scaler.pack(), matrix_scaler_path)
    else:
        matrix_scaler.load(torch.load(matrix_scaler_path))
    return matrix_scaler


class get_DataLoader:
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def get_dataloader(self, test=False):
        dataset = CrystalDataset(
            self.args.dataset, config=self.config, test=test
        )
        all_scaled_matrix = dataset.data["matrix"]
        dataset.matrix_scaler = get_scaler(self.args, all_scaled_matrix)

        if test:
            loader = DataLoader(
                dataset,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                num_workers=16,
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=self.config.training.batch_size,
                shuffle=True,
                num_workers=16,
            )
        return loader
