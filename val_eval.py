"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os
from functools import partial
from typing import Any, Dict, List

import numpy as np
import torch
import wandb
from pymatgen.analysis.structure_matcher import StructureMatcher
from tqdm import tqdm
from typing import Iterable, Literal

from eval.utils import Crystal, compute_cov
from joblib import Parallel, delayed, parallel_config
import math


class CrystalReconstructionEvaluator:
    """Evaluator for crystal reconstruction tasks. Can be used within a Lightning module, appending
    predictions and ground truths during training and computing metrics at the end of an epoch, or
    can be used as a standalone object to evaluate predictions on a dataset.

    Args:
        stol (float): StructureMatcher tolerance for matching sites.
        angle_tol (float): StructureMatcher tolerance for matching angles.
        ltol (float): StructureMatcher tolerance for matching lengths.
    """

    def __init__(self, stol=0.5, angle_tol=10, ltol=0.3, device="cpu"):
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.pred_arrays_list = []  # list of Dict[str, np.array] predictions
        self.gt_arrays_list = []  # list of Dict[str, np.array] ground truths
        self.pred_crys_list = []  # list of Crystal predictions
        self.gt_crys_list = []  # list of Crystal ground truths
        self.device = device

    def append_pred_array(self, pred: Dict[str, np.array]):
        """Append a prediction to the evaluator."""
        self.pred_arrays_list.append(pred)

    def append_gt_array(self, gt: Dict[str, np.array]):
        """Append a ground truth to the evaluator."""
        self.gt_arrays_list.append(gt)

    def clear(self):
        """Clear the stored predictions and ground truths, to be used at the end of an epoch."""
        self.pred_arrays_list = []
        self.gt_arrays_list = []
        self.pred_crys_list = []
        self.gt_crys_list = []

    def _arrays_to_crystals(self, save: bool = False, save_dir: str = "", cart: bool = False, la: bool = False, get_fps: bool = False):
        """Convert stored predictions and ground truths to Crystal objects for evaluation."""
        self.pred_crys_list = joblib_map(
            partial(
                array_dict_to_crystal,
                save=save,
                save_dir_name=f"{save_dir}/pred",
                cart = cart, la = la, get_fps = get_fps
            ),
            self.pred_arrays_list,
            n_jobs=-4,
            inner_max_num_threads=1,
            desc=f"    Pred to Crystal",
            total=len(self.pred_arrays_list),
        )
        self.gt_crys_list = joblib_map(
            partial(
                array_dict_to_crystal,
                save=save,
                save_dir_name=f"{save_dir}/gt",
                gt=True,
                cart = cart, la = la, get_fps = get_fps
            ),
            self.gt_arrays_list,
            n_jobs=-4,
            inner_max_num_threads=1,
            desc=f"    G.T. to Crystal",
            total=len(self.gt_arrays_list),
        )

    def _get_metrics(self, pred, gt, is_valid):
        if not is_valid:
            return float("inf")
        try:
            rms_dist = self.matcher.get_rms_dist(pred.structure, gt.structure)
            rms_dist = float("inf") if rms_dist is None else rms_dist[0]
            return rms_dist
        except Exception:
            return float("inf")

    def get_metrics(self, save: bool = False, save_dir: str = "", cart: bool = False, la: bool = False, get_fps: bool = False) -> Dict[str, Any]:
        """Compute the match rate and avg. RMS distance between predictions and ground truths.

        Note: self.rms_dists can be used to access RMSD per sample but is not returned.

        Returns:
            Dict: Dictionary of metrics, including match rate and avg. RMSD.
        """
        assert len(self.pred_arrays_list) == len(
            self.gt_arrays_list
        ), "Number of predictions and ground truths must match."

        # Convert predictions and ground truths to Crystal objects
        self._arrays_to_crystals(save, save_dir, cart, la, get_fps)

        # Check validity of predictions and ground truths
        validity = [
            c1.valid and c2.valid for c1, c2 in zip(self.pred_crys_list, self.gt_crys_list)
        ]

        self.rms_dists = []
        for i in tqdm(range(len(self.pred_crys_list)), desc="   Reconstruction eval"):
            self.rms_dists.append(
                self._get_metrics(self.pred_crys_list[i], self.gt_crys_list[i], validity[i])
            )
        self.rms_dists = torch.tensor(self.rms_dists, device=self.device)
        match_rate = (~torch.isinf(self.rms_dists)).long().float()
        if match_rate.sum() == 0:
            # No valid predictions --> return large RMSD for logging purposes
            results = {
                "match_rate": match_rate,
                "rms_dist": torch.tensor([10.0] * len(match_rate), device=self.device),
            }
        else:
            results = {
                "match_rate": match_rate,
                "rms_dist": self.rms_dists[~torch.isinf(self.rms_dists)],
            }

        if get_fps:
            results["cov"] = compute_cov(self.pred_crys_list, self.gt_crys_list)
        return results

def array_dict_to_crystal(
    x: dict[str, np.ndarray],
    save: bool = False,
    save_dir_name: str = "",
    gt: bool = False, cart: bool = False, la: bool = False, get_fps: bool = False
) -> Crystal:
    # Check if the lattice angles are in a valid range
    if la:
        angles = x["angle"]
    else:
        angles = matrix2angles(x["matrices"])
    if np.all(50 < angles) and np.all(angles < 130):
        crys = Crystal(x, gt, cart, la, test_validity=False, get_fps=get_fps)
        if crys.valid and save:
            os.makedirs(save_dir_name, exist_ok=True)
            crys.structure.to(os.path.join(save_dir_name, f"crystal_{x['sample_idx']}.cif"))
    else:
        # returns an absurd crystal
        dict = {}
        for k, v in x.items():
            dict[k] = np.zeros_like(v)
        crys = Crystal(dict, gt, cart, la, test_validity=False, get_fps=get_fps)
    return crys

def matrix2angles(L):
    a, b, c = L[:, 0], L[:, 1], L[:, 2]
    la, lb, lc = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)

    # angles in radians
    alpha = np.arccos(np.clip(np.dot(b, c) / (lb * lc), -1.0, 1.0))
    beta  = np.arccos(np.clip(np.dot(a, c) / (la * lc), -1.0, 1.0))
    gamma = np.arccos(np.clip(np.dot(a, b) / (la * lb), -1.0, 1.0))

    # convert to degrees
    alpha_deg = np.degrees(alpha)
    beta_deg  = np.degrees(beta)
    gamma_deg = np.degrees(gamma)
    return np.array([alpha_deg, beta_deg, gamma_deg])

class ParallelTqdm(Parallel):
    """joblib.Parallel, but with a tqdm progressbar.

    Adapted from:
    - https://github.com/facebookresearch/flowmm
    - https://gist.github.com/tsvikas/5f859a484e53d4ef93400751d0a116de

    Additional parameters:
    ----------------------
    total_tasks: int, default: None
        the number of expected jobs. Used in the tqdm progressbar.
        If None, try to infer from the length of the called iterator, and
        fallback to use the number of remaining items as soon as we finish
        dispatching.
        Note: use a list instead of an iterator if you want the total_tasks
        to be inferred from its length.

    desc: str, default: None
        the description used in the tqdm progressbar.

    disable_progressbar: bool, default: False
        If True, a tqdm progressbar is not used.

    show_joblib_header: bool, default: False
        If True, show joblib header before the progressbar.

    Removed parameters:
    -------------------
    verbose: will be ignored


    Usage:
    ------
    >>> from joblib import delayed
    >>> from time import sleep
    >>> ParallelTqdm(n_jobs=-1)([delayed(sleep)(.1) for _ in range(10)])
    80%|████████  | 8/10 [00:02<00:00,  3.12tasks/s]
    """

    def __init__(
        self,
        *,
        total_tasks: int | None = None,
        desc: str | None = None,
        disable_progressbar: bool = False,
        show_joblib_header: bool = False,
        **kwargs,
    ):
        if "verbose" in kwargs:
            raise ValueError(
                "verbose is not supported. " "Use show_progressbar and show_joblib_header instead."
            )
        super().__init__(verbose=(1 if show_joblib_header else 0), **kwargs)
        self.total_tasks = total_tasks
        self.desc = desc
        self.disable_progressbar = disable_progressbar
        self.progress_bar: tqdm | None = None

    def __call__(self, iterable):
        try:
            if self.total_tasks is None:
                # try to infer total_tasks from the length of the called iterator
                try:
                    self.total_tasks = len(iterable)
                except (TypeError, AttributeError):
                    pass
            # call parent function
            return super().__call__(iterable)
        finally:
            # close tqdm progress bar
            if self.progress_bar is not None:
                self.progress_bar.close()

    __call__.__doc__ = Parallel.__call__.__doc__

    def dispatch_one_batch(self, iterator):
        # start progress_bar, if not started yet.
        if self.progress_bar is None:
            self.progress_bar = tqdm(
                desc=self.desc,
                total=self.total_tasks,
                disable=self.disable_progressbar,
                unit="tasks",
            )
        # call parent function
        return super().dispatch_one_batch(iterator)

    dispatch_one_batch.__doc__ = Parallel.dispatch_one_batch.__doc__

    def print_progress(self):
        """Display the process of the parallel execution using tqdm."""
        # if we finish dispatching, find total_tasks from the number of remaining items
        if self.total_tasks is None and self._original_iterator is None:
            self.total_tasks = self.n_dispatched_tasks
            self.progress_bar.total = self.total_tasks
            self.progress_bar.refresh()
        # update progressbar
        self.progress_bar.update(self.n_completed_tasks - self.progress_bar.n)


def joblib_map(
    func: callable,
    iterable: Iterable,
    n_jobs: int = 1,
    inner_max_num_threads: int | None = None,
    desc: str | None = None,
    total: int | None = None,
    backend: Literal["sequential", "loky", "threading", "multiprocessing"] = "loky",
) -> list:
    if backend != "loky" and inner_max_num_threads is not None:
        print(f"{backend=} does not support {inner_max_num_threads=}, setting to None.")
        inner_max_num_threads = None

    if backend != "sequential":
        with parallel_config(backend=backend, inner_max_num_threads=inner_max_num_threads):
            if backend == "loky":
                results = ParallelTqdm(n_jobs=n_jobs, total_tasks=total, desc=desc)(
                    delayed(func)(i) for i in iterable
                )
            else:
                results = Parallel(n_jobs=n_jobs)(delayed(func)(i) for i in iterable)
    else:
        results = [func(i) for i in tqdm(iterable, desc=desc, total=total)]
    return results

