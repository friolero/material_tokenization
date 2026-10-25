import numpy as np
import torch

from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist

import os
from eval.cdvae.constants import CompScalerMeans, CompScalerStds
from utils import StandardScaler, get_DataLoader, get_scaler
import json


CompScaler = StandardScaler(
    means=np.array(CompScalerMeans),
    stds=np.array(CompScalerStds),
    replace_nan_token=0.)

def load_data(file_path):
    file_path = str(file_path)
    if file_path[-3:] == 'npy':
        data = np.load(file_path, allow_pickle=True).item()
        for k, v in data.items():
            if k == 'input_data_batch':
                for k1, v1 in data[k].items():
                    data[k][k1] = torch.from_numpy(v1)
            else:
                data[k] = torch.from_numpy(v).unsqueeze(0)
    else:
        data = torch.load(file_path, weights_only=False)
    return data

def get_ckpt(path, args):
    if args.eval_epoch == -1:
        all_files = os.listdir(path)
        all_pths = []
        for file in all_files:
            if os.path.splitext(file)[1]==".pt":
                all_pths.append(file)
        all_pths.sort()
        return os.path.join(path, all_pths[-1])
    else:
        return os.path.join(path, 'model_'+str(args.eval_epoch)+'.pt')

def get_index2path(ld_kwargs):
    index2path = os.path.join('data', ld_kwargs.dataset, 'index2type.json')
    with open(index2path, 'r') as f:
        index2type = json.load(f)
    index2type = {int(k): v for k, v in index2type.items()}
    return index2type

def get_true_crystal_array_list(args, config):
    from eval.cdvae.generate import deal_result
    def compose(matrix, type, frac):
        b, _, column = type.shape
        pad = torch.zeros(b, 3, column)
        upper = torch.cat([matrix, pad], -1)
        bottom = torch.cat([frac, type], -1)
        whole = torch.cat([upper, bottom], 1)
        return whole
    
    test_dataloader = get_DataLoader(args, config).get_dataloader(test=True)
    x = []
    for i, data in enumerate(test_dataloader):
        matrices, fracs, types  = data['matrix'], data['frac'], data['type']
        x.append(compose(matrices, types, fracs))
    x = torch.cat(x, dim=0)
    matrix_scaler = get_scaler(args)
    index2type = get_index2path(args)

    batch_frac_coords, batch_num_atoms, batch_atom_types, batch_matrices = deal_result(x, matrix_scaler, index2type)
    crystal_array_list = get_crystals_list(batch_frac_coords[0], batch_atom_types[0], batch_matrices[0], batch_num_atoms[0])
    return crystal_array_list

def get_crystal_array_list(file_path, batch_idx=0):
    data = load_data(file_path)
    crys_array_list = get_crystals_list(
        data['frac_coords'][batch_idx],
        data['atom_types'][batch_idx],
        data['matrices'][batch_idx],
        data['num_atoms'][batch_idx])

    return crys_array_list

def get_crystals_list(
        frac_coords, atom_types, matrices, num_atoms):
    """
    args:
        frac_coords: (num_atoms, 3)
        atom_types: (num_atoms)
        matrices: (num_crystals)
        num_atoms: (num_crystals)
    """
    assert frac_coords.size(0) == atom_types.size(0) == num_atoms.sum()
    assert (matrices-matrices.transpose(1,2)).sum() == 0

    start_idx = 0
    crystal_array_list = []
    for batch_idx, num_atom in enumerate(num_atoms.tolist()):
        cur_frac_coords = frac_coords.narrow(0, start_idx, num_atom)
        cur_atom_types = atom_types.narrow(0, start_idx, num_atom)
        cur_matrices = matrices[batch_idx]
        crystal_array_list.append({
            'frac_coords': cur_frac_coords.detach().cpu().numpy(),
            'atom_types': cur_atom_types.long().detach().cpu().numpy(),
            'matrices': cur_matrices.detach().cpu().numpy(),
        })
        start_idx = start_idx + num_atom
    return crystal_array_list

def get_fp_pdist(fp_array):
    if isinstance(fp_array, list):
        fp_array = np.array(fp_array)
    fp_pdists = pdist(fp_array)
    return fp_pdists.mean()

def filter_fps(struc_fps, comp_fps):
    assert len(struc_fps) == len(comp_fps)

    filtered_struc_fps, filtered_comp_fps = [], []

    for struc_fp, comp_fp in zip(struc_fps, comp_fps):
        if struc_fp is not None and comp_fp is not None:
            filtered_struc_fps.append(struc_fp)
            filtered_comp_fps.append(comp_fp)
    return filtered_struc_fps, filtered_comp_fps


def compute_cov(crys, gt_crys,
                struc_cutoff, comp_cutoff, num_gen_crystals=None):
    struc_fps = [c.struct_fp for c in crys]
    comp_fps = [c.comp_fp for c in crys]
    gt_struc_fps = [c.struct_fp for c in gt_crys]
    gt_comp_fps = [c.comp_fp for c in gt_crys]

    assert len(struc_fps) == len(comp_fps)
    assert len(gt_struc_fps) == len(gt_comp_fps)

    # Use number of crystal before filtering to compute COV
    if num_gen_crystals is None:
        num_gen_crystals = len(struc_fps)

    struc_fps, comp_fps = filter_fps(struc_fps, comp_fps)
    
    gt_comp_fps = CompScaler.transform(gt_comp_fps)
    comp_fps = CompScaler.transform(comp_fps)

    struc_fps = np.array(struc_fps)
    gt_struc_fps = np.array(gt_struc_fps)
    comp_fps = np.array(comp_fps)
    gt_comp_fps = np.array(gt_comp_fps)

    struc_pdist = cdist(struc_fps, gt_struc_fps)
    comp_pdist = cdist(comp_fps, gt_comp_fps)

    struc_recall_dist = struc_pdist.min(axis=0)
    struc_precision_dist = struc_pdist.min(axis=1)
    comp_recall_dist = comp_pdist.min(axis=0)
    comp_precision_dist = comp_pdist.min(axis=1)

    cov_recall = np.mean(np.logical_and(
        struc_recall_dist <= struc_cutoff,
        comp_recall_dist <= comp_cutoff))
    cov_precision = np.sum(np.logical_and(
        struc_precision_dist <= struc_cutoff,
        comp_precision_dist <= comp_cutoff)) / num_gen_crystals

    metrics_dict = {
        'cov_recall': cov_recall,
        'cov_precision': cov_precision,
        'amsd_recall': np.mean(struc_recall_dist),
        'amsd_precision': np.mean(struc_precision_dist),
        'amcd_recall': np.mean(comp_recall_dist),
        'amcd_precision': np.mean(comp_precision_dist),
    }

    combined_dist_dict = {
        'struc_recall_dist': struc_recall_dist.tolist(),
        'struc_precision_dist': struc_precision_dist.tolist(),
        'comp_recall_dist': comp_recall_dist.tolist(),
        'comp_precision_dist': comp_precision_dist.tolist(),
    }

    return metrics_dict, combined_dist_dict
