import numpy as np
import os
import json

import numpy as np
from pathlib import Path
from scipy.stats import wasserstein_distance

from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty

from eval.cdvae.utils import CompScaler, get_fp_pdist, get_crystal_array_list, compute_cov, get_true_crystal_array_list
from eval.utils import make_crystals_mp

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')

Percentiles = {
    'mp20': np.array([-3.17562208, -2.82196882, -2.52814761]),
    'carbon': np.array([-154.527093, -154.45865733, -154.44206825]),
    'perovskite': np.array([0.43924842, 0.61202443, 0.7364607]),
}

COV_Cutoffs = {
    'mp20': {'struc': 0.4, 'comp': 10.},
    'carbon': {'struc': 0.2, 'comp': 4.},
    'perovskite': {'struc': 0.2, 'comp': 4},
}

class GenEval(object):
    def __init__(self, pred_crys, gt_crys, n_samples=1000, eval_model_name=None, logger=None):
        self.crys = pred_crys
        self.gt_crys = gt_crys
        self.n_samples = n_samples
        self.eval_model_name = eval_model_name
        self.logger = logger
        valid_crys = [c for c in pred_crys if c.valid]

        if len(valid_crys) >= n_samples:
            sampled_indices = np.random.choice(
                len(valid_crys), n_samples, replace=False)
            self.valid_samples = [valid_crys[i] for i in sampled_indices]
        else:
            raise Exception(
                f'not enough valid crystals in the predicted set: {len(valid_crys)}/{n_samples}')

    def get_validity(self):
        self.logger.info("Calculate comp_valid, struct_valid, valid...")
        comp_valid = np.array([c.comp_valid for c in self.crys]).mean()
        struct_valid = np.array([c.struct_valid for c in self.crys]).mean()
        valid = np.array([c.valid for c in self.crys]).mean()
        return {'comp_valid': comp_valid,
                'struct_valid': struct_valid,
                'valid': valid}

    def get_comp_diversity(self):
        self.logger.info("Calculate comp_div...")
        comp_fps = [c.comp_fp for c in self.valid_samples]
        comp_fps = CompScaler.transform(comp_fps)
        comp_div = get_fp_pdist(comp_fps)
        return {'comp_div': comp_div}

    def get_struct_diversity(self):
        self.logger.info("Calculate struct_div...")
        return {'struct_div': get_fp_pdist([c.struct_fp for c in self.valid_samples])}

    def get_density_wdist(self):
        self.logger.info("Calculate wdist_density...")
        pred_densities = [c.structure.density for c in self.valid_samples]
        gt_densities = [c.structure.density for c in self.gt_crys]
        wdist_density = wasserstein_distance(pred_densities, gt_densities)
        return {'wdist_density': wdist_density}

    def get_num_elem_wdist(self):
        self.logger.info("Calculate wdist_num_elems...")
        pred_nelems = [len(set(c.structure.species))
                       for c in self.valid_samples]
        gt_nelems = [len(set(c.structure.species)) for c in self.gt_crys]

        wdist_num_elems = wasserstein_distance(pred_nelems, gt_nelems)
        return {'wdist_num_elems': wdist_num_elems}

    def get_coverage(self):
        self.logger.info("Calculate coverage...")
        cutoff_dict = COV_Cutoffs[self.eval_model_name]
        (cov_metrics_dict, combined_dist_dict) = compute_cov(
            self.crys, self.gt_crys,
            struc_cutoff=cutoff_dict['struc'],
            comp_cutoff=cutoff_dict['comp'])
        return cov_metrics_dict

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_validity())
        metrics.update(self.get_density_wdist())
        metrics.update(self.get_num_elem_wdist())
        metrics.update(self.get_coverage())
        metrics.update(self.get_comp_diversity())
        metrics.update(self.get_struct_diversity())
        return metrics

def get_metrics(args, config):
    logger = args.logger
    all_metrics = {}
    eval_model_name = config.eval.eval_model_name

    logger.info('Extract generated crystals...')
    crys_array_list = get_crystal_array_list(os.path.join(args.model_path, "eval_gen.pt"))
    gen_crys = make_crystals_mp(crys_array_list)

    logger.info('Extract true crystals...')
    true_crystal_array_list = get_true_crystal_array_list(args, config)
    gt_crys = make_crystals_mp(true_crystal_array_list)

    gen_evaluator = GenEval(
        gen_crys, gt_crys, eval_model_name=eval_model_name, logger=logger)
    gen_metrics = gen_evaluator.get_metrics()
    all_metrics.update(gen_metrics)

    metrics_out_file = 'cdvae_num_'+str(args.num_samples)+'.json'
    metrics_out_file = os.path.join(args.model_path, metrics_out_file)
    
    logger.info("Done!")

    # only overwrite metrics computed in the new run.
    if Path(metrics_out_file).exists():
        with open(metrics_out_file, 'r') as f:
            written_metrics = json.load(f)
            if isinstance(written_metrics, dict):
                written_metrics.update(all_metrics)
            else:
                with open(metrics_out_file, 'w') as f:
                    json.dump(all_metrics, f)
        if isinstance(written_metrics, dict):
            with open(metrics_out_file, 'w') as f:
                json.dump(written_metrics, f)
    else:
        with open(metrics_out_file, 'w') as f:
            json.dump(all_metrics, f)