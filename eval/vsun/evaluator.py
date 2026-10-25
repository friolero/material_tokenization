import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.phase_diagram import PhaseDiagram

from eval.vsun.reference.utils import expand_into_subsystems
from eval.vsun.utils import get_unique, get_matches, get_mask_from_local_index, get_global_match_dict_from_local_dict, matches_to_mask

class Evaluator:
    def __init__(self, args, gen_ref, reference):
        self.args = args
        self.gen_ref = gen_ref
        self.reference = reference
        self.logger = args.logger
        self.num_samples = args.num_samples
        self.stability_threshold = args.stable_delta
        self.matcher = StructureMatcher()
    
    def energy_above_hull(self, df, e_hull_path):
        if not os.path.exists(e_hull_path):
            result = np.zeros(len(self.gen_ref))
            for chemsys, entries in tqdm(
                self.gen_ref.entries_by_chemsys.items(),
                desc="Computing energies above hull",
            ):
                result[[e.entry_id for e in entries]] = np.array(
                    self._get_energy_above_hull_per_atom_chemsys(chemsys)
                )
            out = pd.DataFrame(data={"e_above_hull_per_atom": result})
            out.index = df.index
            out.to_json(e_hull_path)
        else:
            self.logger.info("Energy above hull is ready!")
            e_hull = pd.read_json(e_hull_path)
            result = e_hull['e_above_hull_per_atom']
        # np.save("e_hull.npy", result)
        self.vs_indicator = result <= self.stability_threshold
    
    def _get_energy_above_hull_per_atom_chemsys(self, chemsys: str) -> list[float]:
        """Returns a list of energies above hull per atom for a given chemical system."""
        phase_diagram = self._get_phase_diagram(chemsys)
        e_above_hull = []
        for e in self.gen_ref.entries_by_chemsys[chemsys]:
            try:
                e_above_hull.append(phase_diagram.get_e_above_hull(entry=e, allow_negative=True))
            except:
                e_above_hull.append(100)
        
        for e, ehull in zip(self.gen_ref.entries_by_chemsys[chemsys], e_above_hull):
            self.logger.debug(
                f"{e.composition.reduced_formula}: energy above hull {ehull} (threshold {self.stability_threshold})"
            )
        return e_above_hull
    
    def _get_phase_diagram(self, chemical_system: str) -> PhaseDiagram:
        """Returns the phase diagram for a given chemical system."""
        subsys = expand_into_subsystems(chemical_system)
        reference_entries = [
            entry
            for s in subsys
            for key in ["-".join(sorted(s))]
            for entry in self.reference.entries_by_chemsys.get(key, [])
            if not np.isnan(
                entry.energy
            )  # skip disordered structures, which have nan energy currently
        ]
        assert len(reference_entries) > 0, f"No reference data for {chemical_system}."
        return PhaseDiagram(reference_entries)
    
    def valid_stable(self):
        vs = self.vs_indicator.sum() / self.num_samples
        self.logger.info("vs: {:.3f}".format(vs))
        return vs
    
    def valid_unique(self):
        local_index: dict[str, List[int]] = {}
        for reduced_formula, data_entries in tqdm(
            self.gen_ref.entries_by_reduced_formula.items(),
            desc="Finding unique structures by reduced formula",
        ):
            structures = [e.structure for e in data_entries]
            assert all(
                [s.is_ordered for s in structures]
            ), "OrderedDatasetUniquenessComputer only works for ordered structures."
            local_index[reduced_formula] = get_unique(self.matcher, structures)

        self.vu_indicator = get_mask_from_local_index(self.gen_ref.entries_by_reduced_formula, local_index)
        vu = self.vu_indicator.sum() / self.num_samples
        self.logger.info("vu: {:.3f}".format(vu))
        return vu
    
    def valid_novel(self):
        local_match_indices: dict[str, dict[int, list[int]]] = {}
        grouped_dataset_entries = self.gen_ref.entries_by_reduced_formula
        grouped_reference_entries = self.reference.entries_by_reduced_formula

        for group_key, data_entries in tqdm(
            grouped_dataset_entries.items(),
            desc="Finding novel structures",
        ):
            data_structures = [e.structure for e in data_entries]
            reference_structures = [
                e.structure for e in grouped_reference_entries.get(group_key, [])
            ]
            matches = get_matches(
                self.matcher,
                data_structures,
                reference_structures,
            )
            local_match_indices[group_key] = matches

        global_match_dict = get_global_match_dict_from_local_dict(
            grouped_dataset_entries, grouped_reference_entries, local_match_indices
        )

        self.vn_indicator = np.logical_not(matches_to_mask(global_match_dict.keys(), len(self.gen_ref)))
        vn = self.vn_indicator.sum() / self.num_samples
        self.logger.info("vn: {:.3f}".format(vn))
        return vn

    def get_metrics(self):
        result = {}
        result['v'] = len(self.gen_ref) / self.num_samples
        result['vs'] = self.valid_stable()
        result['vu'] = self.valid_unique()
        result['vn'] = self.valid_novel()

        result['vsu'] = (self.vs_indicator * self.vu_indicator).sum() / self.num_samples
        result['vsn'] = (self.vs_indicator * self.vn_indicator).sum() / self.num_samples
        result['vsun'] = (self.vs_indicator * self.vu_indicator * self.vn_indicator).sum() / self.num_samples

        result['vun'] = (self.vn_indicator * self.vu_indicator).sum() / self.num_samples
        result['s in vun'] = result['vsun'] / result['vun']
        return result
    
