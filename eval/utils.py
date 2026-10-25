import itertools
import os
from collections import Counter
from multiprocessing import Process, Queue

import numpy as np
import smact
from matminer.featurizers.composition.composite import ElementProperty
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from pathos.multiprocessing import ProcessingPool as Pool
from pymatgen.core import Element, Structure
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from smact.screening import pauling_test
from tqdm import tqdm

from eval.cdvae.constants import CompScalerMeans, CompScalerStds
from utils import StandardScaler

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset("magpie")
CompScaler = StandardScaler(
    means=np.array(CompScalerMeans),
    stds=np.array(CompScalerStds),
    replace_nan_token=0.0,
)


class Crystal(object):
    def __init__(
        self,
        crys_array_dict,
        gt=False,
        cart=False,
        la=False,
        test_validity=True,
        get_fps=False,
    ):
        self.cart, self.la = cart, la
        if self.cart:
            self.card_coords = crys_array_dict["cart_coords"]
        else:
            self.frac_coords = crys_array_dict["frac_coords"]

        if self.la:
            self.length = crys_array_dict["length"]
            self.angle = crys_array_dict["angle"]
        else:
            self.matrix = crys_array_dict["matrices"]
        self.atom_types = crys_array_dict["atom_types"]
        self.structure = None
        self.get_structure()
        if get_fps:
            try:
                self.get_fingerprints()
            except:
                self.struct_fp = None
                self.comp_fp = None
        if self.structure is None:
            self.valid = False
        else:
            try:
                self.get_composition()
                if gt:
                    self.valid = True
                else:
                    if test_validity:
                        self.get_validity()
                    else:
                        self.valid = True
            except:
                self.valid = False

    def get_structure(self):
        try:
            if self.la:
                lattice = Lattice.from_parameters(
                    *(self.length.tolist() + self.angle.tolist())
                )
            else:
                lattice = self.matrix
            if self.cart:
                self.structure = Structure(
                    lattice=lattice,
                    species=self.atom_types,
                    coords=self.card_coords,
                    coords_are_cartesian=True,
                )
            else:
                self.structure = Structure(
                    lattice=lattice,
                    species=self.atom_types,
                    coords=self.frac_coords,
                    coords_are_cartesian=False,
                )
            self.constructed = True
        except Exception:
            self.constructed = False

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [
            (elem, elem_counter[elem]) for elem in sorted(elem_counter.keys())
        ]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype("int").tolist())

    def get_validity(self):
        self.comp_valid = self.smact_validity(self.elems, self.comps)
        if self.constructed:
            self.struct_valid = self.structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    def structure_validity(
        self, structure: Structure, cutoff: float = 0.5
    ) -> bool:
        dist_mat = structure.distance_matrix
        dist_mat = dist_mat + np.diag(
            np.ones(dist_mat.shape[0]) * (cutoff + 10.0)
        )
        if dist_mat.min() < cutoff or structure.volume < 0.1:
            return False
        else:
            return True

    def smact_validity(
        self,
        comp: tuple[int, ...] | tuple[str, ...],
        count: tuple[int, ...],
        use_pauling_test: bool = True,
        include_alloys: bool = True,
        include_cutoff: bool = False,
        use_element_symbol: bool = False,
    ) -> bool:
        """Computes SMACT validity.

        Args:
            comp: Tuple of atomic number or element names of elements in a crystal.
            count: Tuple of counts of elements in a crystal.
            use_pauling_test: Whether to use electronegativity test. That is, at least in one
                combination of oxidation states, the more positive the oxidation state of a site,
                the lower the electronegativity of the element for all pairs of sites.
            include_alloys: if True, returns True without checking charge balance or electronegativity
                if the crystal is an alloy (consisting only of metals) (default: True).
            include_cutoff: assumes valid crystal if the combination of oxidation states is more
                than 10^6 (default: False).

        Returns:
            True if the crystal is valid, False otherwise.
        """
        try:
            assert len(comp) == len(count)
            if use_element_symbol:
                elem_symbols = comp
            else:
                elem_symbols = tuple(
                    [str(Element.from_Z(Z=elem)) for elem in comp]
                )
            space = smact.element_dictionary(elem_symbols)
            smact_elems = [e[1] for e in space.items()]
            electronegs = [e.pauling_eneg for e in smact_elems]
            ox_combos = [e.oxidation_states for e in smact_elems]
            if len(set(elem_symbols)) == 1:
                return True
            if include_alloys:
                is_metal_list = [
                    elem_s in smact.metals for elem_s in elem_symbols
                ]
                if all(is_metal_list):
                    return True

            threshold = np.max(count)
            compositions = []
            n_comb = np.prod([len(ls) for ls in ox_combos])
            if n_comb > 1e6 and include_cutoff:
                return True
            for ox_states in itertools.product(*ox_combos):
                stoichs = [(c,) for c in count]
                # Test for charge balance
                cn_e, cn_r = smact.neutral_ratios(
                    ox_states, stoichs=stoichs, threshold=threshold
                )
                # Electronegativity test
                if cn_e:
                    if use_pauling_test:
                        try:
                            electroneg_OK = pauling_test(ox_states, electronegs)
                        except TypeError:
                            electroneg_OK = True
                    else:
                        electroneg_OK = True
                    if electroneg_OK:
                        for ratio in cn_r:
                            compositions.append(
                                tuple([elem_symbols, ox_states, ratio])
                            )
            compositions = [(i[0], i[2]) for i in compositions]
            compositions = list(set(compositions))
            if len(compositions) > 0:
                return True
            else:
                return False
        except:
            return False

    def get_fingerprints(self):
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        try:
            site_fps = [
                CrystalNNFP.featurize(self.structure, i)
                for i in range(len(self.structure))
            ]
        except Exception:
            # counts crystal as invalid if fingerprint cannot be constructed.
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)


def make_crystals_mp(crys_array_list):
    def call_crystal_worker(x, q):
        try:
            result = Crystal(x)
            q.put(result)
        except Exception as e:
            q.put(None)

    def wrapper(x):
        timeout = 60  # sec
        q = Queue(timeout)
        p = Process(target=call_crystal_worker, args=(x, q))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            return None
        return q.get() if not q.empty() else None

    n_workers = int(0.8 * os.cpu_count())
    pool = Pool(nodes=n_workers)
    gen_crys = list(
        tqdm(pool.imap(wrapper, crys_array_list), total=len(crys_array_list))
    )
    return gen_crys


COV_Cutoffs = {"struc": 0.4, "comp": 10.0}


def compute_cov(crys, gt_crys):
    struc_fps, comp_fps, gt_struc_fps, gt_comp_fps = [], [], [], []
    for c in crys:
        if c.struct_fp is not None:
            struc_fps.append(c.struct_fp)
        else:
            struc_fps.append([-100] * 61)
        if c.comp_fp is not None:
            comp_fps.append(c.comp_fp)
        else:
            comp_fps.append([-100] * 132)

    for c in gt_crys:
        if c.struct_fp is not None:
            gt_struc_fps.append(c.struct_fp)
        else:
            gt_struc_fps.append([100] * 61)
        if c.comp_fp is not None:
            gt_comp_fps.append(c.comp_fp)
        else:
            gt_comp_fps.append([100] * 132)

    assert len(struc_fps) == len(comp_fps)
    assert len(gt_struc_fps) == len(gt_comp_fps)

    num_gen_crystals = len(struc_fps)

    gt_comp_fps = CompScaler.transform(gt_comp_fps)
    comp_fps = CompScaler.transform(comp_fps)

    struc_fps = np.array(struc_fps)
    gt_struc_fps = np.array(gt_struc_fps)
    comp_fps = np.array(comp_fps)
    gt_comp_fps = np.array(gt_comp_fps)

    struc_dist = np.linalg.norm(struc_fps - gt_struc_fps, axis=1)
    comp_dist = np.linalg.norm(comp_fps - gt_comp_fps, axis=1)

    cov = (
        np.sum(
            np.logical_and(
                struc_dist <= COV_Cutoffs["struc"],
                comp_dist <= COV_Cutoffs["comp"],
            )
        )
        / num_gen_crystals
    )

    return cov
