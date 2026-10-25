# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from ase import Atoms

from mattersim.forcefield.potential import Potential
from mattersim.datasets.utils.build import build_dataloader

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.constraints import Filter
from ase.filters import ExpCellFilter, FrechetCellFilter
from ase.optimize import BFGS, FIRE
from ase.optimize.optimize import Optimizer
from typing import Dict, List, Union
from tqdm import tqdm
import sys
import pandas as pd


class RelaxationData:
    index = []
    e_relax = []
    num_sites = []
    structure = []
    exception = []
        
class DummyBatchCalculator(Calculator):
    def __init__(self):
        super().__init__()

    def calculate(self, atoms=None, properties=None, system_changes=None):
        pass

    def get_potential_energy(self, atoms=None):
        return atoms.info["total_energy"]

    def get_forces(self, atoms=None):
        return atoms.arrays["forces"]

    def get_stress(self, atoms=None):
        return units.GPa * atoms.info["stress"]

class BatchRelaxer(object):
    """BatchRelaxer is a class for batch structural relaxation.
    It is more efficient than Relaxer when relaxing a large number of structures."""

    SUPPORTED_OPTIMIZERS = {"BFGS": BFGS, "FIRE": FIRE}
    SUPPORTED_FILTERS = {
        "EXPCELLFILTER": ExpCellFilter,
        "FRECHETCELLFILTER": FrechetCellFilter,
    }

    def __init__(
        self,
        potential: Potential,
        mlff: str,
        optimizer: Union[str, type[Optimizer]] = "FIRE",
        filter: Union[type[Filter], str, None] = None,
        fmax: float = 0.05,
        max_natoms_per_batch: int = 512,
        step: int = 500,
        device = None,
        logger = None
    ):
        self.potential = potential
        self.device = device
        self.optimizer = self.SUPPORTED_OPTIMIZERS[optimizer]
        self.filter = self.SUPPORTED_FILTERS[filter]
        
        self.fmax = fmax
        self.max_natoms_per_batch = max_natoms_per_batch
        self.optimizer_instances: List[Optimizer] = []
        self.is_active_instance: List[bool] = []
        self.finished = False
        self.total_converged = 0
        self.trajectories: Dict[int, List[Atoms]] = {}
        
        self.optimize_step = []
        self.mlff = mlff
        self.step = step
        self.logger = logger

    def insert(self, atoms: Atoms):
        atoms.set_calculator(DummyBatchCalculator())
        optimizer_instance = self.optimizer(
            self.filter(atoms) if self.filter else atoms
        )
        optimizer_instance.fmax = self.fmax
        self.optimizer_instances.append(optimizer_instance)
        self.is_active_instance.append(True)

    def mattersim_pred(self, atoms_list):
        dataloader = build_dataloader(
            atoms_list, batch_size=len(atoms_list), only_inference=True
        )
        energy_batch, forces_batch, stress_batch = self.potential.predict_properties(
            dataloader, include_forces=True, include_stresses=True
        )
        return energy_batch, forces_batch, stress_batch
    
    def step_batch(self):
        atoms_list = []
        for idx, opt in enumerate(self.optimizer_instances):
            if self.is_active_instance[idx]:
                atoms_list.append(opt.atoms)

        # Note: we use a batch size of len(atoms_list)
        # because we only want to run one batch at a time
        if self.mlff == "mattersim":
            energy_batch, forces_batch, stress_batch = self.mattersim_pred(atoms_list)

        counter = 0
        self.finished = True
        for idx, opt in enumerate(self.optimizer_instances):
            if self.is_active_instance[idx]:
                # Set the properties so the dummy calculator can
                # return them within the optimizer step
                opt.atoms.info["total_energy"] = energy_batch[counter]
                opt.atoms.arrays["forces"] = forces_batch[counter]
                opt.atoms.info["stress"] = stress_batch[counter]
                try:
                    self.trajectories[opt.atoms.info["structure_index"]].append(
                        opt.atoms.copy()
                    )
                except KeyError:
                    self.trajectories[opt.atoms.info["structure_index"]] = [
                        opt.atoms.copy()
                    ]

                opt.step()
                if opt.converged() or len(self.trajectories[opt.atoms.info["structure_index"]])>self.step:
                    self.optimize_step.append(len(self.trajectories[opt.atoms.info["structure_index"]]))
                    self.is_active_instance[idx] = False
                    self.total_converged += 1
                    if self.total_converged % 100 == 0:
                        self.logger.info(f"Relaxed {self.total_converged} structures. Avg optimization step {sum(self.optimize_step) / len(self.optimize_step)}")
                else:
                    self.finished = False
                counter += 1

        # remove inactive instances
        self.optimizer_instances = [
            opt
            for opt, active in zip(self.optimizer_instances, self.is_active_instance)
            if active
        ]
        self.is_active_instance = [True] * len(self.optimizer_instances)

    def relax(
        self,
        atoms_list: List[Atoms],
    ) -> Dict[int, List[Atoms]]:
        self.trajectories = {}
        self.tqdmcounter = tqdm(total=len(atoms_list), file=sys.stdout)
        pointer = 0
        atoms_list_ = []
        for i in range(len(atoms_list)):
            atoms_list_.append(atoms_list[i].copy())
            atoms_list_[i].info["structure_index"] = i


        while (
            pointer < len(atoms_list) or not self.finished
        ):  # While there are unfinished instances or atoms left to insert
            while pointer < len(atoms_list) and (
                sum([len(opt.atoms) for opt in self.optimizer_instances])
                + len(atoms_list[pointer])
                <= self.max_natoms_per_batch
            ):
                # While there are enough n_atoms slots in the
                # batch and we have not reached the end of the list.
                self.insert(
                    atoms_list_[pointer]
                )  # Insert new structure to fire instances
                self.tqdmcounter.update(1)
                pointer += 1
            self.step_batch()
        self.tqdmcounter.close()

        return self.trajectories

def relax_atoms(
    atoms: list[Atoms], device: str, mlff: str, step: int, logger, **kwargs
) -> tuple[list[Atoms], np.ndarray]:
    if mlff == "mattersim":
        potential = Potential.from_checkpoint(
            device=device, load_path="MatterSim-v1.0.0-1M.pth", load_training_state=False
        )
        batch_relaxer = BatchRelaxer(potential=potential, filter="EXPCELLFILTER", mlff="mattersim", step=step, device=device, logger=logger, **kwargs)
        relaxation_trajectories = batch_relaxer.relax(atoms)
        relaxed_atoms = [t[-1] for t in relaxation_trajectories.values()]
        total_energies = np.array([a.info["total_energy"] for a in relaxed_atoms])
        return relaxed_atoms, total_energies

def relax_structures(
    structures: Structure | list[Structure],
    device: str,
    steps: float,
    mlff: str,
    logger,
    **kwargs
) -> tuple[list[Structure], np.ndarray]:
    if isinstance(structures, Structure):
        structures = [structures]
    atoms = [AseAtomsAdaptor.get_atoms(structures[i]) for i in range(len(structures))]
    if mlff == 'mattersim':
        relaxed_atoms, total_energies = relax_atoms(atoms, device=device, mlff=mlff, step=steps, logger=logger, **kwargs)
        relaxed_structures = [AseAtomsAdaptor.get_structure(a) for a in relaxed_atoms]
    
    rd = RelaxationData()
    for i in tqdm(range(len(structures))):
        try:
            structure = structures[i]
            rd.index.append(i)
            rd.e_relax.append(total_energies[i])
            rd.exception.append(False)
            rd.num_sites.append(structure.num_sites)
            rd.structure.append(relaxed_structures[i].as_dict())
        except Exception as exp:
            print(exp)
            rd.index.append(i)
            rd.e_relax.append(None)
            rd.exception.append(True)
            rd.num_sites.append(pd.NA)
            rd.structure.append({})
    pre_pandas = {
        "index": rd.index,
        "e_relax" : rd.e_relax,
        "num_sites" : rd.num_sites,
        "structure" : rd.structure,
        "exception" : rd.exception
    }
    df = pd.DataFrame.from_dict(pre_pandas).set_index("index")
    return df
