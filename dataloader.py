import os

import numpy as np
import pandas as pd
import torch
from p_tqdm import p_umap
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset

mindis = MinimumDistanceNN()


def type_coords(frac_coords, atomic_numbers):
    atomic_tensor = atomic_numbers.reshape(-1, 1)
    all_tensor = torch.cat([atomic_tensor, frac_coords], -1)
    coe = (
        1000 * all_tensor[:, 0]
        + 100 * all_tensor[:, 1]
        + 10 * all_tensor[:, 2]
        + 1 * all_tensor[:, 3]
    )
    index_order = torch.argsort(coe)
    return index_order


arrange_buffer = {
    "type_coords": type_coords,
}


class CrystalDataset(Dataset):
    def __init__(self, dataname, config, val=False, test=False):
        if test:
            path = os.path.join("data", dataname, "test.csv")
            self.path_prepare = os.path.join(
                "data",
                dataname,
                "prepared_test_data_" + str(config.model.neigh_num) + ".pt",
            )
        elif val:
            path = os.path.join("data", dataname, "val.csv")
            self.path_prepare = os.path.join(
                "data",
                dataname,
                "prepared_val_data_" + str(config.model.neigh_num) + ".pt",
            )
        else:
            path = os.path.join("data", dataname, "train.csv")
            self.path_prepare = os.path.join(
                "data",
                dataname,
                "prepared_train_data_" + str(config.model.neigh_num) + ".pt",
            )

        self.config = config
        self.max_atoms = config.dataset.max_atoms
        self.max_types = config.dataset.max_types
        self.max_neigh_num = config.model.neigh_num
        self.arrange = config.model.arrange
        self.num_workers = config.dataset.num_workers
        self.primitive = config.dataset.primitive
        self.read_from_cif(path)
        self.matrix_scaler = None
        self.offset = torch.cat(
            [torch.tensor([0]), torch.cumsum(self.data["num_atoms"], -1)], 0
        ).long()

    def __len__(self) -> int:
        return len(self.data["material_id"])

    def __getitem__(self, index):
        start, end = self.offset[index], self.offset[index + 1]
        matrix, num_atoms = (
            self.data["matrix"][index],
            self.data["num_atoms"][index],
        )
        cart_coords = self.data["cart_coords"][start:end]
        atomic_numbers = self.data["atomic_numbers"][start:end]
        neigh_coords = self.data["neigh_coords"][start:end]
        neigh_types = self.data["neigh_types"][start:end]

        scaled_cart_coords = cart_coords / num_atoms ** (1 / 3)
        scaled_neigh_coords = neigh_coords / num_atoms ** (1 / 3)

        scalar_cart_coords = self.matrix_scaler.transform(scaled_cart_coords)
        scalar_neigh_coords = self.matrix_scaler.transform(scaled_neigh_coords)
        scalar_matrix = self.matrix_scaler.transform(matrix)

        material_id = self.data["material_id"][index]
        return (
            material_id,
            scalar_matrix,
            scalar_cart_coords,
            atomic_numbers,
            num_atoms,
            scalar_neigh_coords,
            neigh_types,
        )

    def add_scaled_matrix(self, data, scale_len=True):
        matrix = data["matrix"]
        num_atoms = data["num_atoms"].reshape(-1, 1)
        if scale_len:
            matrix = matrix / num_atoms.float() ** (1 / 3)
        data["matrix"] = matrix

    def read_from_cif(self, path):
        if not os.path.exists(self.path_prepare):
            df = pd.read_csv(path)

            unordered_results = np.array(
                p_umap(
                    self.cif_info,
                    [df.iloc[idx] for idx in range(len(df))],
                    num_cpus=self.num_workers,
                )
            )
            order = np.array(
                [result["material_id"] for result in unordered_results]
            ).argsort()
            order_results = unordered_results[order]
            data = self.unpack(order_results)
            self.add_scaled_matrix(data)
            self.data = {
                "material_id": data["material_id"],
                "cart_coords": data["cart_coords"],
                "atomic_numbers": data["atomic_numbers"],
                "matrix": data["matrix"],
                "num_atoms": data["num_atoms"],
                "neigh_coords": data["neigh_coords"],
                "neigh_types": data["neigh_types"],
            }
            torch.save(self.data, self.path_prepare)
        else:
            self.data = torch.load(self.path_prepare, weights_only=False)

    def unpack(self, results):
        (
            material_id,
            cart_coords,
            atomic_numbers,
            matrix,
            num_atoms,
            neigh_coords,
            neigh_types,
        ) = ([], [], [], [], [], [], [])
        for re in results:
            material_id.append(re["material_id"])
            num_atoms.append(torch.LongTensor([re["num_atoms"]]))
            matrix.append(re["matrix"])
            cart_coords.append(re["cart_coords"])
            atomic_numbers.append(re["atomic_numbers"].long())

            neigh_coords.append(re["neigh_coords"])
            neigh_types.append(re["neigh_types"])

        (
            matrix,
            material_id,
            num_atoms,
            cart_coords,
            atomic_numbers,
            neigh_coords,
            neigh_types,
        ) = (
            torch.stack(matrix, 0),
            np.array(material_id),
            torch.cat(num_atoms, 0),
            torch.cat(cart_coords, 0),
            torch.cat(atomic_numbers, 0),
            torch.cat(neigh_coords, 0),
            torch.cat(neigh_types, 0),
        )

        return {
            "material_id": material_id,
            "cart_coords": cart_coords,
            "atomic_numbers": atomic_numbers,
            "matrix": matrix,
            "num_atoms": num_atoms,
            "neigh_coords": neigh_coords,
            "neigh_types": neigh_types,
        }

    def cif_info(self, row):
        cif, material_id = row["cif"], row["material_id"]
        structure = Structure.from_str(cif, fmt="cif")
        if self.primitive:
            structure = structure.get_primitive_structure()
        structure = structure.get_reduced_structure()  # niggli
        canonical_structure = Structure(
            lattice=Lattice.from_parameters(*structure.lattice.parameters),
            species=structure.species,
            coords=structure.frac_coords,
            coords_are_cartesian=False,
        )
        cart_coords = torch.FloatTensor(canonical_structure.cart_coords)
        atomic_numbers = torch.FloatTensor(canonical_structure.atomic_numbers)
        # tensor([ 6, 10,  7, 11,  8,  9,  1,  2,  0,  3,  4,  5])
        index_order = arrange_buffer[self.arrange](cart_coords, atomic_numbers)

        neigh_coords = []
        neigh_types = []
        for i in range(len(canonical_structure)):
            nn_data = mindis.get_nn_info(canonical_structure, i)
            weights = np.array([nn["weight"] for nn in nn_data])
            new_order = np.argsort(-weights)
            neigh_sites = [nn_data[order]["site"] for order in new_order]

            if len(neigh_sites) > self.max_neigh_num:
                neigh_sites = neigh_sites[: self.max_neigh_num]
            one_coord = -torch.ones(self.max_neigh_num, 3)
            one_type = (torch.ones(self.max_neigh_num) * 99).long()
            for n, site in enumerate(neigh_sites):
                one_coord[n] = torch.FloatTensor(site.coords).float()
                one_type[n] = torch.LongTensor([site._species.elements[0].Z])
            neigh_coords.append(one_coord), neigh_types.append(one_type)

        neigh_coords = torch.stack(neigh_coords, 0)
        neigh_types = torch.stack(neigh_types, 0)

        # reorder
        cart_coords = cart_coords[index_order]
        atomic_numbers = atomic_numbers[index_order]
        neigh_coords = neigh_coords[index_order]
        neigh_types = neigh_types[index_order]

        num_atom = len(atomic_numbers)
        matrix = torch.tensor(canonical_structure.lattice.matrix)
        matrix = self.compute_lattice_polar_decomposition(matrix)
        return {
            "material_id": material_id,
            "cart_coords": cart_coords,
            "atomic_numbers": atomic_numbers,
            "matrix": matrix,
            "num_atoms": num_atom,
            "neigh_coords": neigh_coords,
            "neigh_types": neigh_types,
        }

    def compute_lattice_polar_decomposition(
        self, lattice_matrix: torch.Tensor
    ) -> torch.Tensor:
        W, S, V_transp = torch.linalg.svd(lattice_matrix)
        S_square = torch.diag_embed(S)
        V = V_transp.transpose(0, 1)
        U = W @ V_transp
        P = V @ S_square @ V_transp
        P_prime = U @ P @ U.transpose(0, 1)
        # symmetrized lattice matrix
        symm_lattice_matrix = P_prime
        symm_lattice_matrix[torch.abs(symm_lattice_matrix) < 1e-5] = 0.0

        tri_indices = torch.triu_indices(3, 3)
        symm_lattice_matrix = symm_lattice_matrix[
            tri_indices[0], tri_indices[1]
        ]
        return symm_lattice_matrix


def collate(batch):
    """Collate crystals into flat per-atom tensors; neighbour lists are prefixed with the atom itself."""
    (
        material_id,
        matrix,
        cart_coords,
        atomic_numbers,
        num_atoms,
        neigh_coords,
        neigh_types,
    ) = zip(*batch)

    matrix = torch.stack(matrix, 0)
    cart_coords = torch.cat(cart_coords, 0)
    atomic_numbers = torch.cat(atomic_numbers, 0)
    num_atoms = torch.LongTensor(list(num_atoms))

    neigh_coords = torch.cat(neigh_coords, 0)
    neigh_coords = torch.cat([cart_coords.unsqueeze(1), neigh_coords], 1)
    neigh_types = torch.cat(neigh_types, 0)
    neigh_types = torch.cat([atomic_numbers.unsqueeze(1), neigh_types], 1)
    return (
        matrix.float(),
        cart_coords.float(),
        atomic_numbers.long(),
        num_atoms,
        neigh_coords.float(),
        neigh_types.long(),
    )
