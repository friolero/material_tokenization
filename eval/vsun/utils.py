from pymatgen.core import Structure, Composition, Species, Element, DummySpecies
from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.entries.computed_entries import ComputedStructureEntry
from typing import Iterable, List, Mapping
import pandas as pd
import numpy as np
from collections import defaultdict

def to_structure(structure: Structure | dict | str) -> Structure:
    if isinstance(structure, dict):
        return Structure.from_dict(structure)
    elif isinstance(structure, str):
        return Structure.from_str(structure, fmt="cif")
    else:
        return structure

def maybe_get_missing_columns(df, maps) -> pd.DataFrame:
    for name, mapping in maps.items():
        if name not in df.columns:
            df[name] = mapping(df)
    return df

COLUMNS_COMPUTATIONS = {
    "composition": lambda df: df["structure"].map(get_composition_dict),
    "chemsys": lambda df: df["composition"].map(get_chemsys),
    "nary": lambda df: df["chemsys"].map(get_nary),
}

def get_composition(
    structure: Structure | dict,
) -> Composition:
    if structure == {}:
        return Composition()
    return to_structure(structure).composition

def get_composition_dict(
    structure: Structure | dict,
) -> dict[str, float]:
    return get_composition(structure).as_dict()

def get_chemsys(
    composition: Composition | dict,
) -> set[Element | Species | DummySpecies] | None:
    elements = list(elt.name for elt in to_composition(composition).elements)
    return set(elements)

  
def get_nary(
    chemsys: set[Element, Species, DummySpecies] | list[Element, Species, DummySpecies]
) -> int:
    return len(set(chemsys))


def to_composition(composition: Composition | dict) -> Composition:
    if isinstance(composition, dict):
        return Composition.from_dict(composition)
    else:
        return composition


def filter_prerelaxed(
    df: pd.DataFrame,
    num_structures: int | None = None,
    filter_exceptions: bool = True,
    # maxima: dict[str, float | int] = {"e_delta": 15.0},
    maxima: dict[str, float | int] = {},
    maximum_nary: int | None = None,
    minimum_nary: int = 0, # 1
) -> pd.DataFrame:
    if filter_exceptions:
        df = df[df["exception"] == False]

    for key, value in maxima.items():
        df = df[df[key] < value]

    if maximum_nary is not None:
        df = df[df["chemsys"].map(len) <= maximum_nary]
    df = df[df["chemsys"].map(len) > minimum_nary]

    if num_structures is not None:
        print(f"limiting to the first {num_structures} samples after filtering")
        df = df.iloc[:num_structures, :]
    return df


def get_unique(structure_matcher: StructureMatcher, structures: List[Structure]) -> List[int]:
    if len(structures) == 1:
        return [0]
    unique_structures: list[Structure] = []
    unique_idx: list[int] = []
    for idx, structure in enumerate(structures):
        unique = True
        for structure_2 in unique_structures:
            if structure_matcher.fit(structure, structure_2):
                unique = False
                break
        if unique:
            unique_structures.append(structure)
            unique_idx.append(idx)
    return unique_idx

def get_mask_from_local_index(
    entries_mapping_by_key: Mapping[str, list[ComputedStructureEntry]],
    local_index: Mapping[str, List[int]],
) -> np.typing.NDArray[np.bool_]:
    """Turn local structure chemsys index into global structure mask."""
    global_indices = get_global_index_from_local_index(entries_mapping_by_key, local_index)
    total_num_entries = sum(len(v) for v in entries_mapping_by_key.values())
    mask = np.zeros(total_num_entries, dtype=bool)
    mask[global_indices] = True
    return mask

def get_global_index_from_local_index(
    entries_mapping_by_key: Mapping[str, list[ComputedStructureEntry]],
    local_index: Mapping[str, list[int]],
) -> list[int]:
    """Turn local structure chemsys index into global structure mask."""
    global_indices = [
        entries_mapping_by_key[k][vv].entry_id for k, v in local_index.items() for vv in v
    ]
    return global_indices

def get_global_match_dict_from_local_dict(
    data_entries_mapping_by_key: Mapping[str, list[ComputedStructureEntry]],
    reference_entries_mapping_by_key: Mapping[str, list[ComputedStructureEntry]],
    local_index: Mapping[str, dict[int, list[int]]],
) -> dict[int, list[str]]:
    global_match_dict = {}
    for k, match_dict in local_index.items():
        if len(match_dict) == 0 or max(len(v) for v in match_dict.values()) == 0:
            continue
        # Get the mapping of the data and reference entries only once, as it requires disk access
        data_entries_mapping = data_entries_mapping_by_key[k]
        reference_entries_mapping = reference_entries_mapping_by_key[k]
        for d1_ix, ref_ix_list in match_dict.items():
            global_match_dict[data_entries_mapping[d1_ix].entry_id] = [
                reference_entries_mapping[match_ix].data["material_id"] for match_ix in ref_ix_list
            ]
    return global_match_dict

def get_matches(
    structure_matcher: StructureMatcher, d1: List[Structure], d2: List[Structure],
) -> dict[int, list[int]]:
    """
    Iterates d1 to find matches in d2.

    Args:
        structure_matcher: StructureMatcher to use for comparison.
        d1: List of structures to compare.
        d2: List of structures to compare against.

    Returns:
        matches: Dictionary of matches. Key is the index of the structure in d1 and value is the index of the structure in d2.

    """
    matches: dict[int, list[int]] = defaultdict(list)

    for i in range(len(d1)):
        for j in range(len(d2)):
            if structure_matcher.fit(d1[i], d2[j]):
                matches[i].append(j)
    return matches

def matches_to_mask(match_idx: Iterable[int], num_samples: int) -> np.typing.NDArray[bool]:
    """
    Convert matches to a boolean mask.

    Args:
        match_idx: List of indices of the structures from the input dataset which have a match
            in the reference dataset.
        num_samples: Number of structures in the input dataset.

    Returns:
        mask: Boolean mask of length num_samples. True if the structure has a match, False if not.
    """
    mask = np.zeros(num_samples, dtype=bool)
    mask[list(match_idx)] = True
    return mask
