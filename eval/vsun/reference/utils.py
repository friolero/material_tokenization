# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import defaultdict
from itertools import combinations
from typing import Any, Callable, Iterable, TypeVar

from pymatgen.entries.computed_entries import ComputedStructureEntry


OptionalNumber = int | float | None
PropertyConstraint = tuple[
    OptionalNumber, OptionalNumber
]  # These encode the minimum and maximum values for a property

def generate_reduced_formula_dict(
    entries: Iterable[ComputedStructureEntry],
) -> dict[str, list[ComputedStructureEntry]]:
    """Generate a dictionary of entries with the same reduced formula."""

    def keyfunc(entry: ComputedStructureEntry) -> str:
        entry.structure.unset_charge()
        return entry.structure.remove_oxidation_states().composition.reduced_formula

    return group_list_items_into_dict(entries, keyfunc=keyfunc)

def generate_chemsys_dict(
    entries: Iterable[ComputedStructureEntry],
) -> dict[str, list[ComputedStructureEntry]]:
    ### 这里最终返回的是dict，key是chemical systems, e.g., A-B-C, values是所有属于这个system下的entry, e.g., ABC2, AB2C2....
    """Generate a dictionary of entries with the same chemical system."""

    def keyfunc(entry: ComputedStructureEntry) -> str:
        return "-".join(sorted({el.symbol for el in entry.composition.elements}))

    return group_list_items_into_dict(entries, keyfunc=keyfunc)

T = TypeVar("T")

def group_list_items_into_dict(
    items: Iterable[T], keyfunc: Callable[[Any], str]
) -> dict[str, list[T]]:
    """Group a list of items into a dictionary with the same key."""
    result = defaultdict(list)
    # To reduce the number of calls to keyfunc, we use a defaultdict instead of itertools.groupby,
    # which requires the list to be sorted.
    for item in items:
        result[keyfunc(item)].append(item)
    return result

def expand_into_subsystems(chemical_system: str) -> list[tuple[str, ...]]:
    elements = chemical_system.split("-")
    list_combinations = []
    for n in range(1, len(elements) + 1):
        list_combinations += list(combinations(elements, n))  ## C_{elements}^n
    return list_combinations

