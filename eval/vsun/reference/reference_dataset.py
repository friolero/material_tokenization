# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import cached_property
from typing import Iterable, Iterator, Mapping
from pymatgen.entries.computed_entries import ComputedStructureEntry

from eval.vsun.reference.utils import generate_chemsys_dict, generate_reduced_formula_dict


class ReferenceDataset(Iterable[ComputedStructureEntry]):
    """Immutable collection of reference entries with the ability to cache
    some computation (e.g., space groups).
    """

    def __init__(
        self,
        name: str,
        impl: "ReferenceDatasetImpl",
    ):
        self.name = name
        # The Bridge pattern. The actual implementation is defined in ReferenceDatasetImpl.
        self.impl = impl

    @staticmethod
    def from_entries(name: str, entries: Iterable[ComputedStructureEntry]) -> "ReferenceDataset":
        return ReferenceDataset(name, ReferenceDatasetImpl(entries))

    def __iter__(self) -> Iterator[ComputedStructureEntry]:
        yield from self.impl

    def __len__(self) -> int:
        return len(self.impl)

    @property
    def entries_by_reduced_formula(self) -> Mapping[str, list[ComputedStructureEntry]]:
        return self.impl.entries_by_reduced_formula

    @property
    def entries_by_chemsys(self) -> Mapping[str, list[ComputedStructureEntry]]:
        return self.impl.entries_by_chemsys

    @cached_property
    def is_ordered(self) -> bool:
        """Returns True if all structures are ordered."""
        return all(e.structure.is_ordered for e in self)


class ReferenceDatasetImpl(Iterable[ComputedStructureEntry]):
    """The implementation of ReferenceDataset. Direct access to entries is not allowed."""

    def __init__(self, entries: Iterable[ComputedStructureEntry]):
        self._entries = tuple(entries)

    def __iter__(self) -> Iterator[ComputedStructureEntry]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    @cached_property
    def entries_by_reduced_formula(self) -> Mapping[str, list[ComputedStructureEntry]]:
        """This is a slow path. Subclasses may override entries_by_reduced_formula method
        to avoid calling this method."""
        return generate_reduced_formula_dict(self._entries)

    @cached_property
    def entries_by_chemsys(self) -> Mapping[str, list[ComputedStructureEntry]]:
        """This is a slow path. Subclasses may override entries_by_chemsys method
        to avoid calling this method."""
        # 这是一个dict，key是chemical system， 'Ca-Cd-Ce'这种，value是个list，有同种system下的所有entry，i.e., [Ca3 Ce2 Cd2, ...]
        return generate_chemsys_dict(self._entries)
