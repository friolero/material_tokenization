# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import cached_property

from eval.vsun.reference.reference_dataset import ReferenceDataset
from eval.vsun.reference.reference_dataset_serializer import LMDBGZSerializer


class ReferenceMP2020Correction(ReferenceDataset):
    """Reference dataset using the MP2020 Energy Correction scheme.
    This dataset contains entries from the Materials Project [https://next-gen.materialsproject.org/]
    and Alexandria [https://next-gen.materialsproject.org/].
    All 845,997 structures are relaxed using the GGA-PBE functional and have energy corrections applied using the MP2020 scheme.
    """

    def __init__(self):
        super().__init__("MP2020correction", ReferenceMP2020Correction.from_preset())

    @classmethod
    def from_preset(cls) -> "ReferenceMP2020Correction":
        return LMDBGZSerializer().deserialize(
            f"eval/vsun/ref_dataset/alex-mp/reference_MP2020correction.gz"
        )

    @cached_property
    def is_ordered(self) -> bool:
        """Returns True if all structures are ordered."""
        return True # Setting it manually to avoid computation at runtime.

class MP2023(ReferenceDataset):
    """Reference dataset using the MP2020 Energy Correction scheme.
    This dataset contains entries from the Materials Project [https://next-gen.materialsproject.org/]
    and Alexandria [https://next-gen.materialsproject.org/].
    All 845,997 structures are relaxed using the GGA-PBE functional and have energy corrections applied using the MP2020 scheme.
    """

    def __init__(self):
        super().__init__("mp_02072023", MP2023.from_preset())

    @classmethod
    def from_preset(cls) -> "MP2023":
        return LMDBGZSerializer().deserialize(f"eval/vsun/ref_dataset/mp/mp.gz")

    @cached_property
    def is_ordered(self) -> bool:
        """Returns True if all structures are ordered."""
        return True # Setting it manually to avoid computation at runtime.