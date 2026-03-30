"""
Dataset package for AML Detection
"""

from .elliptic_dataset import (
    EllipticInMemoryDataset,
    load_elliptic_dataset,
    get_data_info
)

__all__ = [
    'EllipticInMemoryDataset',
    'load_elliptic_dataset',
    'get_data_info'
]
