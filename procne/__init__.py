"""
ProCNE — Probabilistic Channel Network Evolution

Three simulation modes:
    procne.square_grid        — square planar grid, all borders are outlets
    procne.general_boundary   — arbitrary domain from a PNG mask
    procne.single_basin       — single basin with one specified outlet
"""

from . import square_grid, general_boundary, single_basin

__version__ = "1.0.0"
__all__ = ["square_grid", "general_boundary", "single_basin"]
