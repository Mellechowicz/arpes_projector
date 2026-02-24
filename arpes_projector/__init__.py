"""
ARPES Spectroscopic Simulation and Visualization Suite.

Provides methods to extract VASP band structures, project 3D coordinate spaces 
onto 2D planes, and render simulated photoemission intensities.
"""

__version__ = "1.0.0"

from arpes_projector.parser import VaspDataParser
from arpes_projector.geometry import KSpaceProjector
from arpes_projector.plotter import ARPESPlotter
from arpes_projector.surface_bz import SurfaceBZAnalyzer

__all__ = [
    "VaspDataParser",
    "KSpaceProjector",
    "ARPESPlotter",
    "SurfaceBZAnalyzer"
]

