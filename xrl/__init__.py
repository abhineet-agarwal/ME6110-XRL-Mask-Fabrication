"""
XRL Simulator -- X-ray Lithography Process Modeling
====================================================

A tutorial-style Python package for simulating X-ray lithography
mask transmission, aerial image formation, resist exposure, and
thermal-mechanical behaviour.

Quick start::

    from xrl import XRayMask, AerialImageSimulator

    mask = XRayMask(absorber_material='Ta', feature_size=0.5)
    sim = AerialImageSimulator(mask, gap=10.0)
    x, intensity = sim.compute_aerial_image(energy_kev=1.5)
"""

__version__ = "0.1.0"

from .materials import (
    MaterialProperties,
    ResistProperties,
    MembraneMechanicalProperties,
    MATERIALS,
    RESISTS,
    MEMBRANES,
)
from .aerial_image import XRayMask, AerialImageSimulator
from .resist import ResistExposureModel
from .thermal import MembraneMechanics, ThermalAnalysis
from .config import SimulationConfig, default_config

__all__ = [
    # Materials
    "MaterialProperties",
    "ResistProperties",
    "MembraneMechanicalProperties",
    "MATERIALS",
    "RESISTS",
    "MEMBRANES",
    # Aerial image
    "XRayMask",
    "AerialImageSimulator",
    # Resist
    "ResistExposureModel",
    # Thermal
    "MembraneMechanics",
    "ThermalAnalysis",
    # Config
    "SimulationConfig",
    "default_config",
]
