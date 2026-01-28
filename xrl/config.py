"""
Simulation Configuration for X-ray Lithography
================================================

Provides a single ``SimulationConfig`` dataclass that captures every
tuneable parameter of the XRL simulation pipeline.  Configs can be
created in code, loaded from JSON / YAML files, or serialised back
for reproducibility and future GUI integration.

Author: Abhineet Agarwal
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class SimulationConfig:
    """Complete parameter set for an XRL simulation run.

    Attributes:
        energy_kev: Photon energy in keV.
        gap_um: Mask-to-resist gap in um.
        absorber_material: Key into ``MATERIALS`` dict (e.g. 'Ta').
        absorber_thickness_um: Absorber layer thickness in um.
        membrane_material: Key into ``MATERIALS`` dict (e.g. 'Si3N4').
        membrane_thickness_um: Membrane thickness in um.
        feature_size_um: Minimum feature (line) width in um.
        pitch_um: Pattern pitch in um.
        resist: Key into ``RESISTS`` dict (e.g. 'PMMA').
        dose_factor: Multiplicative factor on clearing dose.
        include_noise: Whether to add photon shot noise.
        n_samples_ler: Number of Monte-Carlo samples for LER estimation.
        beam_power_W: Incident X-ray beam power in W (thermal analysis).
        membrane_size_mm: Membrane side length / diameter in mm.
        membrane_geometry: 'square' or 'circular'.
        resolution: Number of spatial grid points.
        x_range_um: Spatial extent of simulation window in um.
    """
    # Aerial image parameters
    energy_kev: float = 1.5
    gap_um: float = 10.0
    absorber_material: str = 'Ta'
    absorber_thickness_um: float = 0.5
    membrane_material: str = 'Si3N4'
    membrane_thickness_um: float = 2.0
    feature_size_um: float = 0.5
    pitch_um: float = 1.0

    # Resist parameters
    resist: str = 'PMMA'
    dose_factor: float = 1.0
    include_noise: bool = True
    n_samples_ler: int = 50

    # Thermal parameters
    beam_power_W: float = 0.1
    membrane_size_mm: float = 50.0
    membrane_geometry: str = 'square'

    # Grid parameters
    resolution: int = 1000
    x_range_um: float = 3.0

    def to_dict(self) -> dict:
        """Serialise config to a plain dictionary."""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Write config to a JSON file."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> 'SimulationConfig':
        """Load config from a JSON or YAML file.

        YAML support requires ``pyyaml`` to be installed.
        """
        path = Path(path)
        if path.suffix in ('.yaml', '.yml'):
            try:
                import yaml
            except ImportError as exc:
                raise ImportError(
                    "Install pyyaml to load YAML configs: pip install pyyaml"
                ) from exc
            with open(path) as f:
                data = yaml.safe_load(f)
        else:
            with open(path) as f:
                data = json.load(f)
        return cls(**data)


def default_config() -> SimulationConfig:
    """Return a ``SimulationConfig`` with sensible defaults.

    Defaults represent a typical Ta-absorber / Si3N4-membrane mask
    exposing PMMA at 1.5 keV with a 10 um gap.
    """
    return SimulationConfig()
