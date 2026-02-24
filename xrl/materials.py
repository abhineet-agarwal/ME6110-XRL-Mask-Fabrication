"""
Material Database for X-ray Lithography Simulation
===================================================

Consolidated material properties for absorbers, membranes, and resists
used in X-ray lithography mask fabrication and process simulation.

All properties sourced from standard references and NIST XCOM database.
Units are documented per field.

Author: Abhineet Agarwal
"""

from dataclasses import dataclass


@dataclass
class MaterialProperties:
    """X-ray absorber or membrane material properties.

    Attributes:
        name: Full material name (e.g. 'Tantalum').
        density: Mass density in g/cm^3.
        atomic_number: Effective atomic number (averaged for compounds).
        atomic_mass: Effective atomic mass in g/mol.
    """
    name: str
    density: float      # g/cm^3
    atomic_number: float
    atomic_mass: float   # g/mol

    def get_attenuation_coefficient(self, energy_kev: float) -> float:
        """Linear attenuation coefficient μ in 1/cm.

        Values from NIST XCOM tabulated data (log-log interpolation).

        Args:
            energy_kev: Photon energy in keV.

        Returns:
            Linear attenuation coefficient mu in 1/cm.
        """
        from .data.nist_xcom import get_mu_rho
        mu_rho = get_mu_rho(self.name, energy_kev)
        return mu_rho * self.density


@dataclass
class ResistProperties:
    """Photoresist material properties for X-ray exposure.

    Attributes:
        name: Resist trade name (e.g. 'PMMA').
        density: Mass density in g/cm^3.
        sensitivity: Clearing dose D0 in mJ/cm^2.
        contrast: Photoresist contrast gamma (dimensionless).
        blur: Acid diffusion / electron range blur in um.
        thickness: Film thickness in um.
        tone: 'positive' or 'negative'.
    """
    name: str
    density: float       # g/cm^3
    sensitivity: float   # mJ/cm^2 (D0)
    contrast: float      # gamma (dimensionless)
    blur: float          # um (acid diffusion length)
    thickness: float     # um
    tone: str            # 'positive' or 'negative'


@dataclass
class MembraneMechanicalProperties:
    """Mechanical and thermal properties of mask membrane material.

    Attributes:
        name: Full material name.
        youngs_modulus: Young's modulus in GPa.
        poisson_ratio: Poisson's ratio (dimensionless).
        density: Mass density in g/cm^3.
        thermal_expansion: Coefficient of thermal expansion in 1/K (x 1e-6).
        thermal_conductivity: Thermal conductivity in W/(m*K).
        specific_heat: Specific heat capacity in J/(kg*K).
    """
    name: str
    youngs_modulus: float       # GPa
    poisson_ratio: float        # dimensionless
    density: float              # g/cm^3
    thermal_expansion: float    # 1/K (x 1e-6, i.e. value of 2.3 means 2.3e-6 /K)
    thermal_conductivity: float # W/(m*K)
    specific_heat: float        # J/(kg*K)


# ---------------------------------------------------------------------------
# Material databases
# ---------------------------------------------------------------------------

MATERIALS: dict[str, MaterialProperties] = {
    'Ta': MaterialProperties('Tantalum', 16.6, 73, 180.9),
    'W': MaterialProperties('Tungsten', 19.3, 74, 183.8),
    'Au': MaterialProperties('Gold', 19.3, 79, 197.0),
    'Si3N4': MaterialProperties('Silicon Nitride', 3.44, 11.2, 140.3),
    'SiC': MaterialProperties('Silicon Carbide', 3.21, 10, 40.1),
    'PMMA': MaterialProperties('PMMA', 1.18, 3.6, 100.1),
}

RESISTS: dict[str, ResistProperties] = {
    'PMMA': ResistProperties('PMMA', 1.18, 500, 7.0, 0.05, 1.0, 'positive'),
    'ZEP520A': ResistProperties('ZEP520A', 1.11, 80, 9.0, 0.03, 0.5, 'positive'),
    'SU8': ResistProperties('SU-8', 1.19, 150, 4.0, 0.08, 10.0, 'negative'),
    'HSQ': ResistProperties('HSQ', 1.4, 800, 1.5, 0.02, 0.3, 'negative'),
}

MEMBRANES: dict[str, MembraneMechanicalProperties] = {
    'Si3N4': MembraneMechanicalProperties(
        'Silicon Nitride', 250, 0.27, 3.44, 2.3, 20, 700,
    ),
    'SiC': MembraneMechanicalProperties(
        'Silicon Carbide', 450, 0.19, 3.21, 3.7, 120, 750,
    ),
    'Diamond': MembraneMechanicalProperties(
        'Diamond', 1050, 0.20, 3.52, 1.0, 2000, 509,
    ),
    'Polyimide': MembraneMechanicalProperties(
        'Polyimide', 2.5, 0.34, 1.43, 50, 0.2, 1090,
    ),
}
