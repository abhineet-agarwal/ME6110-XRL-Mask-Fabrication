"""
Thermal-Mechanical Modeling for X-ray Masks
============================================

Analytical models for membrane deflection, thermal stress, and
temperature distribution under X-ray exposure.  These provide rapid
parameter-space exploration as simplified alternatives to FEM.

Physics overview
----------------
1. **Pressure deflection** -- classical thin-plate bending for clamped
   circular or square membranes.
2. **Thermal stress** -- biaxial stress from constrained thermal
   expansion: sigma = E * alpha * dT / (1 - nu).
3. **Thermal deflection** -- empirically calibrated formula
   (see ``thermal_deflection`` docstring for details).
4. **Temperature rise** -- steady-state estimates from convective
   cooling and in-plane conduction.

Author: Abhineet Agarwal
"""

import numpy as np
from typing import Tuple, Optional

from .materials import MembraneMechanicalProperties, MEMBRANES


class MembraneMechanics:
    """Analytical plate-theory model for membrane deflection and stress.

    The membrane is assumed to be a thin plate clamped at its edges,
    loaded by uniform pressure or a thermal gradient.

    Args:
        material: ``MembraneMechanicalProperties`` instance.
        thickness: Membrane thickness in um.
        size: Side length (square) or diameter (circular) in mm.
        geometry: ``'square'`` or ``'circular'``.
    """

    def __init__(
        self,
        material: MembraneMechanicalProperties,
        thickness: float,
        size: float,
        geometry: str = 'square',
    ):
        self.material = material
        self.thickness = thickness * 1e-6   # um -> m
        self.size = size * 1e-3             # mm -> m
        self.geometry = geometry

    def pressure_deflection_center(self, pressure: float) -> float:
        """Maximum centre deflection under uniform pressure.

        Uses classical thin-plate formulae for clamped edges.

        Args:
            pressure: Uniform pressure in Pa.

        Returns:
            Centre deflection in um.
        """
        E = self.material.youngs_modulus * 1e9   # GPa -> Pa
        nu = self.material.poisson_ratio
        t = self.thickness
        a = self.size / 2  # half-width or radius

        # Flexural rigidity
        D = E * t**3 / (12 * (1 - nu**2))

        if self.geometry == 'circular':
            # Clamped circular plate: w_max = p * a^4 / (64 * D)
            w_max = pressure * a**4 / (64 * D)
        else:
            # Clamped square plate (Timoshenko):
            # w_max ~ 0.0138 * p * (2a)^4 / D
            w_max = 0.0138 * pressure * (2 * a)**4 / D

        return w_max * 1e6  # m -> um

    def thermal_stress(self, delta_T: float) -> float:
        """Biaxial thermal stress from constrained expansion.

        For a membrane with clamped edges the thermal stress is:
            sigma = E * alpha * dT / (1 - nu)

        Args:
            delta_T: In-plane temperature difference in K.

        Returns:
            Thermal stress in MPa.
        """
        E = self.material.youngs_modulus * 1e9        # Pa
        nu = self.material.poisson_ratio
        alpha = self.material.thermal_expansion * 1e-6  # 1/K

        sigma = E * alpha * delta_T / (1 - nu)
        return sigma / 1e6  # Pa -> MPa

    def thermal_deflection(self, delta_T: float) -> float:
        """Deflection from a through-thickness thermal gradient.

        Uses the empirically calibrated formula:
            w_max = C * alpha * dT * a^2 / t

        where ``C = 1e-6`` is a calibration constant tuned to match
        FEM literature values (Si3N4 membranes yield ~0.01--0.1 um
        deflection at 0.1 W absorbed power).

        .. warning::

           The calibration constant ``C`` is empirical.  For
           quantitative design work validate against FEM or
           experiment for your specific geometry.

        Args:
            delta_T: Through-thickness temperature difference in K.

        Returns:
            Maximum centre deflection in um.
        """
        alpha = self.material.thermal_expansion * 1e-6  # 1/K
        a = self.size / 2   # m
        t = self.thickness  # m

        # Empirical calibration constant (see docstring)
        C = 1e-6

        w_max = C * alpha * delta_T * a**2 / t
        return w_max * 1e6  # m -> um

    def intrinsic_stress_deflection(self, stress: float) -> float:
        """Deflection from intrinsic (residual) film stress.

        The stress is converted to an equivalent pressure and solved
        with the plate bending formula.

        Args:
            stress: Intrinsic stress in MPa (positive = tensile).

        Returns:
            Centre deflection in um.
        """
        a = self.size / 2
        t = self.thickness
        sigma = stress * 1e6  # MPa -> Pa

        p_equiv = 6 * sigma * t / a**2
        return self.pressure_deflection_center(p_equiv)


class ThermalAnalysis:
    """Steady-state thermal analysis of a membrane under X-ray exposure.

    Args:
        membrane: ``MembraneMechanics`` instance.
    """

    def __init__(self, membrane: MembraneMechanics):
        self.membrane = membrane

    def absorbed_power(
        self,
        beam_power: float,
        absorption_fraction: float = 0.1,
        area: Optional[float] = None,
    ) -> float:
        """Power absorbed by the membrane.

        Args:
            beam_power: Incident X-ray beam power in W.
            absorption_fraction: Fraction of incident power absorbed
                (typically ~0.1 for thin low-Z membranes).
            area: Exposure area in mm^2 (unused in current model but
                reserved for non-uniform illumination).

        Returns:
            Absorbed power in W.
        """
        return beam_power * absorption_fraction

    def steady_state_center_temp_rise(self, absorbed_power: float) -> float:
        """Centre temperature rise above ambient (convection estimate).

        Assumes convective cooling from both surfaces with a heat
        transfer coefficient of 10 W/(m^2*K).

        Args:
            absorbed_power: Power deposited in the membrane in W.

        Returns:
            Steady-state temperature rise in K.
        """
        h_conv = 10.0  # W/(m^2*K)
        A_surface = 2 * self.membrane.size**2  # both sides, m^2
        return absorbed_power / (h_conv * A_surface)

    def steady_state_in_plane_gradient(self, absorbed_power: float) -> float:
        """Centre-to-edge in-plane temperature gradient.

        This gradient drives thermal stress and, through the
        assumed 10 % coupling, the through-thickness gradient
        used for deflection.

        Formula: dT = P_abs * L / (k * A * t)
        with L = size/4 as a characteristic conduction length.

        Args:
            absorbed_power: Power deposited in the membrane in W.

        Returns:
            In-plane temperature difference in K.
        """
        k = self.membrane.material.thermal_conductivity
        t = self.membrane.thickness
        L = self.membrane.size / 4
        A = self.membrane.size**2

        return absorbed_power * L / (k * A * t)

    def thermal_time_constant(self) -> float:
        """Thermal time constant for transient analysis.

        tau = rho * c_p * t / (2 * h)  with h = 10 W/(m^2*K).

        Returns:
            Time constant in seconds.
        """
        rho = self.membrane.material.density * 1e3  # g/cm^3 -> kg/m^3
        c_p = self.membrane.material.specific_heat   # J/(kg*K)
        t = self.membrane.thickness
        h = 10.0  # W/(m^2*K)

        return rho * c_p * t / (2 * h)

    def temperature_distribution_1d(
        self, absorbed_power: float, n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """1-D steady-state temperature profile across the membrane.

        Assumes uniform volumetric heat generation and conduction to
        clamped (isothermal) edges, giving a parabolic profile.

        Args:
            absorbed_power: Power deposited in the membrane in W.
            n_points: Number of spatial grid points.

        Returns:
            ``(x_mm, T_K)`` -- position in mm and temperature in K.
        """
        k = self.membrane.material.thermal_conductivity
        t = self.membrane.thickness
        L = self.membrane.size / 2  # half-width, m

        q_vol = absorbed_power / (self.membrane.size**2 * t)
        x = np.linspace(-L, L, n_points)

        T_ref = 300.0  # reference temperature, K
        T = T_ref - (q_vol / (2 * k)) * x**2

        return x * 1e3, T  # m -> mm


# -------------------------------------------------------------------
# High-level analysis helpers
# -------------------------------------------------------------------

# Through-thickness gradient fraction of in-plane gradient.
# This is a simplifying assumption: the actual ratio depends on
# membrane aspect ratio and boundary conditions.  A value of 0.1
# (10 %) gives results consistent with FEM literature for typical
# XRL mask geometries.
_THROUGH_THICKNESS_FRACTION: float = 0.1


def exposure_scenario_analysis(
    membrane: MembraneMechanics,
    beam_power_range: np.ndarray | None = None,
) -> dict:
    """Sweep beam power and return temperature / stress / deflection.

    Args:
        membrane: ``MembraneMechanics`` instance.
        beam_power_range: 1-D array of beam powers in W.
            Defaults to ``np.logspace(-3, 0, 20)``.

    Returns:
        Dict with keys ``beam_power_W``, ``temperature_rise_K``,
        ``thermal_stress_MPa``, ``deflection_um``.
    """
    if beam_power_range is None:
        beam_power_range = np.logspace(-3, 0, 20)

    thermal = ThermalAnalysis(membrane)

    results: dict = {
        'beam_power_W': beam_power_range,
        'temperature_rise_K': np.zeros_like(beam_power_range),
        'thermal_stress_MPa': np.zeros_like(beam_power_range),
        'deflection_um': np.zeros_like(beam_power_range),
    }

    for i, power in enumerate(beam_power_range):
        absorbed = thermal.absorbed_power(power, absorption_fraction=0.1)
        dT_in_plane = thermal.steady_state_in_plane_gradient(absorbed)
        dT_rise = thermal.steady_state_center_temp_rise(absorbed)
        dT_through = dT_in_plane * _THROUGH_THICKNESS_FRACTION

        results['temperature_rise_K'][i] = dT_rise
        results['thermal_stress_MPa'][i] = membrane.thermal_stress(dT_in_plane)
        results['deflection_um'][i] = membrane.thermal_deflection(dT_through)

    return results


def material_comparison(
    thickness: float = 2.0,
    size: float = 50.0,
    beam_power: float = 0.1,
) -> dict:
    """Compare membrane materials at identical geometry and loading.

    Args:
        thickness: Membrane thickness in um.
        size: Membrane size (side length) in mm.
        beam_power: Incident beam power in W.

    Returns:
        Dict mapping material key to ``{'delta_T': float,
        'stress': float, 'deflection': float,
        'time_constant': float}``.
    """
    results: dict = {}
    for mat_name, mat_props in MEMBRANES.items():
        membrane = MembraneMechanics(mat_props, thickness, size, 'square')
        thermal = ThermalAnalysis(membrane)

        absorbed = thermal.absorbed_power(beam_power, absorption_fraction=0.1)
        dT_in_plane = thermal.steady_state_in_plane_gradient(absorbed)
        dT_rise = thermal.steady_state_center_temp_rise(absorbed)
        dT_through = dT_in_plane * _THROUGH_THICKNESS_FRACTION

        results[mat_name] = {
            'delta_T': dT_rise,
            'stress': membrane.thermal_stress(dT_in_plane),
            'deflection': membrane.thermal_deflection(dT_through),
            'time_constant': thermal.thermal_time_constant(),
        }

    return results
