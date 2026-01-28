"""
Aerial Image Modeling for X-ray Lithography
============================================

Computes intensity profiles through X-ray mask stacks using the
Beer-Lambert absorption law and Fresnel diffraction propagation.

Physics overview
----------------
1. Mask transmission is computed from the linear attenuation coefficient
   (empirical fits to NIST XCOM data) and absorber/membrane thicknesses.
2. The transmitted field is propagated across the mask-resist gap using
   a 1-D Fresnel integral.
3. Image contrast and resolution (FWHM) are extracted from the aerial image.

Author: Abhineet Agarwal
"""

import numpy as np
from typing import Tuple

from .materials import MATERIALS, MaterialProperties


class XRayMask:
    """X-ray mask geometry and transmission calculation.

    The mask consists of a patterned high-Z absorber on a thin
    low-Z membrane.  The feature pattern is a periodic line/space
    grating characterised by ``feature_size`` and ``pitch``.

    Args:
        absorber_material: Key into ``MATERIALS`` (e.g. 'Ta', 'W', 'Au').
        absorber_thickness: Absorber thickness in um.
        membrane_material: Key into ``MATERIALS`` (e.g. 'Si3N4', 'SiC').
        membrane_thickness: Membrane thickness in um.
        feature_size: Linewidth of the absorber features in um.
        pitch: Period of the line/space pattern in um.
    """

    def __init__(
        self,
        absorber_material: str = 'Ta',
        absorber_thickness: float = 0.5,
        membrane_material: str = 'Si3N4',
        membrane_thickness: float = 2.0,
        feature_size: float = 0.5,
        pitch: float = 1.0,
    ):
        self.absorber_key = absorber_material
        self.absorber: MaterialProperties = MATERIALS[absorber_material]
        self.absorber_thickness = absorber_thickness

        self.membrane_key = membrane_material
        self.membrane: MaterialProperties = MATERIALS[membrane_material]
        self.membrane_thickness = membrane_thickness

        self.feature_size = feature_size
        self.pitch = pitch

    def get_transmission_profile(
        self, x_positions: np.ndarray, energy_kev: float
    ) -> np.ndarray:
        """Compute mask transmission at each spatial position.

        The transmission includes the membrane background everywhere
        and the absorber attenuation over the patterned regions
        (first ``feature_size`` within each ``pitch``).

        Args:
            x_positions: 1-D array of x coordinates in um.
            energy_kev: Photon energy in keV.

        Returns:
            Transmission coefficient array (values in 0--1).
        """
        # Attenuation coefficients in 1/cm -> convert to 1/um
        mu_abs = self.absorber.get_attenuation_coefficient(energy_kev) * 1e-4
        mu_mem = self.membrane.get_attenuation_coefficient(energy_kev) * 1e-4

        x_mod = np.mod(x_positions, self.pitch)

        t_membrane = np.exp(-mu_mem * self.membrane_thickness)
        t_absorber = np.exp(-mu_abs * self.absorber_thickness)

        transmission = np.where(
            x_mod < self.feature_size,
            t_membrane * t_absorber,  # under absorber
            t_membrane,               # open area
        )
        return transmission


class AerialImageSimulator:
    """Fresnel propagation of the mask transmission to the resist plane.

    Args:
        mask: An ``XRayMask`` instance defining the mask geometry.
        gap: Mask-to-resist gap in um.
    """

    def __init__(self, mask: XRayMask, gap: float = 10.0):
        self.mask = mask
        self.gap = gap

    def fresnel_propagation(
        self,
        field_at_mask: np.ndarray,
        x_mask: np.ndarray,
        x_resist: np.ndarray,
        wavelength: float,
    ) -> np.ndarray:
        """Propagate a complex field from mask to resist via the Fresnel integral.

        For Fresnel numbers F < 0.1 the geometric (shadow) limit is
        returned directly.

        Args:
            field_at_mask: Complex field amplitude at the mask plane.
            x_mask: Spatial grid at the mask in um.
            x_resist: Spatial grid at the resist plane in um.
            wavelength: X-ray wavelength in um.

        Returns:
            Intensity (|E|^2) at the resist plane.
        """
        dx_mask = x_mask[1] - x_mask[0]

        # Fresnel number determines diffraction regime
        fresnel_number = self.mask.feature_size**2 / (wavelength * self.gap)

        if fresnel_number < 0.1:
            # Geometric shadow limit
            return np.abs(field_at_mask) ** 2

        # Numerical Fresnel propagation (direct summation)
        field_at_resist = np.zeros_like(x_resist, dtype=complex)
        for i, x_r in enumerate(x_resist):
            phase = np.exp(
                1j * np.pi * (x_mask - x_r) ** 2 / (wavelength * self.gap)
            )
            field_at_resist[i] = np.sum(field_at_mask * phase) * dx_mask

        # Free-space propagation phase
        field_at_resist *= np.exp(
            1j * 2 * np.pi * self.gap / wavelength
        ) / (1j * wavelength * self.gap)

        return np.abs(field_at_resist) ** 2

    def compute_aerial_image(
        self,
        energy_kev: float,
        x_range: float = 3.0,
        resolution: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the aerial image at the resist plane.

        Args:
            energy_kev: Photon energy in keV.
            x_range: Total spatial extent of the simulation window in um.
            resolution: Number of grid points.

        Returns:
            ``(x, intensity)`` where *x* is in um and *intensity* is the
            normalised irradiance at the resist plane.
        """
        # Wavelength: E [keV] -> lambda [nm] -> um
        wavelength_um = 1.24 / energy_kev / 1000

        x = np.linspace(-x_range / 2, x_range / 2, resolution)
        transmission = self.mask.get_transmission_profile(x, energy_kev)
        field = np.sqrt(transmission)

        if self.gap > 0:
            intensity = self.fresnel_propagation(field, x, x, wavelength_um)
        else:
            intensity = transmission

        return x, intensity

    def calculate_contrast(
        self, x: np.ndarray, intensity: np.ndarray
    ) -> float:
        """Image contrast (I_max - I_min) / (I_max + I_min) over one period.

        Args:
            x: Position array in um.
            intensity: Intensity array.

        Returns:
            Michelson contrast (0 -- 1).
        """
        period_points = int(len(x) * self.mask.pitch / (x[-1] - x[0]))
        center = len(x) // 2
        half_period = period_points // 2
        i_period = intensity[center - half_period : center + half_period]

        I_max = np.max(i_period)
        I_min = np.min(i_period)
        if (I_max + I_min) == 0:
            return 0.0
        return float((I_max - I_min) / (I_max + I_min))

    def calculate_resolution(
        self, x: np.ndarray, intensity: np.ndarray, threshold: float = 0.5
    ) -> float:
        """Estimate resolution as the FWHM of the intensity peak.

        Args:
            x: Position array in um.
            intensity: Intensity array.
            threshold: Fractional intensity level defining the width.

        Returns:
            FWHM in um, or ``nan`` if edges cannot be found.
        """
        I_norm = (intensity - intensity.min()) / (
            intensity.max() - intensity.min()
        )
        above = I_norm > threshold
        edges = np.where(np.diff(above.astype(int)))[0]
        if len(edges) >= 2:
            return float(abs(x[edges[1]] - x[edges[0]]))
        return float('nan')


# -------------------------------------------------------------------
# Parameter sweep helpers
# -------------------------------------------------------------------

def parameter_sweep_energy(
    mask: XRayMask,
    gap: float = 10.0,
    energies: np.ndarray | None = None,
) -> dict:
    """Sweep photon energy and return contrast / resolution arrays.

    Args:
        mask: ``XRayMask`` instance.
        gap: Mask-to-resist gap in um.
        energies: 1-D array of energies in keV. Defaults to
            ``np.linspace(0.5, 5.0, 10)``.

    Returns:
        Dict with keys ``energy_kev``, ``contrast``, ``resolution``.
    """
    if energies is None:
        energies = np.linspace(0.5, 5.0, 10)

    results: dict = {
        'energy_kev': energies,
        'contrast': np.zeros_like(energies),
        'resolution': np.zeros_like(energies),
    }

    sim = AerialImageSimulator(mask, gap)
    for i, energy in enumerate(energies):
        x, intensity = sim.compute_aerial_image(energy)
        results['contrast'][i] = sim.calculate_contrast(x, intensity)
        results['resolution'][i] = sim.calculate_resolution(x, intensity)

    return results


def parameter_sweep_gap(
    mask: XRayMask,
    energy_kev: float = 1.5,
    gaps: np.ndarray | None = None,
) -> dict:
    """Sweep mask-resist gap and return contrast / resolution arrays.

    Args:
        mask: ``XRayMask`` instance.
        energy_kev: Photon energy in keV.
        gaps: 1-D array of gaps in um. Defaults to
            ``np.linspace(1, 50, 10)``.

    Returns:
        Dict with keys ``gap_um``, ``contrast``, ``resolution``.
    """
    if gaps is None:
        gaps = np.linspace(1, 50, 10)

    results: dict = {
        'gap_um': gaps,
        'contrast': np.zeros_like(gaps),
        'resolution': np.zeros_like(gaps),
    }

    for i, gap in enumerate(gaps):
        sim = AerialImageSimulator(mask, gap)
        x, intensity = sim.compute_aerial_image(energy_kev)
        results['contrast'][i] = sim.calculate_contrast(x, intensity)
        results['resolution'][i] = sim.calculate_resolution(x, intensity)

    return results


def parameter_sweep_absorber_thickness(
    mask_base: XRayMask,
    energy_kev: float = 1.5,
    gap: float = 10.0,
    thicknesses: np.ndarray | None = None,
) -> dict:
    """Sweep absorber thickness and return contrast / transmission arrays.

    Args:
        mask_base: Reference mask (membrane params are reused).
        energy_kev: Photon energy in keV.
        gap: Mask-to-resist gap in um.
        thicknesses: 1-D array of absorber thicknesses in um.

    Returns:
        Dict with keys ``thickness_um``, ``contrast``,
        ``transmission_absorber``.
    """
    if thicknesses is None:
        thicknesses = np.linspace(0.1, 1.0, 10)

    results: dict = {
        'thickness_um': thicknesses,
        'contrast': np.zeros_like(thicknesses),
        'transmission_absorber': np.zeros_like(thicknesses),
    }

    for i, thickness in enumerate(thicknesses):
        mask = XRayMask(
            absorber_material=mask_base.absorber_key,
            absorber_thickness=thickness,
            membrane_material=mask_base.membrane_key,
            membrane_thickness=mask_base.membrane_thickness,
            feature_size=mask_base.feature_size,
            pitch=mask_base.pitch,
        )
        sim = AerialImageSimulator(mask, gap)
        x, intensity = sim.compute_aerial_image(energy_kev)
        results['contrast'][i] = sim.calculate_contrast(x, intensity)

        mu = mask.absorber.get_attenuation_coefficient(energy_kev) * 1e-4
        results['transmission_absorber'][i] = np.exp(-mu * thickness)

    return results


def parameter_sweep_absorber_material(
    mask_base: XRayMask,
    energy_kev: float = 1.5,
    gap: float = 10.0,
    absorber_materials: list[str] | None = None,
) -> dict:
    """Compare absorber materials at fixed geometry.

    Args:
        mask_base: Reference mask (geometry is reused).
        energy_kev: Photon energy in keV.
        gap: Mask-to-resist gap in um.
        absorber_materials: List of material keys. Defaults to
            ``['Ta', 'W', 'Au']``.

    Returns:
        Dict mapping material key to ``{'contrast': float,
        'transmission_absorber': float}``.
    """
    if absorber_materials is None:
        absorber_materials = ['Ta', 'W', 'Au']

    results: dict = {}
    for material in absorber_materials:
        mask = XRayMask(
            absorber_material=material,
            absorber_thickness=mask_base.absorber_thickness,
            membrane_material=mask_base.membrane_key,
            membrane_thickness=mask_base.membrane_thickness,
            feature_size=mask_base.feature_size,
            pitch=mask_base.pitch,
        )
        sim = AerialImageSimulator(mask, gap)
        x, intensity = sim.compute_aerial_image(energy_kev)
        contrast = sim.calculate_contrast(x, intensity)

        mu = mask.absorber.get_attenuation_coefficient(energy_kev) * 1e-4
        transmission = np.exp(-mu * mask.absorber_thickness)

        results[material] = {
            'contrast': contrast,
            'transmission_absorber': transmission,
        }

    return results
