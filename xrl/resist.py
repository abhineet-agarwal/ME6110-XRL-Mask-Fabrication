"""
Resist Response Simulation for X-ray Lithography
==================================================

Models photoresist exposure, development, and pattern formation from
aerial images produced by ``aerial_image.py``.

Physics overview
----------------
1. Photon absorption in the resist follows Beer-Lambert with empirical
   attenuation fits for organic materials in the 0.5--5 keV range.
2. Dose is computed from incident flux, exposure time, and absorbed
   fraction.
3. Photon shot noise is added via Poisson statistics; acid/electron
   diffusion blur is modelled as a Gaussian convolution.
4. Development uses a standard contrast-curve model with separate
   positive/negative tone implementations.
5. Critical dimension (CD) and line-edge roughness (LER, 3-sigma)
   are extracted from the developed profile.

Author: Abhineet Agarwal
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Tuple

from .materials import ResistProperties, RESISTS


# Reference photon flux at unit normalised intensity [photons/(s*cm^2)]
REFERENCE_FLUX: float = 1e13


class ResistExposureModel:
    """Resist exposure, development, and metrology model.

    Args:
        resist: A ``ResistProperties`` instance.
    """

    def __init__(self, resist: ResistProperties):
        self.resist = resist

    # -----------------------------------------------------------------
    # Absorption
    # -----------------------------------------------------------------

    def absorption_coefficient(self, energy_kev: float) -> float:
        """Linear absorption coefficient for the resist.

        Empirical power-law fits for generic organic resists in the
        XRL energy range.

        Args:
            energy_kev: Photon energy in keV.

        Returns:
            Absorption coefficient mu in 1/um.
        """
        if energy_kev < 1.0:
            return 0.5 * energy_kev ** (-2.5)
        elif energy_kev < 2.0:
            return 0.3 * energy_kev ** (-2.3)
        else:
            return 0.2 * energy_kev ** (-2.0)

    # -----------------------------------------------------------------
    # Dose calculation
    # -----------------------------------------------------------------

    def absorbed_dose_profile(
        self,
        intensity: np.ndarray,
        energy_kev: float,
        exposure_time: float,
    ) -> np.ndarray:
        """Absorbed energy density in the resist film.

        Args:
            intensity: Normalised aerial-image intensity (0--1).
            energy_kev: Photon energy in keV.
            exposure_time: Exposure time in seconds.

        Returns:
            Absorbed dose in mJ/cm^2.
        """
        mu = self.absorption_coefficient(energy_kev)
        f_absorbed = 1 - np.exp(-mu * self.resist.thickness)

        energy_per_photon_J = energy_kev * 1.602e-16  # keV -> J
        flux = intensity * REFERENCE_FLUX  # photons/(s*cm^2)

        dose_mJ = flux * exposure_time * energy_per_photon_J * f_absorbed * 1e3
        return dose_mJ

    # -----------------------------------------------------------------
    # Stochastic effects
    # -----------------------------------------------------------------

    def add_photon_shot_noise(
        self, dose: np.ndarray, energy_kev: float
    ) -> np.ndarray:
        """Add Poisson shot noise arising from discrete photon statistics.

        Args:
            dose: Dose array in mJ/cm^2.
            energy_kev: Photon energy in keV.

        Returns:
            Noisy dose array in mJ/cm^2.
        """
        energy_per_photon_J = energy_kev * 1.602e-16
        n_photons = dose * 1e-3 / energy_per_photon_J
        n_photons_noisy = np.random.poisson(np.clip(n_photons, 0, None))
        return n_photons_noisy * energy_per_photon_J * 1e3

    def add_resist_blur(
        self, dose: np.ndarray, x: np.ndarray
    ) -> np.ndarray:
        """Gaussian blur from secondary-electron range / acid diffusion.

        The blur sigma is defined by ``resist.blur`` (in um).

        Args:
            dose: Dose array in mJ/cm^2.
            x: Spatial position array in um.

        Returns:
            Blurred dose array.
        """
        dx = x[1] - x[0]
        sigma_points = self.resist.blur / dx
        return gaussian_filter1d(dose, sigma=sigma_points, mode='wrap')

    # -----------------------------------------------------------------
    # Development
    # -----------------------------------------------------------------

    def development_model(
        self, dose: np.ndarray, development_threshold: float = 1.0
    ) -> np.ndarray:
        """Standard contrast-curve development model.

        Returns the normalised remaining resist thickness (1 = fully
        remaining, 0 = fully removed).

        Args:
            dose: Absorbed dose array in mJ/cm^2.
            development_threshold: Multiplicative factor on D0.

        Returns:
            Remaining thickness fraction (0--1).
        """
        D0 = self.resist.sensitivity * development_threshold
        D_norm = dose / D0

        if self.resist.tone == 'positive':
            remaining = self._develop_positive(D_norm)
        else:
            remaining = self._develop_negative(D_norm)

        return np.clip(remaining, 0.0, 1.0)

    def _develop_positive(self, D_norm: np.ndarray) -> np.ndarray:
        """Positive-tone development: higher dose removes more resist."""
        gamma = self.resist.contrast
        remaining = np.ones_like(D_norm)

        # Above clearing dose: power-law removal
        above = D_norm > 1.0
        if np.any(above):
            remaining[above] = np.clip(D_norm[above] ** (-gamma), 0.0, 1.0)

        # Gradual transition near threshold (0.7 -- 1.0 x D0)
        partial = (D_norm > 0.7) & (D_norm <= 1.0)
        if np.any(partial):
            transition = (D_norm[partial] - 0.7) / 0.3
            remaining[partial] = 1.0 - 0.5 * transition

        return remaining

    def _develop_negative(self, D_norm: np.ndarray) -> np.ndarray:
        """Negative-tone development: higher dose cross-links resist."""
        gamma = self.resist.contrast
        remaining = np.zeros_like(D_norm)

        above = D_norm > 0.5
        if np.any(above):
            remaining[above] = 1.0 - D_norm[above] ** (-gamma)

        return remaining

    # -----------------------------------------------------------------
    # Metrology
    # -----------------------------------------------------------------

    def calculate_cd(
        self,
        x: np.ndarray,
        profile: np.ndarray,
        threshold: float = 0.5,
    ) -> float:
        """Critical dimension (linewidth) at a given threshold.

        For positive resist the CD is the width of the *removed*
        region; for negative resist it is the width of the
        *cross-linked* region.

        Args:
            x: Position array in um.
            profile: Developed resist profile (0--1).
            threshold: Not used directly; internal adaptive thresholds
                are applied per tone.

        Returns:
            CD in um, or ``nan`` if edges cannot be resolved.
        """
        smoothed = gaussian_filter1d(profile, sigma=2)
        p_min, p_max = smoothed.min(), smoothed.max()
        p_range = p_max - p_min

        if p_range < 0.05:
            return float('nan')

        if self.resist.tone == 'positive':
            threshold_value = p_min + p_range * 0.4
            selected = smoothed < threshold_value
        else:
            threshold_value = p_min + p_range * 0.6
            selected = smoothed > threshold_value

        transitions = np.diff(selected.astype(int))
        falling = np.where(transitions == -1)[0]
        rising = np.where(transitions == 1)[0]

        widths = []
        for f_idx in falling:
            matches = rising[rising > f_idx]
            if len(matches) > 0:
                w = abs(x[matches[0]] - x[f_idx])
                if 0.05 < w < 2.0:
                    widths.append(w)

        if widths:
            return float(np.median(widths))
        return float('nan')

    def calculate_ler(
        self,
        x: np.ndarray,
        profile: np.ndarray,
        threshold: float = 0.5,
        n_samples: int = 50,
    ) -> float:
        """Line-edge roughness estimated as 3-sigma of edge position.

        Uses Monte-Carlo resampling with resist-blur-scaled noise.

        Args:
            x: Position array in um.
            profile: Developed resist profile (0--1).
            threshold: Not used directly (adaptive thresholds per tone).
            n_samples: Number of Monte-Carlo iterations.

        Returns:
            LER (3-sigma) in nm, or ``nan`` if insufficient edges found.
        """
        smoothed = gaussian_filter1d(profile, sigma=3)
        p_min, p_max = smoothed.min(), smoothed.max()
        p_range = p_max - p_min

        if p_range < 0.05:
            return float('nan')

        if self.resist.tone == 'positive':
            threshold_value = p_min + p_range * 0.4
            detect_below = True
        else:
            threshold_value = p_min + p_range * 0.6
            detect_below = False

        # Noise amplitude scales with sqrt(blur) normalised to PMMA
        noise_scale = np.sqrt(self.resist.blur / 0.05)
        base_noise = p_range * 0.005 * noise_scale

        edge_positions: list[float] = []
        for _ in range(n_samples):
            noisy = smoothed + np.random.normal(0, base_noise, size=profile.shape)
            noisy = gaussian_filter1d(noisy, sigma=1)

            if detect_below:
                selected = noisy < threshold_value
            else:
                selected = noisy > threshold_value

            transitions = np.diff(selected.astype(int))
            falling = np.where(transitions == -1)[0]
            if len(falling) > 0:
                edge_positions.append(x[falling[0]])

        if len(edge_positions) >= 3:
            return float(3 * np.std(edge_positions) * 1000)  # um -> nm
        return float('nan')


# -------------------------------------------------------------------
# High-level convenience functions
# -------------------------------------------------------------------

def simulate_full_exposure(
    mask_intensity: np.ndarray,
    x: np.ndarray,
    resist: ResistProperties,
    energy_kev: float = 1.5,
    dose_factor: float = 1.0,
    include_noise: bool = True,
    n_samples_ler: int = 50,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """End-to-end resist simulation: aerial image -> developed profile.

    Args:
        mask_intensity: Normalised aerial-image intensity (0--1).
        x: Spatial grid in um.
        resist: ``ResistProperties`` instance.
        energy_kev: Photon energy in keV.
        dose_factor: Multiplicative factor on clearing dose D0.
        include_noise: Whether to add shot noise and blur.
        n_samples_ler: Monte-Carlo samples for LER estimation.

    Returns:
        ``(dose, developed, metrics)`` where *metrics* is a dict
        with keys ``cd_um``, ``ler_nm``, ``contrast``, ``dose_min``,
        ``dose_max``.
    """
    model = ResistExposureModel(resist)

    # Exposure time to reach target dose at unit intensity
    target_dose = resist.sensitivity * dose_factor
    energy_per_photon_J = energy_kev * 1.602e-16
    mu = model.absorption_coefficient(energy_kev)
    f_absorbed = 1 - np.exp(-mu * resist.thickness)
    dose_per_second = REFERENCE_FLUX * energy_per_photon_J * f_absorbed * 1e3
    exposure_time = target_dose / dose_per_second

    dose = model.absorbed_dose_profile(mask_intensity, energy_kev, exposure_time)

    if include_noise:
        dose = model.add_photon_shot_noise(dose, energy_kev)
        dose = model.add_resist_blur(dose, x)

    developed = model.development_model(dose)

    cd = model.calculate_cd(x, developed)
    ler = model.calculate_ler(x, developed, n_samples=n_samples_ler) if include_noise else 0.0

    metrics = {
        'cd_um': cd,
        'ler_nm': ler,
        'contrast': float(np.max(developed) - np.min(developed)),
        'dose_min': float(np.min(dose)),
        'dose_max': float(np.max(dose)),
    }
    return dose, developed, metrics


def dose_sweep_study(
    mask_intensity: np.ndarray,
    x: np.ndarray,
    resist: ResistProperties,
    energy_kev: float,
    dose_factors: np.ndarray,
    n_samples_ler: int = 50,
) -> dict:
    """Sweep exposure dose and return CD / LER arrays.

    Args:
        mask_intensity: Normalised aerial-image intensity.
        x: Spatial grid in um.
        resist: ``ResistProperties`` instance.
        energy_kev: Photon energy in keV.
        dose_factors: Array of dose multipliers.
        n_samples_ler: Monte-Carlo samples for LER.

    Returns:
        Dict with keys ``dose_factor``, ``cd_um``, ``ler_nm``.
    """
    results: dict = {
        'dose_factor': dose_factors,
        'cd_um': np.zeros_like(dose_factors),
        'ler_nm': np.zeros_like(dose_factors),
    }

    for i, factor in enumerate(dose_factors):
        _, _, metrics = simulate_full_exposure(
            mask_intensity, x, resist, energy_kev, factor,
            include_noise=True, n_samples_ler=n_samples_ler,
        )
        results['cd_um'][i] = metrics['cd_um']
        results['ler_nm'][i] = metrics['ler_nm']

    return results


def resist_comparison(
    mask_intensity: np.ndarray,
    x: np.ndarray,
    energy_kev: float = 1.5,
    dose_factor: float = 1.0,
    n_samples_ler: int = 50,
) -> dict:
    """Compare all resists in the database at fixed conditions.

    Args:
        mask_intensity: Normalised aerial-image intensity.
        x: Spatial grid in um.
        energy_kev: Photon energy in keV.
        dose_factor: Dose multiplier.
        n_samples_ler: Monte-Carlo samples for LER.

    Returns:
        Dict mapping resist name to ``{'metrics': dict,
        'developed_profile': ndarray}``.
    """
    results: dict = {}
    for name, resist in RESISTS.items():
        _, developed, metrics = simulate_full_exposure(
            mask_intensity, x, resist, energy_kev, dose_factor,
            include_noise=True, n_samples_ler=n_samples_ler,
        )
        results[name] = {'metrics': metrics, 'developed_profile': developed}
    return results
