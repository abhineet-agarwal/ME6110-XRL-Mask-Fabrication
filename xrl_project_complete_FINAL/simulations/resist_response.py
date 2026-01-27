"""
Fixed Resist Response Simulation for X-ray Lithography
=======================================================
... (original header content) ...
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import poisson
from dataclasses import dataclass
from typing import Tuple, Optional


# ... (ResistProperties and RESISTS definition, ResistExposureModel class, and simulate_full_exposure) ...

@dataclass
class ResistProperties:
    """Properties of X-ray resist material"""
    name: str
    density: float  # g/cm³
    sensitivity: float  # mJ/cm² (D0)
    contrast: float  # γ (photoresist contrast)
    blur: float  # μm (acid diffusion length / electron range)
    thickness: float  # μm
    tone: str  # 'positive' or 'negative'


# Resist database
RESISTS = {
    'PMMA': ResistProperties('PMMA', 1.18, 500, 7.0, 0.05, 1.0, 'positive'),
    'ZEP520A': ResistProperties('ZEP520A', 1.11, 80, 9.0, 0.03, 0.5, 'positive'),
    'SU8': ResistProperties('SU-8', 1.19, 150, 4.0, 0.08, 10.0, 'negative'),
    'HSQ': ResistProperties('HSQ', 1.4, 800, 1.5, 0.02, 0.3, 'negative'),
}


class ResistExposureModel:
    """
    Models resist exposure with stochastic photon absorption
    """
    
    def __init__(self, resist: ResistProperties):
        self.resist = resist
    
    def absorption_coefficient(self, energy_kev: float) -> float:
        """
        Calculate resist absorption coefficient.
        Uses realistic values for organic resists in XRL range.
        
        Returns: μ (1/μm)
        """
        if energy_kev < 1.0:
            mu = 0.5 * energy_kev**(-2.5)
        elif energy_kev < 2.0:
            mu = 0.3 * energy_kev**(-2.3)
        else:
            mu = 0.2 * energy_kev**(-2.0)
        
        return mu
    
    def absorbed_dose_profile(self,
                             intensity: np.ndarray,
                             energy_kev: float,
                             exposure_time: float) -> np.ndarray:
        """
        Calculate absorbed energy density in resist.
        
        Args:
            intensity: Incident intensity profile (normalized 0-1)
            energy_kev: Photon energy
            exposure_time: Exposure time (s)
        
        Returns:
            Absorbed dose (mJ/cm²)
        """
        mu = self.absorption_coefficient(energy_kev)
        f_absorbed = 1 - np.exp(-mu * self.resist.thickness)
        
        reference_flux = 1e13  # photons/(s·cm²) at intensity = 1.0
        flux = intensity * reference_flux
        energy_per_photon = energy_kev * 1.602e-16  # J
        
        dose = flux * exposure_time * energy_per_photon * f_absorbed * 1e3  # mJ/cm²
        
        return dose
    
    def add_photon_shot_noise(self,
                             dose: np.ndarray,
                             energy_kev: float) -> np.ndarray:
        """
        Add Poisson shot noise from discrete photon statistics.
        """
        energy_per_photon = energy_kev * 1.602e-16  # J
        n_photons = dose * 1e-3 / energy_per_photon
        n_photons_noisy = np.random.poisson(n_photons)
        dose_noisy = n_photons_noisy * energy_per_photon * 1e3
        
        return dose_noisy
    
    def add_resist_blur(self, dose: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur from electron/acid diffusion in resist.
        """
        dx = x[1] - x[0]
        sigma_points = self.resist.blur / dx
        dose_blurred = gaussian_filter1d(dose, sigma=sigma_points, mode='wrap')
        
        return dose_blurred
    
    def development_model(self,
                         dose: np.ndarray,
                         development_threshold: float = 1.0) -> np.ndarray:
        """
        Standard photoresist development model.
        
        For positive resist: Higher dose removes more resist
        Returns normalized remaining thickness (1 = fully remaining, 0 = fully removed)
        """
        D0 = self.resist.sensitivity * development_threshold
        D_norm = dose / D0
        
        if self.resist.tone == 'positive':
            gamma = self.resist.contrast
            remaining = np.ones_like(dose)
            
            # Where dose exceeds threshold, resist is removed
            mask = D_norm > 1.0
            if np.any(mask):
                # Standard development: remaining = (D0/D)^γ
                remaining[mask] = D_norm[mask]**(-gamma)
                remaining[mask] = np.clip(remaining[mask], 0.0, 1.0)
            
            # Gradual transition near threshold
            partial_mask = (D_norm > 0.7) & (D_norm <= 1.0)
            if np.any(partial_mask):
                # Smooth transition
                transition = (D_norm[partial_mask] - 0.7) / 0.3
                remaining[partial_mask] = 1.0 - 0.5 * transition
        else:
            # Negative resist
            gamma = self.resist.contrast
            remaining = np.zeros_like(dose)
            mask = D_norm > 0.5
            if np.any(mask):
                remaining[mask] = 1.0 - D_norm[mask]**(-gamma)
        
        remaining = np.clip(remaining, 0.0, 1.0)
        
        return remaining
    
    def calculate_cd(self,
                    x: np.ndarray,
                    profile: np.ndarray,
                    threshold: float = 0.5) -> float:
        """
        Calculate critical dimension (linewidth) at given threshold.
        
        For positive resist: HIGH = remaining, LOW = removed → measure LOW regions
        For negative resist: LOW = no resist, HIGH = crosslinked → measure HIGH regions
        
        Returns:
            CD in μm
        """
        # Smooth profile
        smoothed = gaussian_filter1d(profile, sigma=2)
        
        # Check contrast
        profile_min = smoothed.min()
        profile_max = smoothed.max()
        profile_range = profile_max - profile_min
        
        if profile_range < 0.05:
            return np.nan
        
        # Adapt threshold based on resist type
        if self.resist.tone == 'positive':
            # For positive: measure REMOVED areas (LOW values)
            threshold_value = profile_min + profile_range * 0.4
            below_threshold = smoothed < threshold_value
        else:
            # For negative: measure CROSSLINKED areas (HIGH values)
            threshold_value = profile_min + profile_range * 0.6
            below_threshold = smoothed > threshold_value
        
        # Find edges
        transitions = np.diff(below_threshold.astype(int))
        falling_edges = np.where(transitions == -1)[0]
        rising_edges = np.where(transitions == 1)[0]
        
        # Measure widths of feature regions
        widths = []
        
        # Match each falling edge with next rising edge
        for fall_idx in falling_edges:
            matching_rises = rising_edges[rising_edges > fall_idx]
            if len(matching_rises) > 0:
                rise_idx = matching_rises[0]
                width = abs(x[rise_idx] - x[fall_idx])
                # Accept reasonable feature sizes
                if 0.05 < width < 2.0:
                    widths.append(width)
        
        if len(widths) > 0:
            return float(np.median(widths))
        
        return np.nan
    
    def calculate_ler(self,
                     x: np.ndarray,
                     profile: np.ndarray,
                     threshold: float = 0.5,
                     n_samples: int = 50) -> float:
        """
        Calculate line-edge roughness (3σ).
        Handles both positive and negative resists.
        
        Returns:
            LER (3σ) in nm
        """
        # Heavier smoothing to get realistic edge detection
        smoothed = gaussian_filter1d(profile, sigma=3)
        
        profile_min = smoothed.min()
        profile_max = smoothed.max()
        profile_range = profile_max - profile_min
        
        if profile_range < 0.05:
            return np.nan
        
        # Adapt threshold based on resist type
        if self.resist.tone == 'positive':
            threshold_value = profile_min + profile_range * 0.4
            detect_below = True
        else:
            threshold_value = profile_min + profile_range * 0.6
            detect_below = False
        
        edge_positions = []
        
        # Scale noise based on resist blur - lower blur = lower LER
        # Typical LER scales with ~sqrt(blur) in practice
        noise_scale = np.sqrt(self.resist.blur / 0.05)  # Normalized to PMMA blur
        base_noise = profile_range * 0.005 * noise_scale  # Reduced base noise
        
        for _ in range(n_samples):
            # Add small, resist-dependent noise
            noisy = smoothed + np.random.normal(0, base_noise, size=profile.shape)
            # Light smoothing of noise
            noisy = gaussian_filter1d(noisy, sigma=1)
            
            # Find first edge
            if detect_below:
                below_threshold = noisy < threshold_value
            else:
                below_threshold = noisy > threshold_value
                
            transitions = np.diff(below_threshold.astype(int))
            falling_edges = np.where(transitions == -1)[0]
            
            if len(falling_edges) > 0:
                edge_positions.append(x[falling_edges[0]])
        
        if len(edge_positions) >= 3:
            ler_3sigma = 3 * np.std(edge_positions) * 1000  # Convert to nm
            return ler_3sigma
        
        return np.nan


def simulate_full_exposure(mask_intensity: np.ndarray,
                          x: np.ndarray,
                          resist: ResistProperties,
                          energy_kev: float = 1.5,
                          dose_factor: float = 1.0,
                          include_noise: bool = True) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Full simulation from aerial image to developed profile.
    """
    model = ResistExposureModel(resist)
    
    # Calculate exposure parameters
    target_dose = resist.sensitivity * dose_factor
    reference_flux = 1e13
    energy_per_photon = energy_kev * 1.602e-16
    mu = model.absorption_coefficient(energy_kev)
    f_absorbed = 1 - np.exp(-mu * resist.thickness)
    
    dose_per_second = reference_flux * energy_per_photon * f_absorbed * 1e3
    exposure_time = target_dose / dose_per_second
    
    # Absorbed dose
    dose = model.absorbed_dose_profile(mask_intensity, energy_kev, exposure_time)
    
    # Add stochastic effects
    if include_noise:
        dose = model.add_photon_shot_noise(dose, energy_kev)
        dose = model.add_resist_blur(dose, x)
    
    # Development
    developed = model.development_model(dose, development_threshold=1.0)
    
    # Calculate metrics
    cd = model.calculate_cd(x, developed)
    ler = model.calculate_ler(x, developed) if include_noise else 0.0
    
    metrics = {
        'cd_um': cd,
        'ler_nm': ler,
        'contrast': np.max(developed) - np.min(developed),
        'dose_min': np.min(dose),
        'dose_max': np.max(dose),
    }
    
    return dose, developed, metrics

# === NEW FUNCTIONS ADDED FOR COMPREHENSIVE ANALYSIS (Fixing Import Error) ===
def dose_sweep_study(mask_intensity: np.ndarray,
                     x: np.ndarray,
                     resist: ResistProperties,
                     energy_kev: float,
                     dose_factors: np.ndarray) -> dict:
    """
    Sweep exposure dose and calculate CD/LER.
    """
    results = {
        'dose_factor': dose_factors,
        'cd_um': np.zeros_like(dose_factors),
        'ler_nm': np.zeros_like(dose_factors),
    }

    print(f"Dose Sweep for {resist.name}:")

    for i, factor in enumerate(dose_factors):
        _, _, metrics = simulate_full_exposure(
            mask_intensity, x, resist, energy_kev, factor, include_noise=True
        )
        results['cd_um'][i] = metrics['cd_um'] if not np.isnan(metrics['cd_um']) else np.nan
        results['ler_nm'][i] = metrics['ler_nm'] if not np.isnan(metrics['ler_nm']) else np.nan
        
        if i % (len(dose_factors) // 5 + 1) == 0:
            print(f"  Factor {factor:.2f}: CD={results['cd_um'][i]:.3f} μm, LER={results['ler_nm'][i]:.2f} nm")

    return results

def resist_comparison(mask_intensity: np.ndarray,
                      x: np.ndarray,
                      energy_kev: float = 1.5,
                      dose_factor: float = 1.0) -> dict:
    """
    Compare all resists at a fixed aerial image and dose factor.
    """
    results = {}
    
    print(f"Resist Comparison (Energy: {energy_kev} keV, Dose Factor: {dose_factor}):")
    print("-" * 50)
    print(f"{'Resist':<10} {'CD (μm)':<10} {'LER (nm)':<10}")
    print("-" * 50)
    
    for name, resist in RESISTS.items():
        _, developed, metrics = simulate_full_exposure(
            mask_intensity, x, resist, energy_kev, dose_factor, include_noise=True
        )
        results[name] = {
            'metrics': metrics,
            'developed_profile': developed
        }
        
        cd_str = f"{metrics['cd_um']:.3f}" if not np.isnan(metrics['cd_um']) else "N/A"
        ler_str = f"{metrics['ler_nm']:.2f}" if not np.isnan(metrics['ler_nm']) else "N/A"
        print(f"{name:<10} {cd_str:<10} {ler_str:<10}")

    return results
# ==============================================================================


# Test the fix
if __name__ == "__main__":
    # ... (original test code) ...
    print("Testing Fixed Resist Response Simulation")
    print("=" * 60)
    
    # Generate test aerial image
    x = np.linspace(-2, 2, 1000)
    pitch = 1.0
    feature_size = 0.5
    
    # Create intensity profile (high = open, low = absorber)
    x_mod = np.mod(x + pitch/2, pitch)
    intensity = np.where(x_mod < feature_size, 0.2, 1.0)
    
    # Add diffraction blur
    dx = x[1] - x[0]
    sigma_points = 0.05 / dx
    intensity = gaussian_filter1d(intensity, sigma=sigma_points)
    
    # Normalize
    intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())
    intensity = 0.1 + 0.9 * intensity
    
    print(f"\nAerial image contrast: {(intensity.max() - intensity.min())/(intensity.max() + intensity.min()):.3f}")
    
    # Test all resists
    for resist_name, resist in RESISTS.items():
        dose, developed, metrics = simulate_full_exposure(
            intensity, x, resist, energy_kev=1.5, dose_factor=1.2, include_noise=True
        )
        
        print(f"\n{resist_name}:")
        print(f"  CD: {metrics['cd_um']:.3f} μm" if not np.isnan(metrics['cd_um']) else "  CD: calculation failed")
        print(f"  LER: {metrics['ler_nm']:.2f} nm" if not np.isnan(metrics['ler_nm']) else "  LER: calculation failed")
        print(f"  Contrast: {metrics['contrast']:.3f}")
        print(f"  Dose range: {metrics['dose_min']:.1f} - {metrics['dose_max']:.1f} mJ/cm²")