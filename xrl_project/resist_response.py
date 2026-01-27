"""
Resist Response Simulation for X-ray Lithography
================================================

Models photon absorption, energy deposition, and resist exposure including
stochastic effects (photon shot noise, resist blur). Calculates CD and LER.

Author: Abhineet Agarwal
Course: ME6110
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import poisson
from dataclasses import dataclass
from typing import Tuple, Optional


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
        # Empirical values for PMMA-like organic resists
        # Based on NIST data for carbon-rich polymers
        # Energy in keV, returns μ in 1/μm
        
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
        # Absorption coefficient
        mu = self.absorption_coefficient(energy_kev)
        
        # Absorbed fraction (assuming thin resist)
        f_absorbed = 1 - np.exp(-mu * self.resist.thickness)
        
        # Convert intensity to dose
        # For XRL, typical flux ~1e13 photons/(s·cm²) at full intensity
        # Scale with normalized intensity
        reference_flux = 1e13  # photons/(s·cm²) at intensity = 1.0
        flux = intensity * reference_flux  # photons/(s·cm²)
        energy_per_photon = energy_kev * 1.602e-16  # J
        
        # Calculate dose: flux × time × energy/photon × absorption fraction
        dose = flux * exposure_time * energy_per_photon * f_absorbed * 1e3  # mJ/cm²
        
        return dose
    
    def add_photon_shot_noise(self,
                             dose: np.ndarray,
                             energy_kev: float) -> np.ndarray:
        """
        Add Poisson shot noise from discrete photon statistics.
        """
        # Number of photons per unit area
        energy_per_photon = energy_kev * 1.602e-16  # J
        n_photons = dose * 1e-3 / energy_per_photon  # photons/cm²
        
        # Sample from Poisson distribution
        n_photons_noisy = np.random.poisson(n_photons)
        
        # Convert back to dose
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
        Realistic development model with gradual response around D0.
        
        Args:
            dose: Absorbed dose array (mJ/cm²)
            development_threshold: Threshold as fraction of D0
        
        Returns:
            Remaining resist thickness (normalized to initial thickness)
        """
        D_threshold = self.resist.sensitivity * development_threshold
        
        # Normalized dose
        D_norm = dose / D_threshold
        
        if self.resist.tone == 'positive':
            # Positive resist: exposed areas dissolve
            # Use a more realistic model that transitions smoothly
            # Below D0: mostly remains
            # Above D0: dissolves according to contrast
            
            # Model: remaining = 1 / (1 + (D/D0)^γ) for smooth transition
            gamma = self.resist.contrast
            remaining = 1.0 / (1.0 + D_norm**gamma)
            
        else:
            # Negative resist: exposed areas crosslink
            # Model: remaining = (D/D0)^γ / (1 + (D/D0)^γ)
            gamma = self.resist.contrast
            remaining = (D_norm**gamma) / (1.0 + D_norm**gamma)
        
        # Clamp to [0, 1]
        remaining = np.clip(remaining, 0, 1)
        
        return remaining
    
    def calculate_cd(self,
                    x: np.ndarray,
                    profile: np.ndarray,
                    threshold: float = 0.5) -> float:
        """
        Calculate critical dimension (linewidth) at given threshold.
        
        Returns:
            CD in μm
        """
        # Find edges where profile crosses threshold
        above = profile > threshold
        edges = np.where(np.diff(above.astype(int)))[0]
        
        if len(edges) >= 2:
            # Take first feature
            cd = x[edges[1]] - x[edges[0]]
            return abs(cd)
        else:
            return np.nan
    
    def calculate_ler(self,
                     x: np.ndarray,
                     profile: np.ndarray,
                     threshold: float = 0.5,
                     n_samples: int = 10) -> float:
        """
        Calculate line-edge roughness (3σ).
        
        Simulates multiple exposures and measures edge position variance.
        
        Returns:
            LER (3σ) in nm
        """
        edge_positions = []
        
        for _ in range(n_samples):
            # Add noise to profile
            noisy_profile = profile + np.random.normal(0, 0.02, size=profile.shape)
            
            # Find edge
            above = noisy_profile > threshold
            edges = np.where(np.diff(above.astype(int)))[0]
            
            if len(edges) >= 1:
                edge_pos = x[edges[0]]
                edge_positions.append(edge_pos)
        
        if len(edge_positions) > 0:
            ler_3sigma = 3 * np.std(edge_positions) * 1000  # Convert to nm
            return ler_3sigma
        else:
            return np.nan


def simulate_full_exposure(mask_intensity: np.ndarray,
                          x: np.ndarray,
                          resist: ResistProperties,
                          energy_kev: float = 1.5,
                          dose_factor: float = 1.0,
                          include_noise: bool = True) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Full simulation from aerial image to developed profile.
    
    Args:
        mask_intensity: Aerial image intensity (normalized 0-1)
        x: Position array (μm)
        resist: Resist properties
        energy_kev: Photon energy
        dose_factor: Dose multiplier (1.0 = nominal D0)
        include_noise: Whether to include shot noise and blur
    
    Returns:
        (dose_profile, developed_profile, metrics)
    """
    model = ResistExposureModel(resist)
    
    # Calculate exposure time needed to reach target dose
    # At reference flux (1e13 photons/s/cm²), energy 1.5 keV, 
    # and typical absorption (~50%), we get ~0.12 mJ/cm² per second
    # For PMMA (D0 = 500 mJ/cm²), we need ~4200 seconds
    # We'll scale the reference flux higher to make reasonable exposure times
    
    # Target dose is resist sensitivity × dose_factor
    target_dose = resist.sensitivity * dose_factor  # mJ/cm²
    
    # Calculate exposure time: for intensity=1, what time gives target dose?
    # dose = flux × time × energy × absorption
    # Rearrange: time = dose / (flux × energy × absorption)
    
    # Use reasonable XRL parameters
    reference_flux = 1e13  # photons/(s·cm²)
    energy_per_photon = energy_kev * 1.602e-16  # J
    mu = model.absorption_coefficient(energy_kev)
    f_absorbed = 1 - np.exp(-mu * resist.thickness)
    
    # Calculate required exposure time for peak (intensity=1.0) to reach target
    dose_per_second = reference_flux * energy_per_photon * f_absorbed * 1e3  # mJ/cm²/s
    exposure_time = target_dose / dose_per_second  # seconds
    
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


def dose_sweep_study(mask_intensity: np.ndarray,
                    x: np.ndarray,
                    resist: ResistProperties,
                    energy_kev: float = 1.5,
                    dose_factors: np.ndarray = np.linspace(0.5, 2.0, 10)) -> dict:
    """
    Sweep exposure dose and measure CD and LER.
    """
    results = {
        'dose_factor': dose_factors,
        'cd_um': np.zeros_like(dose_factors),
        'ler_nm': np.zeros_like(dose_factors),
    }
    
    print(f"\nDose sweep for {resist.name}:")
    print("-" * 50)
    
    for i, dose_factor in enumerate(dose_factors):
        _, developed, metrics = simulate_full_exposure(
            mask_intensity, x, resist, energy_kev, dose_factor, include_noise=True
        )
        
        results['cd_um'][i] = metrics['cd_um']
        results['ler_nm'][i] = metrics['ler_nm']
        
        print(f"Dose factor: {dose_factor:.2f} | CD: {metrics['cd_um']:.3f} μm | "
              f"LER: {metrics['ler_nm']:.2f} nm")
    
    return results


def resist_comparison(mask_intensity: np.ndarray,
                     x: np.ndarray,
                     energy_kev: float = 1.5,
                     dose_factor: float = 1.0) -> dict:
    """
    Compare different resist materials.
    """
    results = {}
    
    print("\nResist comparison:")
    print("=" * 60)
    
    for resist_name, resist in RESISTS.items():
        dose, developed, metrics = simulate_full_exposure(
            mask_intensity, x, resist, energy_kev, dose_factor, include_noise=True
        )
        
        results[resist_name] = {
            'dose': dose,
            'developed': developed,
            'metrics': metrics
        }
        
        print(f"{resist_name:12s} | CD: {metrics['cd_um']:.3f} μm | "
              f"LER: {metrics['ler_nm']:.2f} nm | Contrast: {metrics['contrast']:.3f}")
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("X-ray Resist Response Simulation")
    print("=" * 60)
    
    # Generate sample aerial image with proper intensity
    x = np.linspace(-2, 2, 1000)
    pitch = 1.0
    feature_size = 0.5
    
    # Create a realistic intensity profile
    # Bright field (transmission through open areas) = 1.0
    # Dark field (transmission through absorber) = 0.2
    x_mod = np.mod(x + pitch/2, pitch)
    intensity = np.where(x_mod < feature_size, 0.2, 1.0)
    
    # Apply realistic blur from diffraction (sigma ~ 0.05 μm)
    dx = x[1] - x[0]
    sigma_points = 0.05 / dx
    intensity = gaussian_filter1d(intensity, sigma=sigma_points)
    
    # Normalize to ensure we have contrast
    intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())
    # Scale to realistic range: 0.1 (dark) to 1.0 (bright)
    intensity = 0.1 + 0.9 * intensity
    
    print(f"\nGenerated aerial image:")
    print(f"  Intensity range: {intensity.min():.3f} to {intensity.max():.3f}")
    print(f"  Contrast: {(intensity.max() - intensity.min())/(intensity.max() + intensity.min()):.3f}")
    
    # Simulate PMMA resist
    resist = RESISTS['PMMA']
    dose, developed, metrics = simulate_full_exposure(
        intensity, x, resist, energy_kev=1.5, dose_factor=1.0, include_noise=True
    )
    
    print(f"\nSimulation results for {resist.name}:")
    print(f"  CD: {metrics['cd_um']:.3f} μm")
    print(f"  LER (3σ): {metrics['ler_nm']:.2f} nm")
    print(f"  Contrast: {metrics['contrast']:.3f}")
    print(f"  Dose range: {metrics['dose_min']:.1f} - {metrics['dose_max']:.1f} mJ/cm²")
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    # Aerial image
    axes[0].plot(x, intensity, 'b-', linewidth=2)
    axes[0].set_ylabel('Normalized Intensity', fontsize=11)
    axes[0].set_title('Aerial Image at Resist', fontsize=12, fontweight='bold')
    axes[0].set_ylim([0, 1.1])
    axes[0].grid(True, alpha=0.3)
    
    # Dose profile
    axes[1].plot(x, dose, 'r-', linewidth=2)
    axes[1].axhline(y=resist.sensitivity, color='k', linestyle='--', 
                    alpha=0.5, label=f'D₀ = {resist.sensitivity} mJ/cm²')
    axes[1].set_ylabel('Absorbed Dose (mJ/cm²)', fontsize=11)
    axes[1].set_title('Dose Profile in Resist', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Developed profile
    axes[2].plot(x, developed, 'g-', linewidth=2)
    axes[2].axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='50% threshold')
    axes[2].set_xlabel('Position (μm)', fontsize=11)
    axes[2].set_ylabel('Remaining Thickness (norm.)', fontsize=11)
    axes[2].set_title('Developed Profile', fontsize=12, fontweight='bold')
    axes[2].set_ylim([0, 1.1])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create data directory if it doesn't exist
    import os
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    output_path = os.path.join(data_dir, 'resist_response_example.png')
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved to: {output_path}")

