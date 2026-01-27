"""
Aerial Image Modeling for X-ray Lithography
============================================

This module computes intensity profiles through X-ray mask stacks using
Beer-Lambert absorption law. Supports parametric sweeps over photon energy,
mask-resist gap, and absorber thickness.

Author: Abhineet Agarwal
Course: ME6110
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import fresnel
from dataclasses import dataclass
from typing import Tuple, Optional
import pickle


@dataclass
class MaterialProperties:
    """Material properties for X-ray absorption"""
    name: str
    density: float  # g/cm³
    atomic_number: float
    atomic_mass: float  # g/mol
    
    def get_attenuation_coefficient(self, energy_kev: float) -> float:
        """
        Calculate mass attenuation coefficient using lookup-based approximation.
        
        Based on NIST XCOM database for X-rays in 0.5-5 keV range.
        
        Returns: μ (1/cm)
        """
        # Use empirical fits for common materials in XRL energy range
        # These are approximate fits to NIST data
        
        if self.name == 'Tantalum':
            # Ta: Strong absorption in XRL range
            if energy_kev < 1.0:
                mu_over_rho = 3000 / energy_kev**2.8
            elif energy_kev < 2.0:
                mu_over_rho = 1500 / energy_kev**2.6
            else:
                mu_over_rho = 800 / energy_kev**2.4
                
        elif self.name == 'Tungsten':
            # W: Similar to Ta
            if energy_kev < 1.0:
                mu_over_rho = 2800 / energy_kev**2.8
            else:
                mu_over_rho = 1200 / energy_kev**2.5
                
        elif self.name == 'Gold':
            # Au: High Z absorber
            if energy_kev < 1.0:
                mu_over_rho = 2500 / energy_kev**2.7
            else:
                mu_over_rho = 1000 / energy_kev**2.4
                
        elif 'Nitride' in self.name or 'Carbide' in self.name:
            # Low-Z membrane materials
            mu_over_rho = 20 / energy_kev**2.5
            
        else:
            # Generic organic (resist)
            mu_over_rho = 10 / energy_kev**2.6
        
        return mu_over_rho * self.density


# Material database
MATERIALS = {
    'Ta': MaterialProperties('Tantalum', 16.6, 73, 180.9),
    'W': MaterialProperties('Tungsten', 19.3, 74, 183.8),
    'Au': MaterialProperties('Gold', 19.3, 79, 197.0),
    'Si3N4': MaterialProperties('Silicon Nitride', 3.44, 11.2, 140.3),  # effective Z, A
    'SiC': MaterialProperties('Silicon Carbide', 3.21, 10, 40.1),
    'PMMA': MaterialProperties('PMMA', 1.18, 3.6, 100.1),
}


class XRayMask:
    """X-ray mask geometry and properties"""
    
    def __init__(self, 
                 absorber_material: str = 'Ta',
                 absorber_thickness: float = 0.5,  # μm
                 membrane_material: str = 'Si3N4',
                 membrane_thickness: float = 2.0,  # μm
                 feature_size: float = 0.5,  # μm
                 pitch: float = 1.0):  # μm
        
        self.absorber = MATERIALS[absorber_material]
        self.absorber_thickness = absorber_thickness
        self.membrane = MATERIALS[membrane_material]
        self.membrane_thickness = membrane_thickness
        self.feature_size = feature_size
        self.pitch = pitch
    
    def get_transmission_profile(self, 
                                 x_positions: np.ndarray,
                                 energy_kev: float) -> np.ndarray:
        """
        Calculate transmission through mask at given positions.
        
        Returns: Transmission coefficient (0-1)
        """
        # Attenuation coefficients
        mu_abs = self.absorber.get_attenuation_coefficient(energy_kev)
        mu_mem = self.membrane.get_attenuation_coefficient(energy_kev)
        
        # Convert to μm^-1
        mu_abs *= 1e-4
        mu_mem *= 1e-4
        
        # Create periodic pattern
        x_mod = np.mod(x_positions, self.pitch)
        
        # Membrane transmission (everywhere)
        t_membrane = np.exp(-mu_mem * self.membrane_thickness)
        
        # Absorber transmission (only where features are)
        t_absorber = np.exp(-mu_abs * self.absorber_thickness)
        
        # Combined transmission
        transmission = np.where(
            x_mod < self.feature_size,
            t_membrane * t_absorber,  # Through absorber
            t_membrane  # Open area
        )
        
        return transmission


class AerialImageSimulator:
    """
    Simulates aerial image formation accounting for diffraction and proximity effects
    """
    
    def __init__(self, mask: XRayMask, gap: float = 10.0):
        """
        Args:
            mask: XRayMask object
            gap: Mask-to-resist gap in μm
        """
        self.mask = mask
        self.gap = gap
    
    def fresnel_propagation(self,
                           field_at_mask: np.ndarray,
                           x_mask: np.ndarray,
                           x_resist: np.ndarray,
                           wavelength: float) -> np.ndarray:
        """
        Propagate field from mask to resist using Fresnel diffraction.
        
        Args:
            field_at_mask: Complex field amplitude at mask
            x_mask: Position array at mask (μm)
            x_resist: Position array at resist (μm)
            wavelength: X-ray wavelength (μm)
        
        Returns:
            Intensity at resist plane
        """
        dx_mask = x_mask[1] - x_mask[0]
        
        # Fresnel number
        F = self.mask.feature_size**2 / (wavelength * self.gap)
        
        # For very small Fresnel numbers (sharp shadows), use geometric projection
        if F < 0.1:
            return np.abs(field_at_mask)**2
        
        # Fresnel propagation kernel
        field_at_resist = np.zeros_like(x_resist, dtype=complex)
        
        for i, x_r in enumerate(x_resist):
            # Simplified Fresnel integral
            phase_factor = np.exp(1j * np.pi * (x_mask - x_r)**2 / (wavelength * self.gap))
            field_at_resist[i] = np.sum(field_at_mask * phase_factor) * dx_mask
        
        # Normalize
        field_at_resist *= np.exp(1j * 2 * np.pi * self.gap / wavelength) / (1j * wavelength * self.gap)
        
        return np.abs(field_at_resist)**2
    
    def compute_aerial_image(self,
                            energy_kev: float,
                            x_range: float = 3.0,
                            resolution: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the aerial image at resist plane.
        
        Args:
            energy_kev: Photon energy in keV
            x_range: Spatial range in μm
            resolution: Number of points
        
        Returns:
            (positions, intensity) tuples
        """
        # X-ray wavelength (nm)
        wavelength_nm = 1.24 / energy_kev
        wavelength_um = wavelength_nm / 1000
        
        # Position arrays
        x = np.linspace(-x_range/2, x_range/2, resolution)
        
        # Transmission at mask
        transmission = self.mask.get_transmission_profile(x, energy_kev)
        
        # Field amplitude (sqrt of transmission)
        field = np.sqrt(transmission)
        
        # Propagate to resist
        if self.gap > 0:
            intensity = self.fresnel_propagation(field, x, x, wavelength_um)
        else:
            intensity = transmission
        
        return x, intensity
    
    def calculate_contrast(self,
                          x: np.ndarray,
                          intensity: np.ndarray) -> float:
        """
        Calculate image contrast: (I_max - I_min) / (I_max + I_min)
        """
        # Find one period
        period_points = int(len(x) * self.mask.pitch / (x[-1] - x[0]))
        
        # Get max and min over central period
        center = len(x) // 2
        half_period = period_points // 2
        i_period = intensity[center-half_period:center+half_period]
        
        I_max = np.max(i_period)
        I_min = np.min(i_period)
        
        contrast = (I_max - I_min) / (I_max + I_min) if (I_max + I_min) > 0 else 0
        
        return contrast
    
    def calculate_resolution(self,
                            x: np.ndarray,
                            intensity: np.ndarray,
                            threshold: float = 0.5) -> float:
        """
        Estimate resolution as FWHM of intensity peak.
        """
        # Normalize intensity
        I_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min())
        
        # Find FWHM
        above_threshold = I_norm > threshold
        edges = np.where(np.diff(above_threshold.astype(int)))[0]
        
        if len(edges) >= 2:
            fwhm = x[edges[1]] - x[edges[0]]
            return abs(fwhm)
        else:
            return np.nan


def parameter_sweep_energy(mask: XRayMask,
                          gap: float = 10.0,
                          energies: np.ndarray = np.linspace(0.5, 5.0, 10)) -> dict:
    """
    Sweep photon energy and calculate contrast/resolution.
    
    Returns:
        Dictionary with energy, contrast, and resolution arrays
    """
    results = {
        'energy_kev': energies,
        'contrast': np.zeros_like(energies),
        'resolution': np.zeros_like(energies),
    }
    
    sim = AerialImageSimulator(mask, gap)
    
    for i, energy in enumerate(energies):
        x, intensity = sim.compute_aerial_image(energy)
        results['contrast'][i] = sim.calculate_contrast(x, intensity)
        results['resolution'][i] = sim.calculate_resolution(x, intensity)
        
        print(f"Energy: {energy:.2f} keV | Contrast: {results['contrast'][i]:.3f} | "
              f"Resolution: {results['resolution'][i]:.3f} μm")
    
    return results


def parameter_sweep_gap(mask: XRayMask,
                       energy_kev: float = 1.5,
                       gaps: np.ndarray = np.linspace(1, 50, 10)) -> dict:
    """
    Sweep mask-resist gap and calculate contrast/resolution.
    """
    results = {
        'gap_um': gaps,
        'contrast': np.zeros_like(gaps),
        'resolution': np.zeros_like(gaps),
    }
    
    for i, gap in enumerate(gaps):
        sim = AerialImageSimulator(mask, gap)
        x, intensity = sim.compute_aerial_image(energy_kev)
        results['contrast'][i] = sim.calculate_contrast(x, intensity)
        results['resolution'][i] = sim.calculate_resolution(x, intensity)
        
        print(f"Gap: {gap:.1f} μm | Contrast: {results['contrast'][i]:.3f} | "
              f"Resolution: {results['resolution'][i]:.3f} μm")
    
    return results


def parameter_sweep_absorber_thickness(mask_base: XRayMask,
                                      energy_kev: float = 1.5,
                                      gap: float = 10.0,
                                      thicknesses: np.ndarray = np.linspace(0.1, 1.0, 10)) -> dict:
    """
    Sweep absorber thickness and calculate contrast.
    """
    results = {
        'thickness_um': thicknesses,
        'contrast': np.zeros_like(thicknesses),
        'transmission_absorber': np.zeros_like(thicknesses),
    }
    
    for i, thickness in enumerate(thicknesses):
        mask = XRayMask(
            absorber_material='Ta',
            absorber_thickness=thickness,
            membrane_material='Si3N4',
            membrane_thickness=mask_base.membrane_thickness,
            feature_size=mask_base.feature_size,
            pitch=mask_base.pitch
        )
        
        sim = AerialImageSimulator(mask, gap)
        x, intensity = sim.compute_aerial_image(energy_kev)
        results['contrast'][i] = sim.calculate_contrast(x, intensity)
        
        # Calculate transmission through absorber
        mu = mask.absorber.get_attenuation_coefficient(energy_kev) * 1e-4
        results['transmission_absorber'][i] = np.exp(-mu * thickness)
        
        print(f"Thickness: {thickness:.2f} μm | Contrast: {results['contrast'][i]:.3f} | "
              f"Transmission: {results['transmission_absorber'][i]:.4f}")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("X-ray Lithography Aerial Image Simulation")
    print("=" * 60)
    
    # Create a standard mask
    mask = XRayMask(
        absorber_material='Ta',
        absorber_thickness=0.5,
        membrane_material='Si3N4',
        membrane_thickness=2.0,
        feature_size=0.5,
        pitch=1.0
    )
    
    # Simulate at 1.5 keV with 10 μm gap
    print("\nSimulating aerial image at 1.5 keV, 10 μm gap...")
    sim = AerialImageSimulator(mask, gap=10.0)
    x, intensity = sim.compute_aerial_image(energy_kev=1.5)
    
    contrast = sim.calculate_contrast(x, intensity)
    resolution = sim.calculate_resolution(x, intensity)
    
    print(f"Contrast: {contrast:.3f}")
    print(f"Resolution (FWHM): {resolution:.3f} μm")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, intensity / intensity.max(), 'b-', linewidth=2)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% threshold')
    plt.xlabel('Position (μm)', fontsize=12)
    plt.ylabel('Normalized Intensity', fontsize=12)
    plt.title(f'Aerial Image: {mask.feature_size} μm features, {sim.gap} μm gap', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/aerial_image_example.png', dpi=300)
    print("\nPlot saved to: data/aerial_image_example.png")
