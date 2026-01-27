"""
Analysis Utilities for Cross-Variable Sweeps
===========================================

New functions for multi-parameter comparison beyond simple 1D sweeps.
"""

import numpy as np
from aerial_image import AerialImageSimulator, XRayMask, MATERIALS
from thermal_mechanical import MembraneMechanics, ThermalAnalysis, MEMBRANE_MATERIALS

def sweep_gap_energy_matrix(mask: XRayMask, 
                            gaps: np.ndarray, 
                            energies: np.ndarray) -> dict:
    """
    Computes a 2D matrix of contrast vs. mask-resist gap and photon energy.
    """
    contrast_matrix = np.zeros((len(gaps), len(energies)))
    print("\nStarting 2D Gap vs. Energy Contrast Sweep...")
    
    for i, gap in enumerate(gaps):
        for j, energy in enumerate(energies):
            sim = AerialImageSimulator(mask, gap)
            x, intensity = sim.compute_aerial_image(energy)
            contrast_matrix[i, j] = sim.calculate_contrast(x, intensity)
        print(f"  Finished row for Gap: {gap:.1f} μm")
        
    results = {
        'gaps_um': gaps,
        'energies_kev': energies,
        'contrast_matrix': contrast_matrix
    }
    return results

def sweep_thermal_material_vs_power(thickness: float, 
                                    size: float, 
                                    beam_powers: np.ndarray) -> dict:
    """
    Sweeps all membrane materials across the full beam power range.
    """
    all_results = {}
    
    print("\nStarting Thermal Sweep: All Materials vs. Beam Power...")
    
    for mat_name, mat_props in MEMBRANE_MATERIALS.items():
        print(f"  Simulating {mat_name}...")
        membrane = MembraneMechanics(mat_props, thickness, size, 'square')
        thermal = ThermalAnalysis(membrane)
        
        results = {
            'beam_power_W': beam_powers,
            'deflection_um': np.zeros_like(beam_powers),
            'thermal_stress_MPa': np.zeros_like(beam_powers),
        }
        
        for i, power in enumerate(beam_powers):
            absorbed = thermal.absorbed_power(power, absorption_fraction=0.1)
            
            # Use in-plane gradient for driving forces
            delta_T_in_plane = thermal.steady_state_in_plane_gradient(absorbed) 
            delta_T_through_thickness = delta_T_in_plane / 10 
            
            results['thermal_stress_MPa'][i] = membrane.thermal_stress(delta_T_in_plane)
            results['deflection_um'][i] = membrane.thermal_deflection(delta_T_through_thickness)
        
        all_results[mat_name] = results
        
    return all_results