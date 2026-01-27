"""
Integrated X-ray Lithography Simulation Suite
=============================================

Main runner script that executes all simulation modules and generates
comprehensive analysis for Track B objectives.

Author: Abhineet Agarwal
Course: ME6110
Date: November 2025
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import simulation modules
from aerial_image import (
    XRayMask, AerialImageSimulator, MATERIALS,
    parameter_sweep_energy, parameter_sweep_gap, parameter_sweep_absorber_thickness
)
from resist_response import (
    ResistExposureModel, RESISTS, simulate_full_exposure,
    dose_sweep_study, resist_comparison
)
from thermal_mechanical import (
    MembraneMechanics, ThermalAnalysis, MEMBRANE_MATERIALS,
    exposure_scenario_analysis, material_comparison
)


class IntegratedXRLSimulation:
    """
    Comprehensive XRL simulation combining all physics modules.
    """
    
    def __init__(self, output_dir: str = 'data'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default parameters
        self.energy_kev = 1.5
        self.gap_um = 10.0
        self.feature_size_um = 0.5
        self.pitch_um = 1.0
        
        # Create default mask
        self.mask = XRayMask(
            absorber_material='Ta',
            absorber_thickness=0.5,
            membrane_material='Si3N4',
            membrane_thickness=2.0,
            feature_size=self.feature_size_um,
            pitch=self.pitch_um
        )
        
        self.results = {}
    
    def run_aerial_image_analysis(self):
        """Execute aerial image simulations"""
        print("\n" + "="*70)
        print("AERIAL IMAGE ANALYSIS")
        print("="*70)
        
        # Energy sweep
        print("\n1. Energy Sweep (0.5 - 5.0 keV)")
        energies = np.linspace(0.5, 5.0, 20)
        energy_results = parameter_sweep_energy(self.mask, self.gap_um, energies)
        self.results['energy_sweep'] = energy_results
        
        # Gap sweep
        print("\n2. Gap Sweep (1 - 50 μm)")
        gaps = np.linspace(1, 50, 20)
        gap_results = parameter_sweep_gap(self.mask, self.energy_kev, gaps)
        self.results['gap_sweep'] = gap_results
        
        # Absorber thickness sweep
        print("\n3. Absorber Thickness Sweep (0.1 - 1.0 μm)")
        thicknesses = np.linspace(0.1, 1.0, 20)
        thickness_results = parameter_sweep_absorber_thickness(
            self.mask, self.energy_kev, self.gap_um, thicknesses
        )
        self.results['thickness_sweep'] = thickness_results
        
        # Generate plots
        self._plot_aerial_image_results()
        
        return self.results
    
    def run_resist_response_analysis(self):
        """Execute resist exposure simulations"""
        print("\n" + "="*70)
        print("RESIST RESPONSE ANALYSIS")
        print("="*70)
        
        # Generate aerial image for resist simulation
        sim = AerialImageSimulator(self.mask, self.gap_um)
        x, intensity = sim.compute_aerial_image(self.energy_kev)
        
        # Resist comparison
        print("\n1. Resist Material Comparison")
        resist_results = resist_comparison(intensity, x, self.energy_kev, dose_factor=1.0)
        self.results['resist_comparison'] = resist_results
        
        # Dose sweep for PMMA
        print("\n2. Dose Sweep Study (PMMA)")
        dose_factors = np.linspace(0.5, 2.0, 15)
        dose_results = dose_sweep_study(
            intensity, x, RESISTS['PMMA'], self.energy_kev, dose_factors
        )
        self.results['dose_sweep'] = dose_results
        
        # Generate plots
        self._plot_resist_response_results()
        
        return self.results
    
    def run_thermal_mechanical_analysis(self):
        """Execute thermal-mechanical simulations"""
        print("\n" + "="*70)
        print("THERMAL-MECHANICAL ANALYSIS")
        print("="*70)
        
        # Create membrane
        mat = MEMBRANE_MATERIALS['Si3N4']
        membrane = MembraneMechanics(mat, thickness=2.0, size=50.0, geometry='square')
        
        # Beam power sweep
        print("\n1. Beam Power Sweep")
        beam_powers = np.logspace(-3, 0, 25)
        thermal_results = exposure_scenario_analysis(membrane, beam_powers)
        self.results['thermal_sweep'] = thermal_results
        
        # Material comparison
        print("\n2. Material Comparison")
        material_results = material_comparison(
            thickness=2.0, size=50.0, beam_power=0.1
        )
        self.results['material_comparison'] = material_results
        
        # Generate plots
        self._plot_thermal_mechanical_results()
        
        return self.results
    
    def _plot_aerial_image_results(self):
        """Generate comprehensive plots for aerial image analysis"""
        fig = plt.figure(figsize=(15, 10))
        
        # Energy sweep - Contrast
        ax1 = plt.subplot(2, 3, 1)
        energy_data = self.results['energy_sweep']
        ax1.plot(energy_data['energy_kev'], energy_data['contrast'], 'b-o', linewidth=2)
        ax1.set_xlabel('Photon Energy (keV)', fontsize=11)
        ax1.set_ylabel('Contrast', fontsize=11)
        ax1.set_title('Contrast vs Energy', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Energy sweep - Resolution
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(energy_data['energy_kev'], energy_data['resolution'], 'r-o', linewidth=2)
        ax2.set_xlabel('Photon Energy (keV)', fontsize=11)
        ax2.set_ylabel('Resolution (μm)', fontsize=11)
        ax2.set_title('Resolution vs Energy', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Gap sweep - Contrast
        ax3 = plt.subplot(2, 3, 3)
        gap_data = self.results['gap_sweep']
        ax3.plot(gap_data['gap_um'], gap_data['contrast'], 'g-o', linewidth=2)
        ax3.set_xlabel('Gap (μm)', fontsize=11)
        ax3.set_ylabel('Contrast', fontsize=11)
        ax3.set_title('Contrast vs Gap', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Gap sweep - Resolution
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(gap_data['gap_um'], gap_data['resolution'], 'm-o', linewidth=2)
        ax4.set_xlabel('Gap (μm)', fontsize=11)
        ax4.set_ylabel('Resolution (μm)', fontsize=11)
        ax4.set_title('Resolution vs Gap', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Thickness sweep - Contrast
        ax5 = plt.subplot(2, 3, 5)
        thick_data = self.results['thickness_sweep']
        ax5.plot(thick_data['thickness_um'], thick_data['contrast'], 'c-o', linewidth=2)
        ax5.set_xlabel('Absorber Thickness (μm)', fontsize=11)
        ax5.set_ylabel('Contrast', fontsize=11)
        ax5.set_title('Contrast vs Absorber Thickness', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Thickness sweep - Transmission
        ax6 = plt.subplot(2, 3, 6)
        ax6.semilogy(thick_data['thickness_um'], thick_data['transmission_absorber'], 
                     'k-o', linewidth=2)
        ax6.set_xlabel('Absorber Thickness (μm)', fontsize=11)
        ax6.set_ylabel('Transmission', fontsize=11)
        ax6.set_title('Absorber Transmission', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'aerial_image_comprehensive.png', dpi=300)
        print(f"\nSaved: {self.output_dir / 'aerial_image_comprehensive.png'}")
    
    def _plot_resist_response_results(self):
        """Generate comprehensive plots for resist response"""
        fig = plt.figure(figsize=(12, 8))
        
        # Resist comparison - CD
        ax1 = plt.subplot(2, 2, 1)
        resist_names = list(self.results['resist_comparison'].keys())
        cd_values = [self.results['resist_comparison'][r]['metrics']['cd_um'] 
                     for r in resist_names]
        ax1.bar(resist_names, cd_values, color=['blue', 'red', 'green', 'orange'], alpha=0.7)
        ax1.set_ylabel('CD (μm)', fontsize=11)
        ax1.set_title('Critical Dimension by Resist', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Resist comparison - LER
        ax2 = plt.subplot(2, 2, 2)
        ler_values = [self.results['resist_comparison'][r]['metrics']['ler_nm'] 
                      for r in resist_names]
        ax2.bar(resist_names, ler_values, color=['blue', 'red', 'green', 'orange'], alpha=0.7)
        ax2.set_ylabel('LER (3σ, nm)', fontsize=11)
        ax2.set_title('Line-Edge Roughness by Resist', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Dose sweep - CD
        ax3 = plt.subplot(2, 2, 3)
        dose_data = self.results['dose_sweep']
        ax3.plot(dose_data['dose_factor'], dose_data['cd_um'], 'b-o', linewidth=2)
        ax3.axhline(y=self.feature_size_um, color='r', linestyle='--', 
                    label=f'Target: {self.feature_size_um} μm', alpha=0.7)
        ax3.set_xlabel('Dose Factor (× D₀)', fontsize=11)
        ax3.set_ylabel('CD (μm)', fontsize=11)
        ax3.set_title('CD vs Exposure Dose (PMMA)', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Dose sweep - LER
        ax4 = plt.subplot(2, 2, 4)
        ax4.plot(dose_data['dose_factor'], dose_data['ler_nm'], 'r-o', linewidth=2)
        ax4.set_xlabel('Dose Factor (× D₀)', fontsize=11)
        ax4.set_ylabel('LER (3σ, nm)', fontsize=11)
        ax4.set_title('LER vs Exposure Dose (PMMA)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'resist_response_comprehensive.png', dpi=300)
        print(f"\nSaved: {self.output_dir / 'resist_response_comprehensive.png'}")
    
    def _plot_thermal_mechanical_results(self):
        """Generate comprehensive plots for thermal-mechanical analysis"""
        fig = plt.figure(figsize=(14, 10))
        
        thermal_data = self.results['thermal_sweep']
        
        # Temperature rise
        ax1 = plt.subplot(2, 2, 1)
        ax1.loglog(thermal_data['beam_power_W'], thermal_data['temperature_rise_K'], 
                   'b-o', linewidth=2)
        ax1.set_xlabel('Beam Power (W)', fontsize=11)
        ax1.set_ylabel('Temperature Rise (K)', fontsize=11)
        ax1.set_title('Temperature Rise vs Beam Power', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Thermal stress
        ax2 = plt.subplot(2, 2, 2)
        ax2.loglog(thermal_data['beam_power_W'], thermal_data['thermal_stress_MPa'], 
                   'r-o', linewidth=2)
        ax2.set_xlabel('Beam Power (W)', fontsize=11)
        ax2.set_ylabel('Thermal Stress (MPa)', fontsize=11)
        ax2.set_title('Thermal Stress vs Beam Power', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Deflection
        ax3 = plt.subplot(2, 2, 3)
        ax3.loglog(thermal_data['beam_power_W'], thermal_data['deflection_um'], 
                   'g-o', linewidth=2)
        ax3.axhline(y=0.1, color='r', linestyle='--', label='0.1 μm tolerance', alpha=0.7)
        ax3.set_xlabel('Beam Power (W)', fontsize=11)
        ax3.set_ylabel('Deflection (μm)', fontsize=11)
        ax3.set_title('Membrane Deflection vs Beam Power', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Material comparison
        ax4 = plt.subplot(2, 2, 4)
        mat_data = self.results['material_comparison']
        materials = list(mat_data.keys())
        deflections = [mat_data[m]['deflection'] for m in materials]
        colors = ['blue', 'red', 'green', 'orange'][:len(materials)]
        ax4.bar(materials, deflections, color=colors, alpha=0.7)
        ax4.set_ylabel('Deflection (μm)', fontsize=11)
        ax4.set_title('Membrane Material Comparison (0.1 W)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'thermal_mechanical_comprehensive.png', dpi=300)
        print(f"\nSaved: {self.output_dir / 'thermal_mechanical_comprehensive.png'}")
    
    def generate_summary_report(self):
        """Generate text summary of all results"""
        report_path = self.output_dir / 'simulation_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("X-RAY LITHOGRAPHY SIMULATION SUMMARY\n")
            f.write("Track B: Modeling and Simulation\n")
            f.write("="*70 + "\n\n")
            
            f.write("SIMULATION PARAMETERS:\n")
            f.write(f"  Photon Energy: {self.energy_kev} keV\n")
            f.write(f"  Mask-Resist Gap: {self.gap_um} μm\n")
            f.write(f"  Feature Size: {self.feature_size_um} μm\n")
            f.write(f"  Pitch: {self.pitch_um} μm\n")
            f.write(f"  Absorber: Ta, {self.mask.absorber_thickness} μm\n")
            f.write(f"  Membrane: Si3N4, {self.mask.membrane_thickness} μm\n\n")
            
            f.write("-"*70 + "\n")
            f.write("KEY FINDINGS:\n")
            f.write("-"*70 + "\n\n")
            
            # Aerial image findings
            energy_data = self.results['energy_sweep']
            optimal_idx = np.argmax(energy_data['contrast'])
            f.write("1. AERIAL IMAGE ANALYSIS:\n")
            f.write(f"   Optimal Energy: {energy_data['energy_kev'][optimal_idx]:.2f} keV\n")
            f.write(f"   Maximum Contrast: {energy_data['contrast'][optimal_idx]:.3f}\n")
            f.write(f"   Resolution at Optimal: {energy_data['resolution'][optimal_idx]:.3f} μm\n\n")
            
            gap_data = self.results['gap_sweep']
            f.write(f"   Gap Effect: Contrast degrades from {gap_data['contrast'][0]:.3f} ")
            f.write(f"to {gap_data['contrast'][-1]:.3f} over 1-50 μm\n\n")
            
            # Resist findings
            f.write("2. RESIST RESPONSE ANALYSIS:\n")
            resist_comp = self.results['resist_comparison']
            for resist_name, data in resist_comp.items():
                cd = data['metrics']['cd_um']
                ler = data['metrics']['ler_nm']
                f.write(f"   {resist_name:12s}: CD = {cd:.3f} μm, LER = {ler:.2f} nm\n")
            f.write("\n")
            
            # Thermal findings
            f.write("3. THERMAL-MECHANICAL ANALYSIS:\n")
            mat_comp = self.results['material_comparison']
            for mat_name, data in mat_comp.items():
                f.write(f"   {mat_name:12s}: ΔT = {data['delta_T']:.2f} K, ")
                f.write(f"Deflection = {data['deflection']:.3f} μm\n")
            f.write("\n")
            
            f.write("-"*70 + "\n")
            f.write("RECOMMENDATIONS:\n")
            f.write("-"*70 + "\n")
            f.write("• Use 1.0-2.0 keV X-rays for optimal contrast/resolution balance\n")
            f.write("• Maintain mask-resist gap < 20 μm for acceptable proximity effects\n")
            f.write("• ZEP520A or HSQ recommended for sub-100 nm features (low LER)\n")
            f.write("• SiC or Diamond membranes preferred for high-power applications\n")
            f.write("• Thermal management critical for beam powers > 0.1 W\n\n")
            
            f.write("="*70 + "\n")
            f.write("End of Report\n")
            f.write("="*70 + "\n")
        
        print(f"\nSummary report saved: {report_path}")
        return report_path
    
    def run_all(self):
        """Execute complete simulation suite"""
        print("\n" + "="*70)
        print("INTEGRATED X-RAY LITHOGRAPHY SIMULATION SUITE")
        print("="*70)
        
        self.run_aerial_image_analysis()
        self.run_resist_response_analysis()
        self.run_thermal_mechanical_analysis()
        self.generate_summary_report()
        
        print("\n" + "="*70)
        print("SIMULATION COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print("="*70 + "\n")


if __name__ == "__main__":
    # Run integrated simulation
    sim = IntegratedXRLSimulation()
    sim.run_all()
