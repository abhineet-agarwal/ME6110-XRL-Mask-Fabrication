#!/usr/bin/env python3
"""
Tutorial 5: End-to-End Simulation Pipeline
============================================

Runs the complete XRL simulation workflow:
1. Configure simulation parameters
2. Compute aerial image through the mask
3. Expose and develop resist
4. Evaluate thermal-mechanical behaviour
5. Print a summary report

Demonstrates using ``SimulationConfig`` to drive all modules.

Run::

    python examples/05_full_simulation.py
"""

import numpy as np
from xrl import (
    SimulationConfig,
    default_config,
    XRayMask,
    AerialImageSimulator,
    RESISTS,
    MEMBRANES,
    MembraneMechanics,
    ThermalAnalysis,
)
from xrl.resist import simulate_full_exposure
from xrl.thermal import exposure_scenario_analysis
from xrl.plotting import plot_aerial_image, plot_resist_profile
import matplotlib.pyplot as plt


def main():
    # -- 1. Configuration --------------------------------------------------
    cfg = default_config()
    cfg.energy_kev = 1.5
    cfg.gap_um = 10.0
    cfg.absorber_material = 'Ta'
    cfg.absorber_thickness_um = 0.5
    cfg.resist = 'PMMA'
    cfg.dose_factor = 1.2
    cfg.beam_power_W = 0.1

    print("=" * 60)
    print("X-ray Lithography Full Simulation")
    print("=" * 60)
    print(f"Energy:    {cfg.energy_kev} keV")
    print(f"Gap:       {cfg.gap_um} um")
    print(f"Absorber:  {cfg.absorber_material} ({cfg.absorber_thickness_um} um)")
    print(f"Membrane:  {cfg.membrane_material} ({cfg.membrane_thickness_um} um)")
    print(f"Resist:    {cfg.resist} (dose factor {cfg.dose_factor})")
    print(f"Beam power:{cfg.beam_power_W} W")
    print()

    # -- 2. Aerial image ---------------------------------------------------
    mask = XRayMask(
        absorber_material=cfg.absorber_material,
        absorber_thickness=cfg.absorber_thickness_um,
        membrane_material=cfg.membrane_material,
        membrane_thickness=cfg.membrane_thickness_um,
        feature_size=cfg.feature_size_um,
        pitch=cfg.pitch_um,
    )
    sim = AerialImageSimulator(mask, gap=cfg.gap_um)
    x, intensity = sim.compute_aerial_image(
        energy_kev=cfg.energy_kev,
        x_range=cfg.x_range_um,
        resolution=cfg.resolution,
    )
    contrast = sim.calculate_contrast(x, intensity)
    resolution = sim.calculate_resolution(x, intensity)
    print(f"Aerial image contrast: {contrast:.3f}")
    print(f"Aerial image FWHM:     {resolution:.3f} um")

    # -- 3. Resist exposure ------------------------------------------------
    resist = RESISTS[cfg.resist]
    dose, developed, metrics = simulate_full_exposure(
        intensity, x, resist,
        energy_kev=cfg.energy_kev,
        dose_factor=cfg.dose_factor,
        include_noise=cfg.include_noise,
        n_samples_ler=cfg.n_samples_ler,
    )
    cd_str = f"{metrics['cd_um']:.3f}" if not np.isnan(metrics['cd_um']) else "N/A"
    ler_str = f"{metrics['ler_nm']:.1f}" if not np.isnan(metrics['ler_nm']) else "N/A"
    print(f"CD:  {cd_str} um")
    print(f"LER: {ler_str} nm (3-sigma)")
    print(f"Dose range: {metrics['dose_min']:.1f} -- {metrics['dose_max']:.1f} mJ/cm^2")

    # -- 4. Thermal analysis -----------------------------------------------
    mem_mat = MEMBRANES[cfg.membrane_material]
    membrane = MembraneMechanics(
        mem_mat, cfg.membrane_thickness_um, cfg.membrane_size_mm,
        geometry=cfg.membrane_geometry,
    )
    thermal = ThermalAnalysis(membrane)
    absorbed = thermal.absorbed_power(cfg.beam_power_W)
    dT = thermal.steady_state_center_temp_rise(absorbed)
    tau = thermal.thermal_time_constant()
    print(f"\nThermal analysis ({cfg.membrane_material} membrane):")
    print(f"  Temperature rise: {dT:.2f} K")
    print(f"  Time constant:    {tau:.4f} s")

    # -- 5. Summary plots --------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    plot_aerial_image(x, intensity,
                      title=f'Aerial Image ({cfg.energy_kev} keV, {cfg.gap_um} um gap)',
                      fig=fig, ax=ax1)

    plot_resist_profile(x, developed, resist_name=cfg.resist,
                        fig=fig, ax=ax2)

    fig.tight_layout()
    plt.show()

    # -- 6. Save config for reproducibility --------------------------------
    cfg.save('last_simulation.json')
    print("\nConfig saved to last_simulation.json")


if __name__ == '__main__':
    main()
