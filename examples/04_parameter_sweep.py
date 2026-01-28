#!/usr/bin/env python3
"""
Tutorial 4: Parameter Space Exploration
=========================================

Performs 2-D sweeps over photon energy and mask-resist gap to
build a contrast heatmap, and sweeps absorber thickness to find
the optimal operating point.

Physics covered:
- Systematic design-of-experiments for lithography optimisation
- Trade-offs between contrast, resolution, and throughput
- Absorber material comparison

Run::

    python examples/04_parameter_sweep.py
"""

import numpy as np
from xrl import XRayMask, AerialImageSimulator
from xrl.aerial_image import (
    parameter_sweep_absorber_thickness,
    parameter_sweep_absorber_material,
)
from xrl.plotting import plot_parameter_heatmap
import matplotlib.pyplot as plt


def main():
    mask = XRayMask(
        absorber_material='Ta',
        absorber_thickness=0.5,
        membrane_material='Si3N4',
        membrane_thickness=2.0,
        feature_size=0.5,
        pitch=1.0,
    )

    # -- 1. 2-D energy x gap heatmap --------------------------------------
    energies = np.linspace(0.5, 5.0, 15)
    gaps = np.linspace(1, 50, 15)
    contrast_map = np.zeros((len(energies), len(gaps)))

    for i, energy in enumerate(energies):
        for j, gap in enumerate(gaps):
            sim = AerialImageSimulator(mask, gap=gap)
            x, intensity = sim.compute_aerial_image(energy)
            contrast_map[i, j] = sim.calculate_contrast(x, intensity)
        print(f"  energy {energy:.1f} keV done")

    fig1, _ = plot_parameter_heatmap(gaps, energies, contrast_map)

    # -- 2. Absorber thickness sweep ---------------------------------------
    thicknesses = np.linspace(0.1, 1.5, 15)
    thick_results = parameter_sweep_absorber_thickness(
        mask, energy_kev=1.5, gap=10.0, thicknesses=thicknesses,
    )

    fig2, (ax_c, ax_t) = plt.subplots(1, 2, figsize=(12, 5))
    ax_c.plot(thick_results['thickness_um'], thick_results['contrast'],
              'o-', linewidth=2)
    ax_c.set_xlabel('Absorber Thickness (um)')
    ax_c.set_ylabel('Contrast')
    ax_c.set_title('Contrast vs Absorber Thickness')
    ax_c.grid(True, alpha=0.3)

    ax_t.semilogy(thick_results['thickness_um'],
                  thick_results['transmission_absorber'], 's-', linewidth=2)
    ax_t.set_xlabel('Absorber Thickness (um)')
    ax_t.set_ylabel('Absorber Transmission')
    ax_t.set_title('Transmission vs Absorber Thickness')
    ax_t.grid(True, alpha=0.3)
    fig2.tight_layout()

    # -- 3. Absorber material comparison -----------------------------------
    mat_results = parameter_sweep_absorber_material(
        mask, energy_kev=1.5, gap=10.0,
    )
    print("\nAbsorber material comparison at 1.5 keV:")
    for mat, data in mat_results.items():
        print(f"  {mat}: contrast={data['contrast']:.3f}, "
              f"transmission={data['transmission_absorber']:.4e}")

    plt.show()


if __name__ == '__main__':
    main()
