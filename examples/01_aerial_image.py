#!/usr/bin/env python3
"""
Tutorial 1: Aerial Image Formation in X-ray Lithography
=========================================================

This script demonstrates how X-ray mask transmission and Fresnel
diffraction produce the aerial image at the resist plane.

Physics covered:
- Beer-Lambert absorption through absorber and membrane layers
- Fresnel diffraction propagation across the mask-resist gap
- Image contrast as a function of photon energy and gap distance

Run::

    python examples/01_aerial_image.py
"""

import numpy as np
from xrl import XRayMask, AerialImageSimulator
from xrl.aerial_image import parameter_sweep_energy, parameter_sweep_gap
from xrl.plotting import (
    plot_aerial_image,
    plot_contrast_vs_energy,
    plot_contrast_vs_gap,
)
import matplotlib.pyplot as plt


def main():
    # -- 1. Create a standard X-ray mask ----------------------------------
    # 0.5 um Ta absorber on 2 um Si3N4 membrane, 500 nm lines at 1 um pitch
    mask = XRayMask(
        absorber_material='Ta',
        absorber_thickness=0.5,
        membrane_material='Si3N4',
        membrane_thickness=2.0,
        feature_size=0.5,
        pitch=1.0,
    )

    # -- 2. Compute aerial image at a single condition --------------------
    sim = AerialImageSimulator(mask, gap=10.0)
    x, intensity = sim.compute_aerial_image(energy_kev=1.5)

    contrast = sim.calculate_contrast(x, intensity)
    resolution = sim.calculate_resolution(x, intensity)
    print(f"Single image: contrast = {contrast:.3f}, FWHM = {resolution:.3f} um")

    fig1, _ = plot_aerial_image(
        x, intensity,
        title=f'Aerial Image: {mask.feature_size} um lines, {sim.gap} um gap',
    )

    # -- 3. Energy sweep ---------------------------------------------------
    energies = np.linspace(0.5, 5.0, 15)
    energy_results = parameter_sweep_energy(mask, gap=10.0, energies=energies)

    fig2, _ = plot_contrast_vs_energy(
        energy_results['energy_kev'],
        energy_results['contrast'],
    )

    # -- 4. Gap sweep ------------------------------------------------------
    gaps = np.linspace(1, 50, 15)
    gap_results = parameter_sweep_gap(mask, energy_kev=1.5, gaps=gaps)

    fig3, _ = plot_contrast_vs_gap(
        gap_results['gap_um'],
        gap_results['contrast'],
    )

    plt.show()


if __name__ == '__main__':
    main()
