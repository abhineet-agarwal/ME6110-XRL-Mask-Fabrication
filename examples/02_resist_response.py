#!/usr/bin/env python3
"""
Tutorial 2: Resist Exposure and Development
=============================================

This script simulates resist exposure from an aerial image through
to the developed resist profile, including stochastic effects.

Physics covered:
- Dose calculation from photon flux
- Photon shot noise (Poisson statistics)
- Acid diffusion / secondary-electron blur
- Positive and negative tone development models
- Critical dimension (CD) and line-edge roughness (LER) extraction

Run::

    python examples/02_resist_response.py
"""

import numpy as np
from xrl import XRayMask, AerialImageSimulator, RESISTS
from xrl.resist import simulate_full_exposure, dose_sweep_study
from xrl.plotting import plot_resist_profile, plot_cd_vs_dose, plot_ler_vs_dose
import matplotlib.pyplot as plt


def main():
    # -- Generate an aerial image to use as input --------------------------
    mask = XRayMask(
        absorber_material='Ta',
        absorber_thickness=0.5,
        membrane_material='Si3N4',
        membrane_thickness=2.0,
        feature_size=0.5,
        pitch=1.0,
    )
    sim = AerialImageSimulator(mask, gap=10.0)
    x, intensity = sim.compute_aerial_image(energy_kev=1.5)

    # -- 1. Expose and develop PMMA (positive tone) -----------------------
    resist = RESISTS['PMMA']
    dose, developed, metrics = simulate_full_exposure(
        intensity, x, resist, energy_kev=1.5, dose_factor=1.2,
    )
    print(f"PMMA: CD = {metrics['cd_um']:.3f} um, LER = {metrics['ler_nm']:.1f} nm")

    fig1, _ = plot_resist_profile(x, developed, resist_name='PMMA')

    # -- 2. Compare all resists at same conditions -------------------------
    fig2, ax2 = plt.subplots(2, 2, figsize=(12, 9))
    for i, (name, res) in enumerate(RESISTS.items()):
        row, col = divmod(i, 2)
        _, dev, met = simulate_full_exposure(
            intensity, x, res, energy_kev=1.5, dose_factor=1.2,
        )
        ax2[row, col].fill_between(x, 0, dev, alpha=0.4)
        ax2[row, col].plot(x, dev, linewidth=1.5)
        cd_s = f"{met['cd_um']:.3f}" if not np.isnan(met['cd_um']) else "N/A"
        ler_s = f"{met['ler_nm']:.1f}" if not np.isnan(met['ler_nm']) else "N/A"
        ax2[row, col].set_title(f"{name}  (CD={cd_s} um, LER={ler_s} nm)")
        ax2[row, col].set_ylim(-0.05, 1.1)
        ax2[row, col].set_xlabel('Position (um)')
        ax2[row, col].set_ylabel('Remaining thickness')
        ax2[row, col].grid(True, alpha=0.3)
    fig2.suptitle('Resist Comparison', fontsize=14, fontweight='bold')
    fig2.tight_layout()

    # -- 3. Dose sweep for PMMA -------------------------------------------
    dose_factors = np.linspace(0.5, 3.0, 12)
    sweep = dose_sweep_study(intensity, x, resist, energy_kev=1.5,
                             dose_factors=dose_factors)

    fig3, _ = plot_cd_vs_dose(sweep['dose_factor'], sweep['cd_um'],
                              resist_name='PMMA')
    fig4, _ = plot_ler_vs_dose(sweep['dose_factor'], sweep['ler_nm'],
                               resist_name='PMMA')

    plt.show()


if __name__ == '__main__':
    main()
