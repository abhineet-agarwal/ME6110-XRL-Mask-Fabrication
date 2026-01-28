#!/usr/bin/env python3
"""
Tutorial 3: Thermal-Mechanical Analysis of XRL Masks
=====================================================

This script compares membrane materials under X-ray beam loading
and plots deflection, stress, and temperature as a function of
beam power.

Physics covered:
- Steady-state conduction and convection in thin membranes
- Biaxial thermal stress from constrained expansion
- Empirically calibrated thermal deflection model
- Material selection: Si3N4 vs SiC vs Diamond

Run::

    python examples/03_thermal_analysis.py
"""

import numpy as np
from xrl import MEMBRANES, MembraneMechanics, ThermalAnalysis
from xrl.thermal import exposure_scenario_analysis, material_comparison
from xrl.plotting import (
    plot_thermal_deflection,
    plot_thermal_stress,
    plot_material_comparison_bar,
)
import matplotlib.pyplot as plt


def main():
    # -- 1. Single-material scenario analysis ------------------------------
    mat = MEMBRANES['Si3N4']
    membrane = MembraneMechanics(mat, thickness=2.0, size=50.0, geometry='square')
    powers = np.logspace(-3, 0, 20)
    results = exposure_scenario_analysis(membrane, beam_power_range=powers)

    fig1, _ = plot_thermal_deflection(
        results['beam_power_W'], results['deflection_um'],
        materials=['Si3N4'],
    )
    fig2, _ = plot_thermal_stress(
        results['beam_power_W'], results['thermal_stress_MPa'],
        materials=['Si3N4'],
    )

    # -- 2. Material comparison at 0.1 W -----------------------------------
    comp = material_comparison(thickness=2.0, size=50.0, beam_power=0.1)

    names = list(comp.keys())
    deflections = [comp[n]['deflection'] for n in names]
    stresses = [comp[n]['stress'] for n in names]

    fig3, _ = plot_material_comparison_bar(
        names, deflections,
        ylabel='Deflection (um)',
        title='Membrane Deflection at 0.1 W',
    )
    fig4, _ = plot_material_comparison_bar(
        names, stresses,
        ylabel='Thermal Stress (MPa)',
        title='Thermal Stress at 0.1 W',
    )

    # -- 3. Multi-material power sweep -------------------------------------
    materials_to_compare = ['Si3N4', 'SiC', 'Diamond']
    defl_dict: dict[str, np.ndarray] = {}
    stress_dict: dict[str, np.ndarray] = {}

    for mat_name in materials_to_compare:
        mem = MembraneMechanics(MEMBRANES[mat_name], 2.0, 50.0, 'square')
        res = exposure_scenario_analysis(mem, beam_power_range=powers)
        defl_dict[mat_name] = res['deflection_um']
        stress_dict[mat_name] = res['thermal_stress_MPa']

    fig5, _ = plot_thermal_deflection(powers, defl_dict)
    fig6, _ = plot_thermal_stress(powers, stress_dict)

    plt.show()


if __name__ == '__main__':
    main()
