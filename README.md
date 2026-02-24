# XRL Simulator — X-ray Lithography Process Modeling

A physics-accurate Python package for simulating X-ray lithography (XRL)
mask transmission, aerial image formation, resist exposure, and
thermal-mechanical behaviour of mask membranes.  Developed as part of the
ME6110 Nanomanufacturing course at IIT Bombay (Abhineet Agarwal, Dept. of EE).

---

## What is X-ray Lithography?

X-ray lithography uses short-wavelength radiation (0.5–5 keV) to transfer
sub-micron patterns from a mask to a photoresist.  The mask consists of a
high-Z absorber (Ta, W, or Au) patterned on a thin low-Z membrane (Si₃N₄,
SiC, Diamond, or Polyimide).  The short wavelength minimises diffraction,
enabling high resolution at practical mask-to-resist gaps.

---

## Installation

```bash
git clone https://github.com/abhineet-agarwal/ME6110-XRL-Mask-Fabrication.git
cd ME6110-XRL-Mask-Fabrication
pip install -e .
```

**Dependencies:** numpy ≥ 1.24, scipy ≥ 1.10, matplotlib ≥ 3.7 (pyyaml optional).

---

## Quick Start

```python
from xrl import XRayMask, AerialImageSimulator, ResistExposureModel
from xrl.materials import RESISTS

mask = XRayMask(absorber_material='Ta', absorber_thickness=0.5,
                feature_size=0.5, pitch=1.0)
sim  = AerialImageSimulator(mask, gap=10.0)
x, intensity = sim.compute_aerial_image(energy_kev=1.5)

print(f"Contrast : {sim.calculate_contrast(x, intensity):.3f}")
print(f"FWHM     : {sim.calculate_resolution(x, intensity):.3f} um")
```

---

## Tutorials

| Script | Topic | Physics |
|--------|-------|---------|
| `01_aerial_image.py` | Mask transmission & contrast | Beer-Lambert, Fresnel diffraction |
| `02_resist_response.py` | Exposure & development | Dose, shot noise, CD/LER |
| `03_thermal_analysis.py` | Membrane mechanics | Thermal stress, deflection |
| `04_parameter_sweep.py` | Parameter space exploration | Energy × gap heatmap |
| `05_full_simulation.py` | End-to-end pipeline | Full workflow |

```bash
python examples/01_aerial_image.py
```

---

## Package Structure

```
xrl/
  __init__.py          Public API
  materials.py         Material databases (absorbers, resists, membranes)
  aerial_image.py      Mask transmission + Fresnel propagation
  resist.py            Exposure, development, CD/LER
  thermal.py           Membrane deflection + thermal stress
  plotting.py          Visualisation (separated from physics)
  config.py            SimulationConfig dataclass + JSON/YAML loader
  data/
    nist_xcom.py       Embedded NIST XCOM attenuation tables + interpolator
scripts/
  fetch_nist_xcom.py   Regenerate xrl/data/nist_xcom.py from live NIST pages
tests/
  test_physics.py      35 pytest tests (physics + NIST accuracy)
```

---

## API Overview

| Class / Function | Module | Description |
|------------------|--------|-------------|
| `XRayMask` | `aerial_image` | Mask geometry and Beer-Lambert transmission |
| `AerialImageSimulator` | `aerial_image` | Fresnel propagation to resist plane |
| `ResistExposureModel` | `resist` | Dose, shot noise, development, CD/LER |
| `simulate_full_exposure` | `resist` | End-to-end aerial image → developed profile |
| `MembraneMechanics` | `thermal` | Plate-theory deflection and stress |
| `ThermalAnalysis` | `thermal` | Steady-state temperature and time constant |
| `SimulationConfig` | `config` | Serialisable parameter set (JSON/YAML) |
| `get_mu_rho` | `data.nist_xcom` | NIST XCOM μ/ρ interpolation (log-log) |

---

## Material Database

### Absorbers

| Key | Material | Density (g/cm³) | T at 1.5 keV (0.5 µm) |
|-----|----------|----------------|----------------------|
| `Ta` | Tantalum | 16.6 | 27% |
| `W`  | Tungsten | 19.3 | 20% |
| `Au` | Gold     | 19.3 | 13% |

### Membranes

| Key | Material | Density (g/cm³) | T at 1.5 keV (2 µm) |
|-----|----------|----------------|---------------------|
| `Si3N4`    | Silicon Nitride | 3.44 | 60% |
| `SiC`      | Silicon Carbide | 3.21 | 62% |
| `Diamond`  | Diamond (C)     | 3.52 | 61% |
| `Polyimide`| Kapton (PMDA-ODA)| 1.43 | 78% |

### Resists

| Key | Tone | D₀ (mJ/cm²) | Blur (µm) |
|-----|------|-------------|-----------|
| `PMMA`   | positive | 500 | 0.05 |
| `ZEP520A`| positive |  80 | 0.03 |
| `SU8`    | negative | 150 | 0.08 |
| `HSQ`    | negative | 800 | 0.02 |

---

## Physics Notes

### Attenuation (NIST XCOM data)

All X-ray attenuation coefficients are taken directly from NIST XCOM tabulated
data and interpolated in log-log space using `scipy.interpolate.interp1d`.
Absorption edge discontinuities (stored as consecutive identical energy entries
in NIST tables) are handled by shifting the above-edge row by
`_EDGE_EPS = 1e-8 MeV` (0.01 eV), making them distinct for float64 while
keeping the edge physically sharp.

**Key corrections vs the previous empirical power-law fits:**

| Material | Energy | Old μ/ρ (cm²/g) | NIST μ/ρ (cm²/g) | Error |
|----------|--------|----------------|------------------|-------|
| Ta       | 1.5 keV | 440            | 1566             | 3.6× |
| Ta       | 1.5 keV | M-edges absent | 5 M-edges present (1.7–2.7 keV) | — |
| Si₃N₄    | 1.84 keV | K-edge absent  | 10× jump at Si K-edge | — |
| N        | 1.5 keV | 2140           | 1083             | 2× |

Compound tables (Si₃N₄, SiC, PMMA, Diamond, Polyimide) are computed via the
mixture rule: (μ/ρ)_cmpd = Σ wᵢ (μ/ρ)ᵢ.

### Fresnel Propagation

The Huygens-Fresnel propagator divides by `√(j·λ·z)` (not `j·λ·z`), ensuring a
unit plane wave produces unit intensity at the resist plane.  The previous
implementation amplified intensity by ~120×, making all dose calculations
unphysical.

### Dose Calibration

`simulate_full_exposure` calibrates the exposure time so that `dose_factor × D₀`
lands on the **brightest pixel** of the aerial image (open area after membrane
absorption), not on a hypothetical unit intensity.  This matches real practice:
`dose_factor = 1.5` means the open area receives 1.5× the resist clearing dose.

---

## Running Tests

```bash
pytest tests/ -v   # 35 tests
```

The test suite includes a `TestNISTAccuracy` class with spot-checks against
directly-tabulated NIST values (Ta at 1 keV, Ta M5 edge jump, Si K-edge jump,
Si₃N₄ transmission above and below the Si K-edge).

---

## Changelog

### Physics corrections (2026-02-24)

- **NIST XCOM data** — replaced empirical power-law attenuation fits with
  embedded NIST XCOM tables for H, C, N, O, Si, Ta, W, Au and compound tables
  for Si₃N₄, SiC, PMMA, Diamond, Polyimide.  M-edges and the Si K-edge are now
  correctly represented.
- **Fresnel normalisation** — fixed `/(j·λ·z)` → `/√(j·λ·z)` in the Huygens-
  Fresnel propagator; intensity was previously amplified by ~120×.
- **Dose calibration** — exposure time now calibrated to peak aerial-image
  intensity so `dose_factor` means open-area dose / D₀.
- **Diamond & Polyimide** — added to `MATERIALS` (were in `MEMBRANES` only);
  would previously cause `KeyError` if attenuation was requested.

### Before → After (default conditions: Ta 0.5 µm, Si₃N₄ 2 µm, 1.5 keV)

| Quantity | Before | After |
|----------|--------|-------|
| Ta 0.5 µm transmission | 69% | 27% |
| Si₃N₄ 2 µm transmission | ~100% | 60% |
| Dose range | 11,906–61,569 mJ/cm² | 116–437 mJ/cm² |
| PMMA CD (0.5 µm feature, dose_factor=1.2) | nan | 0.65 µm |

---

## Project Context

This repository also contains materials from the full ME6110 project:

- **`fabrication/`** — CAD files (DXF, STEP) and manufacturing proposal for
  the tantalum X-ray mask fabricated in collaboration with ISRO.
- **`ME6110-Abhineet/`** — Final report, presentation slides, and plagiarism check.
- **`ME6110_projects.pdf`** — Problem statement.

---

## License

MIT

## Author

Abhineet Agarwal
Dept. of Electrical Engineering, IIT Bombay
