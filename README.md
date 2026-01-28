# XRL Simulator -- X-ray Lithography Process Modeling

A tutorial-style Python package for simulating X-ray lithography (XRL) mask
transmission, aerial image formation, resist exposure, and thermal-mechanical
behaviour.  Developed as part of the ME6110 Nanomanufacturing course at
IIT Bombay.

## What is X-ray Lithography?

X-ray lithography uses short-wavelength radiation (0.5--5 keV) to transfer
sub-micron patterns from a mask to a photoresist.  The mask consists of a
high-Z absorber (Ta, W, or Au) patterned on a thin low-Z membrane (Si3N4 or
SiC).  The short wavelength minimises diffraction, enabling high resolution
at practical mask-to-resist gaps.

## Installation

```bash
# Clone the repository
git clone https://github.com/abhineet-agarwal/ME6110-XRL-Mask-Fabrication.git
cd ME6110-XRL-Mask-Fabrication

# Install in editable mode
pip install -e .
```

**Dependencies:** numpy, scipy, matplotlib (see `requirements.txt`).

## Quick Start

```python
from xrl import XRayMask, AerialImageSimulator

mask = XRayMask(absorber_material='Ta', feature_size=0.5, pitch=1.0)
sim  = AerialImageSimulator(mask, gap=10.0)
x, intensity = sim.compute_aerial_image(energy_kev=1.5)

print(f"Contrast: {sim.calculate_contrast(x, intensity):.3f}")
```

## Tutorials

The `examples/` directory contains five numbered tutorials that walk through
the full simulation workflow:

| Script | Topic | Physics |
|--------|-------|---------|
| `01_aerial_image.py` | Mask transmission & contrast | Beer-Lambert, Fresnel diffraction |
| `02_resist_response.py` | Exposure & development | Dose, shot noise, CD/LER |
| `03_thermal_analysis.py` | Membrane mechanics | Thermal stress, deflection |
| `04_parameter_sweep.py` | Parameter space exploration | Energy x gap heatmap |
| `05_full_simulation.py` | End-to-end pipeline | Complete workflow |

Run any tutorial with:

```bash
python examples/01_aerial_image.py
```

## Package Structure

```
xrl/
  __init__.py        # Public API
  materials.py       # Material databases (absorbers, resists, membranes)
  aerial_image.py    # Mask transmission + Fresnel propagation
  resist.py          # Exposure, development, CD/LER
  thermal.py         # Membrane deflection + thermal stress
  plotting.py        # All visualization (separated from physics)
  config.py          # SimulationConfig dataclass + JSON/YAML loader
```

## API Overview

| Class / Function | Module | Description |
|------------------|--------|-------------|
| `XRayMask` | `aerial_image` | Mask geometry and Beer-Lambert transmission |
| `AerialImageSimulator` | `aerial_image` | Fresnel propagation to resist plane |
| `ResistExposureModel` | `resist` | Dose, noise, development, CD/LER |
| `MembraneMechanics` | `thermal` | Plate-theory deflection and stress |
| `ThermalAnalysis` | `thermal` | Steady-state temperature and time constant |
| `SimulationConfig` | `config` | Serialisable parameter set |

## Physics Background

For detailed derivations and validation against literature, see the project
report: [`ME6110-Abhineet/ME6110_report.pdf`](ME6110-Abhineet/ME6110_report.pdf).

## Project Context

This repository also contains materials from the full ME6110 project:

- **`fabrication/`** -- CAD files (DXF, STEP) and manufacturing proposal for
  the tantalum X-ray mask fabricated in collaboration with ISRO.
- **`ME6110-Abhineet/`** -- Final report, presentation slides, and plagiarism
  check.
- **`ME6110_projects.pdf`** -- Problem statement.

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## License

MIT

## Author

Abhineet Agarwal
Dept of Electrical Engineering
IIT Bombay
