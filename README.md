# ME6110: X-ray Mask Fabrication and XRL Feasibility Study

**Course:** ME6110: Nanomanufacturing Processes (Autumn 2025, IIT Bombay)
**Instructor:** Prof. Rakesh Mote
**Student:** Abhineet Agarwal (22B1219)

---

## Overview

High-precision X-ray mask fabrication for satellite imaging applications, in collaboration with ISRO. The project covers:

1. **Mask Fabrication** — Designing and fabricating tantalum X-ray masks using laser micromachining, EDM, and etching techniques to achieve sub-10 µm pattern accuracy
2. **XRL Simulation** — Comprehensive simulation of X-ray lithography (XRL) processes including aerial image formation, resist response modeling, and thermal-mechanical analysis
3. **Commercialization** — Exploring the feasibility of X-ray lithography for CMOS fabrication

## Simulations

Python-based simulation suite covering:
- **Aerial image analysis** — Multi-energy and multi-gap diffraction modeling for X-ray proximity lithography
- **Resist response** — CD vs dose, line-edge roughness, stochastic effects, resist profile simulation
- **Thermal-mechanical** — Mask deflection and stress analysis under X-ray beam power loading
- **Absorber material comparison** — Evaluating different mask absorber materials

## Repository Structure

```
ME6110-Abhineet/              # Report, presentation, plagiarism check
xrl_project/                  # Project simulation code (v1)
xrl_project_complete_FINAL/   # Final simulation package
  simulations/                # Python simulation modules
  layouts/                    # GDS test pattern generation
  docs/                       # Beamtime proposal, integration roadmap
  data/                       # Simulation output data

High_Precision_X_Ray_Mask_../ # Figures, LaTeX source, fabricated mask images
aerial-image.py               # Aerial image simulation (standalone)
resist_response.py            # Resist response simulation (standalone)
CAM-20x20_v1_sitare-1.DXF    # CAD mask layout (DXF)
CAM-20x20_v1_sitare-1.STEP   # CAD mask layout (STEP)
```

## Tools

- **Python** (NumPy, SciPy, Matplotlib) — Simulation and analysis
- **GDSpy** — GDSII layout generation for test patterns
- **LaTeX** — Report typesetting
