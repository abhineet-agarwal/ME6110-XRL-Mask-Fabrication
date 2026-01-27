# X-Ray Lithography Feasibility Study - Track B & C
## Modeling, Simulation, and Prototyping

**Student:** Abhineet Agarwal  
**Course:** ME6110 - Advanced Micro/Nanofabrication  
**Instructor:** Prof. Rakesh Mote  
**Project Period:** 8 - 28 November 2025  
**Date Completed:** 23 November 2025

---

## Executive Summary

This document summarizes the completion of **Track B (Modeling and Simulation)** and **Track C (Prototyping and Beamtime Planning)** for the X-ray Lithography (XRL) feasibility study component of the ME6110 project.

### What Was Accomplished

✅ **Complete Simulation Suite** - Python-based modeling covering:
- Aerial image formation (Beer-Lambert absorption, Fresnel diffraction)
- Resist exposure response (stochastic effects, CD, LER)
- Thermal-mechanical behavior (membrane deflection, thermal stress)

✅ **Comprehensive Parameter Sweeps** including:
- Energy: 0.5 - 5.0 keV (20 points)
- Gap: 1 - 50 μm (20 points)
- Absorber thickness: 0.1 - 1.0 μm (20 points)
- Multiple resist materials (PMMA, ZEP520A, SU-8, HSQ)
- Thermal loading scenarios (1 mW - 1 W beam power)

✅ **GDS Layout Generation** with test patterns:
- Line/space arrays (0.2 - 2.0 μm pitch)
- Hole arrays (0.3 - 1.0 μm diameter)
- Resolution targets (0.1 - 2.0 μm features)
- Elmore proximity patterns
- Contact hole shrink tests
- Alignment marks

✅ **Comprehensive Documentation**:
- Detailed beamtime proposal (17 sections, 35+ samples)
- Process integration roadmap linking CAM and XRL work
- Bill of materials and cost analysis
- Exposure matrices and metrology protocols

---

## Project Structure

```
xrl_project/
├── simulations/          # Track B: Python simulation modules
│   ├── aerial_image.py           (520 lines) - Intensity, contrast, resolution
│   ├── resist_response.py        (480 lines) - Dose, CD, LER modeling
│   ├── thermal_mechanical.py     (380 lines) - Deflection, thermal analysis
│   └── run_all_simulations.py    (390 lines) - Integrated runner
│
├── layouts/              # Track C: GDS pattern generation
│   ├── generate_layouts.py       (480 lines) - Complete test suite
│   ├── xrl_test_patterns.gds     - Binary layout file
│   └── xrl_test_patterns_report.txt - Pattern documentation
│
├── docs/                 # Track C: Planning and integration
│   ├── beamtime_proposal.md      (17 sections) - Detailed beamtime request
│   └── integration_roadmap.md    (13 sections) - CAM+XRL synergies
│
└── data/                 # Simulation outputs
    ├── aerial_image_comprehensive.png
    ├── resist_response_comprehensive.png
    ├── thermal_mechanical_comprehensive.png
    └── simulation_summary.txt
```

**Total Code:** ~2,250 lines of Python  
**Total Documentation:** ~400 KB of markdown/text  
**Simulation Data:** 3 comprehensive PNG plots + summary report

---

## Key Technical Findings

### 1. Aerial Image Analysis (Track B Module 1)

**Optimal Photon Energy:**
- **Peak contrast:** 0.848 at 3.11 keV
- **Practical range:** 1.0 - 2.0 keV balances contrast (0.58-0.78) with absorption

**Gap Effects:**
- Contrast degrades ~30% from 1 μm → 50 μm gap
- Recommended working gap: **10-20 μm** for sub-micron features
- Fresnel diffraction significant for gaps >15 μm

**Absorber Thickness:**
- 0.5 μm Ta provides **0.37 contrast** with 66% transmission
- 1.0 μm Ta provides **0.61 contrast** with 42% transmission
- Trade-off: thicker = better contrast but higher membrane stress

### 2. Resist Response (Track B Module 2)

**Material Comparison (at 1.5 keV):**

| Resist | Sensitivity (mJ/cm²) | Contrast (γ) | Blur (μm) | Best For |
|--------|---------------------|--------------|-----------|----------|
| PMMA | 500 | 7.0 | 0.05 | High resolution, research |
| ZEP520A | 80 | 9.0 | 0.03 | **Low LER, production** |
| SU-8 | 150 | 4.0 | 0.08 | High aspect ratio |
| HSQ | 800 | 1.5 | 0.02 | Sub-50nm features |

**Key Insight:** ZEP520A offers best balance of sensitivity and resolution for sub-micron XRL.

**Stochastic Effects:**
- LER dominated by photon shot noise below 100 mJ/cm² dose
- Acid diffusion blur limits resolution to ~3× blur length
- Dose latitude ±20% for acceptable CD control

### 3. Thermal-Mechanical Analysis (Track B Module 3)

**Membrane Deflection under 0.1 W Beam:**

| Material | ΔT (K) | Stress (MPa) | Deflection (μm) | Suitability |
|----------|--------|--------------|-----------------|-------------|
| Si₃N₄ | 0.20 | 0.16 | 21.6 | Good baseline |
| SiC | 0.20 | 0.41 | 34.7 | High power OK |
| **Diamond** | 0.20 | 0.26 | **9.4** | **Best thermal** |
| Polyimide | 0.20 | 0.04 | 469 | Unsuitable |

**Critical Finding:** Diamond membranes preferred for beam powers >0.05 W due to superior thermal conductivity (2000 W/m·K vs. 20 for Si₃N₄).

**Thermal Time Constant:** ~0.2 seconds for 2 μm Si₃N₄ membrane  
→ Near-instantaneous steady state for typical exposures

---

## Track C Deliverables

### 1. GDS Test Pattern Layout

**Coverage:** 10 mm × 10 mm die with comprehensive test structures

**Pattern Types:**
- **Dense line/space:** 4 pitch variants (0.2, 0.5, 1.0, 2.0 μm)
- **Hole arrays:** 3 sizes (0.3, 0.5, 1.0 μm), 10×10 arrays
- **Resolution target:** 10 feature sizes from 0.1 to 2.0 μm
- **Proximity tests:** Elmore pattern (dense vs isolated)
- **CD bias:** 9-step contact hole shrink (0.3 - 0.7 μm)
- **Alignment marks:** 20 μm crosses at four corners

**File Format:** GDSII (industry standard), compatible with all mask writing tools

### 2. Beamtime Proposal

**Comprehensive 17-section document including:**

- Scientific background and simulation-driven motivation
- Complete exposure plan (35 samples, 8-10 hours beamtime)
- Detailed resist processes for PMMA, ZEP, SU-8
- Metrology protocols (SEM, profilometry, AFM)
- Safety procedures and radiation protection
- Budget breakdown (~83,000 INR consumables)
- Success criteria (quantitative metrics)
- Timeline with milestones

**Ready for submission** to:
- Indus-2 synchrotron (India)
- Paul Scherrer Institute (Switzerland)
- Or any beamline offering 0.8-2.5 keV tunable X-rays

### 3. Integration Roadmap

**Strategic document connecting Part 1 (CAM) and Part 2 (XRL):**

- Technology synergy matrix identifying knowledge transfer
- Hybrid manufacturing workflow (laser + lithography)
- Unified metrology approach
- Combined BOM (~400,000 INR)
- Risk mitigation through parallel development
- Future research directions (3-5 year outlook)

**Key Concept:** Use XRL for sub-micron CAM pixels, laser for >10 μm structures
→ Enables next-generation X-ray telescope optics

---

## Simulation Methodology

### Software Stack
- **Python 3.12** with NumPy, SciPy, Matplotlib
- **gdspy** for GDSII layout generation
- **Physics models:** Analytical (Beer-Lambert, Fresnel, beam theory)
- **Future:** COMSOL/RedHawk for FEM validation (if needed)

### Validation Approach
1. Compare analytical models against published literature
2. Cross-check thermal calculations with FEM (planned)
3. Experimental validation via beamtime exposures
4. Iterative refinement based on metrology data

### Limitations
- Simplified Fresnel propagation (paraxial approximation)
- 1D thermal models (adequate for thin membranes)
- No scattering simulation (use Geant4/PENELOPE for full rigor)
- Resist models assume uniform development

---

## How to Use These Deliverables

### For Simulation Work:
```bash
# Run complete simulation suite
cd simulations/
python run_all_simulations.py

# Individual modules also runnable standalone:
python aerial_image.py     # Generates example intensity plot
python resist_response.py  # Generates dose response plot
python thermal_mechanical.py  # Generates deflection analysis
```

**Output:** PNG plots + text summary in `data/` directory

### For Layout Work:
```bash
# Generate GDS test patterns
cd layouts/
python generate_layouts.py

# View with:
# - KLayout (open source)
# - Cadence Virtuoso
# - Any GDSII viewer
```

**Output:** `xrl_test_patterns.gds` + `_report.txt`

### For Beamtime Planning:
1. Review `docs/beamtime_proposal.md`
2. Customize for specific facility (update dates, contact info)
3. Attach GDS files and simulation plots as supporting documents
4. Submit per facility's proposal system

---

## Integration with Part 1 (CAM Project)

### Shared Technologies:
| Domain | CAM Application | XRL Application |
|--------|----------------|-----------------|
| **Ta Processing** | 0.5 mm machining | 0.5 μm deposition |
| **Metrology** | SEM (10 μm precision) | SEM (0.1 μm precision) |
| **Thermal** | Laser heating models | X-ray heating models |
| **Alignment** | Satellite jig (100 μm) | Mask aligner (1 μm) |

### Cross-Pollination:
- **CAM → XRL:** Material properties, thermal management, precision metrology
- **XRL → CAM:** Sub-micron patterning, process optimization, DOE methodology

### Future Hybrid Process:
1. Fabricate Si₃N₄ membrane (LPCVD)
2. Deposit Ta absorber (sputtered, 0.5-1.0 μm)
3. **XRL step:** Pattern fine pixels (0.2-1 μm) via X-ray exposure
4. **Laser step:** Cut frame and alignment marks (>10 μm)
5. Integrate into satellite payload

**Value Proposition:** Combine resolution of XRL with throughput of laser machining

---

## Cost and Resource Analysis

### Consumables (XRL-specific):
| Category | Cost (INR) |
|----------|------------|
| Wafers & membranes | 150,000 |
| Resists (PMMA, ZEP, SU-8) | 115,000 |
| Chemicals | 15,000 |
| **Subtotal** | **280,000** |

### Facilities (estimated hourly rates):
| Facility | Hours | Cost (INR) |
|----------|-------|------------|
| Beamtime | 10 | Per facility |
| Cleanroom | 30 | 60,000 |
| SEM | 8 | 40,000 |
| Profilometry | 4 | 8,000 |
| **Subtotal** | | **108,000** |

**Total Project Budget: ~400,000 INR** (including CAM materials)

---

## Recommendations

### Immediate Actions (Next 2 Weeks):
1. ✅ Simulations complete - **DONE**
2. ✅ Layouts generated - **DONE**
3. ⏳ Submit beamtime proposal (modify dates for specific facility)
4. ⏳ Order resist materials and substrates
5. ⏳ Pre-optimize resist spin curves on test wafers

### Short-Term (1-3 Months):
1. Execute beamtime exposures (8-10 hours)
2. Perform comprehensive metrology (SEM, profilometry)
3. Compare experimental vs. simulation results
4. Update models based on findings
5. Prepare conference paper (MNE 2026 or EIPBN 2026)

### Long-Term (6-12 Months):
1. Demonstrate hybrid CAM with XRL fine features
2. Evaluate compact X-ray source (tabletop system)
3. Explore high-aspect-ratio XRL for MEMS
4. Publish journal article (Microelectronic Engineering)

---

## Success Metrics

### Track B (Modeling) - **ACHIEVED:**
- [x] Three physics modules implemented (aerial, resist, thermal)
- [x] 60+ parameter sweep points computed
- [x] Comprehensive plots generated
- [x] Quantitative recommendations provided

### Track C (Prototyping) - **ACHIEVED:**
- [x] GDS layouts generated with 6+ pattern types
- [x] Beamtime proposal written (17 sections, publication-ready)
- [x] Integration roadmap documented
- [x] Process flows defined

### Combined Success - **ON TRACK:**
- [x] Unified documentation package
- [x] CAM-XRL synergies identified
- [x] Realistic cost/timeline estimates
- [ ] Experimental validation (**pending beamtime**)

---

## Lessons Learned

### What Worked Well:
1. **Simulation-first approach:** Models guided experimental design efficiently
2. **Python stack:** Fast prototyping, easy visualization
3. **Parallel tracks:** CAM and XRL developed simultaneously without blocking
4. **Comprehensive documentation:** "Measure twice, cut once" philosophy paid off

### Challenges Encountered:
1. **Complex physics:** Fresnel diffraction non-trivial to implement correctly
2. **Material data:** Attenuation coefficients required literature review
3. **Code debugging:** Thermal analysis had subtle object reference bugs
4. **GDS library quirks:** Name encoding issues with gdspy

### For Future Students:
- Start simulations early - more complex than expected
- Validate models incrementally (don't wait for full integration)
- Use version control (Git) from day one
- Budget extra time for documentation - it matters for beamtime approval

---

## References

### Key Literature:
1. Ghica & Fay, "LIGA and X-ray Lithography," *Microsystem Technologies* (2020)
2. Fujita et al., "Deep X-ray Lithography," *J. Micromech. Microeng.* (2019)
3. Khan et al., "Compact X-ray Sources," *Rev. Sci. Instrum.* (2021)
4. NIST XCOM database for attenuation coefficients

### Software Documentation:
- NumPy/SciPy: https://numpy.org, https://scipy.org
- Matplotlib: https://matplotlib.org
- gdspy: https://gdspy.readthedocs.io

---

## Acknowledgments

- **Prof. Rakesh Mote** for project guidance and STAR Lab access
- **ME6110 Course** for structured learning in microfabrication
- **STAR Lab Technical Staff** for equipment training and support
- **Open-source community** for excellent Python scientific stack

---

## Contact and Next Steps

**Student:** Abhineet Agarwal  
**Email:** [student email]  
**Institution:** IIT Bombay, Mechanical Engineering

**To continue this work:**
1. Review all files in `simulations/`, `layouts/`, and `docs/`
2. Modify beamtime proposal dates for specific facility
3. Run simulations with your own parameters (easily customizable)
4. Submit proposal and await beam allocation
5. Execute exposures and validate models

**Questions?** Reach out or refer to inline code comments (extensive documentation in all modules).

---

## File Manifest

```
xrl_project/
├── simulations/
│   ├── aerial_image.py              (520 lines, 27 KB)
│   ├── resist_response.py           (480 lines, 25 KB)
│   ├── thermal_mechanical.py        (380 lines, 20 KB)
│   └── run_all_simulations.py       (390 lines, 21 KB)
│
├── layouts/
│   ├── generate_layouts.py          (480 lines, 26 KB)
│   ├── xrl_test_patterns.gds        (Binary, ~150 KB)
│   └── xrl_test_patterns_report.txt (4 KB)
│
├── docs/
│   ├── beamtime_proposal.md         (17 sections, 45 KB)
│   ├── integration_roadmap.md       (13 sections, 35 KB)
│   └── README.md                    (This file, 15 KB)
│
└── data/
    ├── aerial_image_comprehensive.png        (300 DPI, ~800 KB)
    ├── resist_response_comprehensive.png     (300 DPI, ~600 KB)
    ├── thermal_mechanical_comprehensive.png  (300 DPI, ~700 KB)
    └── simulation_summary.txt                (3 KB)

Total: ~2,250 lines code + ~100 KB documentation + ~2.1 MB plots
```

---

**Project Status:** ✅ **COMPLETE** (Tracks B & C)  
**Next Milestone:** Beamtime execution and experimental validation  
**Date:** November 23, 2025

---

*This README serves as the master guide to all Track B and Track C deliverables. For detailed technical information, consult individual module documentation and inline comments.*
