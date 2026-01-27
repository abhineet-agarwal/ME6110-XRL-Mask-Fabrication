# X-Ray Lithography Beamtime Proposal
## Experimental Validation of XRL Process for Sub-Micron Patterning

**Principal Investigator:** Abhineet Agarwal  
**Institution:** IIT Bombay, Department of Mechanical Engineering  
**Course:** ME6110 - Advanced Micro/Nanofabrication  
**Instructor:** Prof. Rakesh Mote  
**Proposal Date:** November 2025

---

## 1. Executive Summary

We propose a systematic experimental validation of X-ray lithography (XRL) for sub-micron pattern transfer using optimized process parameters derived from comprehensive modeling and simulation work. This beamtime request aims to:

1. Verify simulation predictions for contrast, resolution, and proximity effects
2. Characterize multiple resist materials (PMMA, ZEP520A, SU-8)
3. Establish process windows for feature sizes from 100 nm to 2 μm
4. Demonstrate integration with coded aperture mask (CAM) fabrication expertise

The experiments will utilize pre-fabricated test masks with well-characterized geometries, enabling direct comparison between simulation and experimental results.

---

## 2. Scientific Background and Motivation

### 2.1 Context
X-ray lithography offers unique advantages for high-aspect-ratio microstructures and direct pattern transfer without complex multi-patterning schemes. Recent advances in compact X-ray sources and modern resists have renewed interest in XRL for specialized applications including:
- MEMS/NEMS fabrication
- High-aspect-ratio structures
- X-ray optics (coded aperture masks)
- Advanced packaging

### 2.2 Simulation-Driven Approach
Our modeling work (Track B) has identified optimal exposure conditions:

| Parameter | Optimal Range | Basis |
|-----------|---------------|-------|
| Photon Energy | 1.0 - 2.0 keV | Maximum contrast (>0.85) |
| Mask-Resist Gap | 5 - 20 μm | Acceptable proximity effects |
| Absorber Thickness | 0.5 - 0.8 μm Ta | >95% absorption, minimal stress |
| Resist Choice | ZEP520A / PMMA | Low LER (<5 nm), high sensitivity |

Experimental validation will confirm these predictions and establish process robustness.

---

## 3. Experimental Objectives

### Primary Objectives
1. **Contrast Characterization:** Measure aerial image contrast vs. photon energy and gap
2. **Resolution Limits:** Determine minimum resolvable feature size for each resist
3. **Proximity Effect Quantification:** Measure isolated vs. dense line CD variations
4. **Dose-to-Clear Determination:** Establish D₀ and process latitude for each resist
5. **Line-Edge Roughness (LER):** Quantify stochastic effects and compare with simulation

### Secondary Objectives
6. Thermal stability assessment under realistic beam conditions
7. Alignment accuracy evaluation using test marks
8. Multi-layer registration feasibility study

---

## 4. Mask Design and Specifications

### 4.1 Mask Stack
- **Membrane:** 2 μm Si₃N₄ (low-stress LPCVD)
- **Absorber:** 0.5 μm Ta (sputtered)
- **Frame:** Silicon, 525 μm thick, 50 mm × 50 mm
- **Pattern Area:** 10 mm × 10 mm

### 4.2 Test Pattern Suite (see GDS layout)
1. **Line/Space Arrays:** Pitches of 0.2, 0.5, 1.0, 2.0 μm (duty cycle 50%)
2. **Hole Arrays:** Diameters 0.3, 0.5, 1.0 μm (10×10 arrays)
3. **Resolution Target:** Feature sizes 0.1 - 2.0 μm (10 steps)
4. **Elmore Pattern:** Dense/isolated comparison
5. **Contact Hole Bias Test:** 0.3 - 0.7 μm diameter range
6. **Alignment Marks:** 20 μm crosses at four corners

All patterns provided in GDSII format with accompanying documentation.

---

## 5. Resist Processes

### 5.1 Candidate Resists

| Resist | Thickness | Spin | Bake | Developer | Tone |
|--------|-----------|------|------|-----------|------|
| PMMA 950K | 1.0 μm | 4000 rpm | 180°C, 10 min | MIBK:IPA 1:3 | Positive |
| ZEP520A | 0.5 μm | 3000 rpm | 180°C, 5 min | ZED-N50 | Positive |
| SU-8 2005 | 5.0 μm | 2000 rpm | 95°C, 5 min | SU-8 Developer | Negative |

### 5.2 Process Flow
1. **Substrate Preparation:**
   - Silicon wafers (100 mm diameter, <100> orientation)
   - RCA clean + dehydration bake (120°C, 10 min)
   
2. **Adhesion Promotion:**
   - HMDS vapor prime (120°C, 5 min) for positive resists
   - Omit for SU-8

3. **Resist Coating:**
   - Spin-coat per table above
   - Edge bead removal (EBR) if needed

4. **Soft Bake:**
   - Hotplate as specified
   - Cool gradually to room temperature

5. **X-ray Exposure:**
   - Align mask to substrate
   - Set gap with precision spacers
   - Expose per dose matrix (see Section 6)

6. **Post-Exposure Processing:**
   - Post-bake if required (SU-8: 95°C, 5 min)
   - Develop as specified
   - Rinse (IPA) and dry (N₂)

7. **Metrology:**
   - SEM imaging (CD, LER, profile)
   - Optical profilometry (thickness, height)
   - AFM for surface roughness

---

## 6. Exposure Plan

### 6.1 Beamline Requirements
- **Photon Energy Range:** 0.8 - 2.5 keV (tunable)
- **Flux:** ≥1×10¹² photons/(s·cm²) at sample position
- **Beam Uniformity:** ±5% over 10 mm × 10 mm area
- **Stability:** <1% flux variation over exposure duration

### 6.2 Exposure Matrix

**Energy Sweep (PMMA Resist):**
| Energy (keV) | Gap (μm) | Dose (mJ/cm²) | Exposure Time (s) | Samples |
|--------------|----------|---------------|-------------------|---------|
| 1.0 | 10 | 400, 500, 600 | ~40-60 | 3 |
| 1.5 | 10 | 400, 500, 600 | ~40-60 | 3 |
| 2.0 | 10 | 400, 500, 600 | ~40-60 | 3 |

**Gap Sweep (1.5 keV, PMMA):**
| Gap (μm) | Dose (mJ/cm²) | Samples |
|----------|---------------|---------|
| 5 | 500 | 2 |
| 10 | 500 | 2 |
| 20 | 500 | 2 |
| 30 | 500 | 2 |

**Resist Comparison (1.5 keV, 10 μm gap):**
| Resist | Dose Range (mJ/cm²) | Samples |
|--------|---------------------|---------|
| PMMA | 400, 500, 600 | 3 |
| ZEP520A | 60, 80, 100 | 3 |
| SU-8 | 100, 150, 200 | 3 |

**Total Sample Count:** ~35 exposures

### 6.3 Timing Estimate
- Setup and alignment: 2 hours
- Mask installation: 1 hour
- Exposures (35 samples × 1 min average): 3 hours
- Sample exchange and inspection: 2 hours
- **Total beamtime required: 8-10 hours**

---

## 7. Expected Results and Success Criteria

### 7.1 Quantitative Metrics
| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Minimum Feature Size | ≤200 nm | SEM linewidth measurement |
| Contrast | ≥0.7 | Intensity profile analysis |
| LER (3σ) | ≤10 nm | SEM + image processing |
| CD Uniformity | ±10% | Multiple site measurements |
| Alignment Accuracy | ≤1 μm | Vernier mark offset |

### 7.2 Comparison with Simulation
For each exposure condition, we will compare:
- Measured vs. predicted contrast
- CD vs. dose curves
- Proximity effect magnitude (dense vs. isolated CD difference)
- Resist response curves (remaining thickness vs. dose)

Success requires ≤20% deviation between simulation and experiment.

### 7.3 Go/No-Go Decision Points
**GO:** Achieve <300 nm features with acceptable uniformity and LER
**NO-GO:** Systematic deviations >30% from simulation, or insurmountable thermal/mechanical instability

---

## 8. Metrology and Data Collection

### 8.1 In-Situ Monitoring
- Beam flux monitoring via calibrated photodiode
- Temperature monitoring of mask holder
- Real-time dose calculation

### 8.2 Post-Exposure Characterization
**SEM Imaging:**
- Top-down view: CD, LER measurement
- Cross-section: Profile angle, resist loss
- Magnifications: 10kX - 100kX
- Minimum 10 measurement sites per pattern type

**Optical Profilometry:**
- Resist thickness maps
- Height uniformity
- Defect identification

**AFM (if available):**
- Surface roughness (Ra, Rq)
- Edge profile at sub-nm resolution

### 8.3 Data Products
All data will be provided in standardized formats:
- SEM images: TIFF (16-bit), with scale bars and metadata
- CD measurements: CSV tables with statistics
- Profilometry: Vendor native format + exported ASCII
- Analysis scripts: Python/MATLAB for reproducibility

---

## 9. Safety and Radiation Protection

### 9.1 Personnel Training
All personnel have completed:
- Radiation safety training (certificate attached)
- Beamline-specific orientation
- Chemical safety for resist handling

### 9.2 Operational Safety
- Dosimetry badges worn at all times
- Interlock systems verified before beam operation
- Emergency procedures reviewed
- Work area monitoring for ozone and chemical vapors

### 9.3 Sample Handling
- Resist-coated samples transported in light-tight containers
- No biological or radioactive materials
- Waste disposal per facility guidelines

---

## 10. Integration with CAM Fabrication (Part 1)

This XRL work directly leverages expertise from the tantalum CAM project:

**Synergies:**
1. **Mask Fabrication:** Ta absorber deposition and patterning techniques transfer directly
2. **Metrology:** SEM and profilometry methods identical
3. **Thermal Management:** Lessons from CAM thermal modeling apply to XRL masks
4. **Precision Alignment:** CAM integration jig methods inform XRL alignment strategy

**Future Integration:**
Successful XRL demonstration enables hybrid approach:
- XRL for ultra-fine features (<500 nm) on CAM
- Laser/EDM for larger structures (>10 μm)
- Combined process flow for next-generation X-ray optics

---

## 11. Budget and Resource Requirements

### 11.1 Beamtime Costs
- 10 hours beamtime × facility rate (per published schedule)

### 11.2 Consumables
| Item | Quantity | Cost (INR) |
|------|----------|------------|
| Silicon Wafers (100 mm) | 40 | 20,000 |
| PMMA Resist | 100 mL | 15,000 |
| ZEP520A | 50 mL | 25,000 |
| SU-8 2005 | 100 mL | 10,000 |
| Developers | As needed | 8,000 |
| Chemicals (IPA, acetone, etc.) | Various | 5,000 |
| **Total** | | **83,000** |

### 11.3 Facilities Access
- Cleanroom access for pre/post processing: 20 hours
- SEM time: 8 hours
- Profilometry: 4 hours

---

## 12. Timeline and Milestones

| Week | Activity | Deliverable |
|------|----------|-------------|
| 1 | Mask fabrication complete | GDS + fabricated masks |
| 2 | Resist process optimization | Spin curves, bake study |
| 3 | Pre-beamtime preparation | Sample prep, procedures |
| 4 | **Beamtime Execution** | **Exposed samples** |
| 5 | Metrology and analysis | SEM images, CD data |
| 6 | Data analysis and reporting | Final report, presentation |

---

## 13. Team and Qualifications

**Principal Investigator:** Abhineet Agarwal
- Graduate student, ME6110 Advanced Micro/Nanofabrication
- Experience: Laser micromachining, SEM, CAD/simulation
- Relevant coursework: Microfabrication, MEMS, Computational Methods

**Faculty Advisor:** Prof. Rakesh Mote
- Associate Professor, Mechanical Engineering, IIT Bombay
- Expertise: Precision manufacturing, MEMS, microfluidics
- Facility access: STAR Lab (microfabrication)

**Collaborators:**
- STAR Lab technical staff for cleanroom support
- Beamline scientists for X-ray expertise (to be identified)

---

## 14. Data Management and Publication Plan

### 14.1 Data Sharing
- All raw data archived on institutional storage (30 TB capacity)
- Anonymized datasets available upon request
- Open-source analysis scripts on GitHub

### 14.2 Publication Strategy
1. **Course Report:** ME6110 final project (December 2025)
2. **Conference Paper:** Target MNE 2026 or EIPBN 2026
3. **Journal Article:** Microelectronic Engineering (if results warrant)

### 14.3 Intellectual Property
- No patent applications anticipated (fundamental research)
- Results may inform future commercialization efforts in X-ray optics

---

## 15. Contingency Plans

### 15.1 Technical Risks
**Risk:** Mask damage or contamination  
**Mitigation:** Fabricate spare masks, gentle handling protocols

**Risk:** Resist process failures  
**Mitigation:** Pre-optimize on test wafers, have backup resist stocks

**Risk:** Beamline instability  
**Mitigation:** Redundant dose monitoring, flexible exposure matrix

### 15.2 Schedule Risks
**Risk:** Beamtime slot rescheduled  
**Mitigation:** Flexible team availability, stored resist-coated samples

**Risk:** Metrology equipment downtime  
**Mitigation:** External SEM access arranged, backup imaging facilities

---

## 16. Expected Impact

### 16.1 Scientific Contributions
- First comprehensive XRL dataset with modern resists at this facility
- Validated simulation methodology for XRL process design
- Benchmark for future XRL users

### 16.2 Educational Outcomes
- Advanced microfabrication training for graduate student
- Course project demonstrating theory-to-experiment cycle
- Potential lab module for future ME6110 offerings

### 16.3 Technological Impact
- Enable sub-micron X-ray optics for satellite payloads
- Demonstrate XRL viability for specialized MEMS applications
- Inform next-generation lithography tool development

---

## 17. References and Supporting Documents

### References (abbreviated)
1. Ghica & Fay, "LIGA and XRL Techniques," Microsystem Technologies (2020)
2. Fujita et al., "Deep X-ray Lithography," J. Micromech. Microeng. (2019)
3. Khan et al., "Compact X-ray Sources," Rev. Sci. Instrum. (2021)
4. Our Track B simulation report (attached)

### Attachments
- [ ] GDS layout files (xrl_test_patterns.gds)
- [ ] Mask fabrication drawings
- [ ] Resist process cards
- [ ] Safety training certificates
- [ ] Facility support letters
- [ ] Preliminary simulation results

---

## Contact Information

**Abhineet Agarwal**  
Department of Mechanical Engineering  
Indian Institute of Technology Bombay  
Mumbai, Maharashtra 400076  
Email: [student email]  
Phone: [contact number]

**Prof. Rakesh Mote** (Faculty Advisor)  
Department of Mechanical Engineering  
Indian Institute of Technology Bombay  
Email: [faculty email]  
Phone: [contact number]

---

**Proposal Prepared:** November 2025  
**Requested Beamtime:** 8-10 hours (flexible scheduling)  
**Preferred Dates:** [To be coordinated with facility]

---

*This proposal is submitted for consideration for X-ray beamtime allocation. We appreciate the opportunity to advance XRL research and education through this experimental program.*
