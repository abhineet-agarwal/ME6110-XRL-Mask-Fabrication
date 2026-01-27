# Process Integration Roadmap
## Linking CAM Fabrication Expertise with XRL Technology

**Project:** ME6110 Combined CAM + XRL Study  
**Author:** Abhineet Agarwal  
**Date:** November 2025

---

## 1. Executive Summary

This document outlines the technical and strategic integration between:
- **Part 1:** High-precision tantalum coded aperture mask (CAM) fabrication
- **Part 2:** X-ray lithography (XRL) feasibility study

While these appear as separate projects, they form a complementary capability set for advanced X-ray optics manufacturing. This roadmap identifies synergies, knowledge transfer pathways, and future integration opportunities.

---

## 2. Technology Synergies Matrix

| Technology Domain | CAM Fabrication | XRL Development | Integration Benefit |
|-------------------|-----------------|-----------------|---------------------|
| **Ta Processing** | Laser/EDM/etch of 0.5mm absorber | Sputter/pattern 0.5μm absorber | Process parameter transfer |
| **Metrology** | SEM, profilometry, CMM | SEM, profilometry, CD-SEM | Shared equipment & expertise |
| **Thermal Modeling** | Laser heating, warpage | X-ray heating, deflection | Common FEA methodology |
| **Precision Alignment** | Satellite jig integration | Mask-wafer alignment | Tolerance stack-up analysis |
| **Pattern Transfer** | Direct machining | Resist-based lithography | Complementary resolution ranges |

---

## 3. Process Flow Integration

### 3.1 Hybrid Manufacturing Approach

**Concept:** Combine direct machining (CAM) with lithography (XRL) for multi-scale features.

```
┌─────────────────────────────────────────────────────────┐
│  HYBRID X-RAY OPTICS FABRICATION FLOW                  │
└─────────────────────────────────────────────────────────┘

SUBSTRATE PREPARATION
├─ Si₃N₄ membrane (LPCVD, 2μm, low stress)
└─ Frame: Silicon, 525μm thick

COARSE FEATURE DEFINITION (>10μm)
├─ Laser micromachining (fs laser, CAM expertise)
│  └─ Alignment fiducials, frame cutouts
└─ Advantages: Fast, direct write, no resist

FINE FEATURE DEFINITION (0.1-10μm)
├─ Ta deposition (sputtered, 0.5-1.0μm)
├─ X-ray resist coating (PMMA/ZEP)
├─ XRL exposure (1.5 keV, optimized dose)
├─ Resist development
└─ Ta etch (RIE or wet)

POST-PROCESSING
├─ Resist strip
├─ SEM inspection
└─ Integration into payload

ADVANTAGES:
• Sub-micron XRL resolution where needed
• Rapid laser patterning for non-critical features
• Reduced XRL exposure time
• Flexible design iteration
```

### 3.2 Example Application: Next-Gen CAM

**Scenario:** Satellite X-ray telescope coded aperture with mixed feature scales.

| Feature Type | Size Range | Method | Rationale |
|--------------|------------|--------|-----------|
| Fine pixels | 0.2-1 μm | XRL | Resolution, uniformity |
| Alignment marks | 10-20 μm | Laser | Speed, robustness |
| Frame cutouts | >100 μm | EDM/laser | Structural integrity |
| Support struts | 5-50 μm | Combined | Hybrid approach |

---

## 4. Knowledge Transfer Pathways

### 4.1 CAM → XRL
**What CAM Teaches XRL:**

1. **Ta Material Properties**
   - Sputter deposition parameters (pressure, power, rate)
   - Adhesion to Si₃N₄ (adhesion layers if needed)
   - Stress management (annealing, multilayers)
   - Etch selectivity and profiles

2. **Thermal Management**
   - Heat dissipation in thin membranes
   - Thermal expansion mismatch (Ta vs. Si₃N₄)
   - Deflection prediction under heat load
   - Steady-state temperature calculations

3. **Precision Metrology**
   - SEM imaging protocols (voltage, tilt, dose)
   - CD measurement methodology (multiple sites, statistics)
   - Edge detection algorithms
   - Profilometry for film stress

4. **Tolerance Analysis**
   - Sub-10μm accuracy requirements
   - Error budget allocation
   - Cumulative tolerance stack-up
   - Acceptance criteria definition

### 4.2 XRL → CAM
**What XRL Teaches CAM:**

1. **Nanoscale Resolution**
   - Sub-micron patterning capability
   - Potential for finer CAM pixels (current: 10μm → future: 0.5μm)
   - Pattern fidelity improvement

2. **Process Optimization Methodology**
   - Simulation-driven parameter selection
   - DOE for exposure optimization
   - Statistical process control
   - Yield enhancement strategies

3. **Advanced Metrology**
   - Line-edge roughness quantification
   - Contrast measurement techniques
   - Proximity effect characterization
   - CD-SEM best practices

4. **Resist Chemistry**
   - Polymer behavior under radiation
   - Developer kinetics
   - Selectivity enhancement
   - Potential resist-based sacrificial processes

---

## 5. Integrated Facility Requirements

### 5.1 Cleanroom Capabilities
To support hybrid CAM+XRL:

| Facility | Class | Usage | Shared Equipment |
|----------|-------|-------|------------------|
| Photobay | 1000 | Lithography, XRL mask prep | Spinners, hotplates |
| Main cleanroom | 100 | Ta deposition, etching | Sputterer, RIE |
| Metrology bay | - | Inspection | SEM, profilometer, CMM |
| Laser lab | - | Direct machining | fs laser, EDM |

### 5.2 Equipment Sharing Strategy
- **SEM:** Schedule blocks for CAM (thick samples) and XRL (resist profiles)
- **Profilometer:** CAM stress, XRL thickness mapping
- **Sputter Tool:** CAM thick Ta, XRL thin absorber (different targets/conditions)

---

## 6. Simulation-to-Fabrication Workflow

### 6.1 Unified Modeling Framework

```
DESIGN INPUTS
├─ CAM: Pixel layout, absorption requirements
└─ XRL: Feature size, dose requirements

SIMULATION SUITE
├─ Aerial image (Python) → Predicts XRL contrast
├─ Resist response (Python) → Estimates CD, LER
├─ Thermal-mechanical (COMSOL/Python) → Deflection, stress
└─ Dose modeling → Exposure time calculation

FABRICATION PLANNING
├─ CAM: Laser/EDM parameters from thermal limits
├─ XRL: Energy, gap, dose from simulation
└─ Metrology: Predicted vs. measured comparison

POST-FABRICATION VALIDATION
├─ SEM measurements → Update model calibration
├─ Performance testing → Satellite integration
└─ Lessons learned → Next iteration
```

### 6.2 Closed-Loop Optimization

1. **Initial Design:** Use simulation to predict optimal parameters
2. **Fabricate:** Small-scale test coupons
3. **Measure:** SEM, profilometry, functional test
4. **Compare:** Simulation vs. experiment
5. **Update Models:** Improve material constants, boundary conditions
6. **Iterate:** Full-scale production with refined parameters

---

## 7. Risk Mitigation Through Integration

### 7.1 Technical Risks

| Risk | CAM Approach | XRL Approach | Integrated Mitigation |
|------|--------------|--------------|----------------------|
| Feature size limits | Laser: ~10μm | XRL: ~0.1μm | Use appropriate tool per feature |
| Thermal damage | Pulse duration control | Beam power limit | Both use thermal modeling |
| Alignment errors | Jig-based | Mask aligner | Learn from both systems |
| Burr/edge defects | EDM, etch | Resist sidewall | Cross-apply solutions |

### 7.2 Schedule Risks
- **CAM delays** don't block XRL modeling/layout work
- **XRL beamtime availability** doesn't affect CAM characterization
- Parallel paths enable overall project completion on time

---

## 8. Future Research Directions

### 8.1 Near-Term (6 months)
1. Demonstrate XRL on 0.5μm features using test masks
2. Complete CAM fabrication and satellite integration
3. Publish combined results in ME6110 report

### 8.2 Medium-Term (1-2 years)
1. Hybrid CAM with XRL fine pixels and laser-cut frame
2. Multi-layer XRL for 3D structures
3. Compact X-ray source evaluation (tabletop XRL)

### 8.3 Long-Term (3-5 years)
1. Commercialization of hybrid X-ray optics
2. Advanced MEMS via XRL (high aspect ratio)
3. Integration with other lithography (e-beam, DUV) for multi-scale devices

---

## 9. Bill of Materials (BOM) - Integrated Process

### 9.1 Substrate/Membrane
| Item | Specification | Quantity | Source | Cost (INR) |
|------|---------------|----------|--------|------------|
| Si₃N₄ membrane wafers | 100mm, 2μm LPCVD | 25 | Norcada/SiMPore | 125,000 |
| Silicon frames | 525μm, <100> | 50 | Vendor A | 25,000 |

### 9.2 Absorber Material
| Item | Specification | Quantity | Source | Cost (INR) |
|------|---------------|----------|--------|------------|
| Ta target (thick) | 3" dia, 99.95% | 1 | Kurt J. Lesker | 80,000 |
| Ta target (thin) | 2" dia, 99.95% | 1 | Kurt J. Lesker | 40,000 |

### 9.3 Resists (XRL)
| Item | Specification | Quantity | Source | Cost (INR) |
|------|---------------|----------|--------|------------|
| PMMA 950K A4 | 500mL | 1 bottle | MicroChem | 30,000 |
| ZEP520A | 100mL | 1 bottle | Zeon | 50,000 |
| SU-8 2005 | 500mL | 1 bottle | MicroChem | 20,000 |

### 9.4 Processing Chemicals
| Item | Use | Quantity | Cost (INR) |
|------|-----|----------|------------|
| MIBK | PMMA developer | 1L | 3,000 |
| IPA | Rinse | 5L | 2,000 |
| ZED-N50 | ZEP developer | 100mL | 8,000 |
| Acetone | Cleaning | 5L | 2,000 |

### 9.5 Gases
| Item | Use | Source | Cost (INR/month) |
|------|-----|--------|------------------|
| Ar | Sputtering | Facility supply | Included |
| N₂ | Purge, dry | Facility supply | Included |
| CF₄/O₂ | RIE etch | Cylinder | 10,000 |

**Total Integrated BOM:** ~400,000 INR

---

## 10. Deliverables and Milestones

### 10.1 Combined Project Deliverables

| Deliverable | Part 1 (CAM) | Part 2 (XRL) | Integration Value |
|-------------|--------------|--------------|-------------------|
| Fabricated samples | Ta CAM, 0.5mm | XRL test patterns | Compare methods |
| Metrology data | CMM, SEM, profilometry | SEM, AFM, CD-SEM | Unified analysis |
| Process documentation | Laser/EDM procedures | XRL exposure/develop | Hybrid workflow |
| Simulation models | Thermal FEA | Aerial image, resist | Cross-validation |
| Final report | CAM characterization | XRL feasibility | Integrated conclusions |

### 10.2 Milestone Schedule

```
Week 1 (8-14 Nov)
├─ CAM: Design finalization, fab trial #1
└─ XRL: Literature review, Python simulation setup

Week 2 (15-21 Nov)
├─ CAM: Fab trials #2-3, metrology
└─ XRL: Simulation sweeps, COMSOL thermal

Week 3 (22-28 Nov)
├─ CAM: Final samples, integration prep
├─ XRL: GDS layouts, beamtime proposal
└─ INTEGRATION: Combined report drafting

Week 4 (Post-deadline)
├─ Beamtime execution (if scheduled)
├─ Final analysis and comparison
└─ Presentation preparation
```

---

## 11. Success Metrics

### 11.1 Individual Project Success

**CAM (Part 1):**
- ✓ Dimensional tolerance <10μm
- ✓ Burr-free edges
- ✓ Warpage <50μm
- ✓ Manufacturability documentation

**XRL (Part 2):**
- ✓ Simulation-experiment agreement <20% deviation
- ✓ Demonstrated <500nm features
- ✓ Process window established
- ✓ Beamtime proposal ready

### 11.2 Integration Success
- ✓ Identified ≥3 technology transfer opportunities
- ✓ Hybrid process flow documented
- ✓ Combined BOM and cost analysis
- ✓ Roadmap for future work
- ✓ Unified final presentation

---

## 12. Lessons Learned Framework

After project completion, capture:

1. **What Worked Well**
   - Which simulation predictions were accurate?
   - Which fabrication methods were most reliable?
   - What synergies were most valuable?

2. **What Could Improve**
   - Where did simulation deviate from reality?
   - What process steps need optimization?
   - Which integrations were less useful than expected?

3. **Unexpected Findings**
   - Novel failure modes
   - Serendipitous process interactions
   - New research questions

4. **Actionable Recommendations**
   - For future students doing similar work
   - For STAR Lab facility upgrades
   - For curriculum development in ME6110

---

## 13. Conclusion

The integration of CAM fabrication expertise with XRL feasibility research creates a unique capability set for advanced X-ray optics manufacturing. By treating these as complementary rather than separate projects, we maximize:

- **Technical Learning:** Cross-pollination of methods and insights
- **Resource Efficiency:** Shared equipment, facilities, and expertise
- **Research Impact:** Broader applicability and publication potential
- **Educational Value:** Holistic view of micro/nanofabrication

This roadmap provides a clear path for leveraging both projects toward a common goal: advancing the state of the art in precision X-ray optics for satellite payloads and beyond.

---

**Document Prepared By:** Abhineet Agarwal  
**Course:** ME6110 Advanced Micro/Nanofabrication  
**Instructor:** Prof. Rakesh Mote  
**Date:** November 2025

---

## Appendices

### Appendix A: Acronyms and Abbreviations
- CAM: Coded Aperture Mask
- XRL: X-ray Lithography
- EDM: Electrical Discharge Machining
- RIE: Reactive Ion Etching
- SEM: Scanning Electron Microscopy
- CD: Critical Dimension
- LER: Line-Edge Roughness
- DOE: Design of Experiments
- FEA: Finite Element Analysis

### Appendix B: References
1. ME6110 Course Notes (2025)
2. CAM Project Proposal (November 2025)
3. XRL Simulation Results (Track B deliverable)
4. Industry standards: SEMI, ISO microfabrication

### Appendix C: Contact Information
[Same as beamtime proposal]
