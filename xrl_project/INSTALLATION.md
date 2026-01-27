# Installation and Running Instructions

## System Requirements

### Minimum Requirements:
- **Operating System:** Windows 10/11, macOS 10.14+, or Linux (Ubuntu 20.04+)
- **Python:** Version 3.8 or higher
- **RAM:** 4 GB (8 GB recommended for large simulations)
- **Disk Space:** 500 MB for code and results
- **Display:** 1920×1080 or higher (for viewing plots)

### Required Software:
1. **Python 3.8+** - Download from [python.org](https://www.python.org/downloads/)
2. **pip** - Usually included with Python
3. **GDS Viewer** (optional) - KLayout, Cadence Virtuoso, or any GDSII viewer

---

## Installation Methods

### Method 1: Automated Setup (Recommended)

#### On Linux/macOS:
```bash
# Navigate to project directory
cd xrl_project/

# Run setup script
bash setup.sh

# Verify installation
python3 -c "import numpy, scipy, matplotlib, gdspy; print('All packages OK!')"
```

#### On Windows:
```cmd
REM Navigate to project directory
cd xrl_project

REM Run setup script
setup.bat

REM Verify installation
python -c "import numpy, scipy, matplotlib, gdspy; print('All packages OK!')"
```

### Method 2: Manual Installation

#### Step 1: Install Python packages
```bash
# Using pip (Linux/macOS)
pip3 install --user numpy scipy matplotlib gdspy

# Using pip (Windows)
python -m pip install --user numpy scipy matplotlib gdspy
```

#### Step 2: Verify installation
```bash
python3 -c "import numpy; print(f'NumPy {numpy.__version__}')"
python3 -c "import scipy; print(f'SciPy {scipy.__version__}')"
python3 -c "import matplotlib; print(f'Matplotlib {matplotlib.__version__}')"
python3 -c "import gdspy; print(f'gdspy {gdspy.__version__}')"
```

### Method 3: Using requirements.txt
```bash
pip3 install --user -r requirements.txt
```

---

## Running the Simulations

### Complete Simulation Suite (Track B)

This runs all three physics modules and generates comprehensive results.

```bash
cd simulations/
python3 run_all_simulations.py
```

**Expected Output:**
- Console output with progress updates (~180 seconds runtime)
- 3 PNG plots saved to `../data/`:
  - `aerial_image_comprehensive.png` (6 subplots)
  - `resist_response_comprehensive.png` (4 subplots)
  - `thermal_mechanical_comprehensive.png` (4 subplots)
- `simulation_summary.txt` with key findings

**Typical Runtime:** 2-3 minutes on modern hardware

### Individual Modules

You can also run each module independently for testing:

#### Aerial Image Simulation:
```bash
cd simulations/
python3 aerial_image.py
```
- Generates example aerial image plot
- Shows contrast and resolution for default parameters

#### Resist Response Simulation:
```bash
cd simulations/
python3 resist_response.py
```
- Generates resist exposure example
- Shows dose profile and developed pattern

#### Thermal-Mechanical Analysis:
```bash
cd simulations/
python3 thermal_mechanical.py
```
- Generates thermal analysis plots
- Shows deflection vs beam power

---

## Generating GDS Layouts (Track C)

```bash
cd layouts/
python3 generate_layouts.py
```

**Expected Output:**
- Console output describing patterns being created
- `xrl_test_patterns.gds` - Binary GDSII file
- `xrl_test_patterns_report.txt` - Pattern documentation

**Viewing the GDS file:**
- **KLayout** (free): File → Open → xrl_test_patterns.gds
- **Any GDSII viewer**: Import as GDSII format
- **Online viewer**: [gdspy documentation](https://gdspy.readthedocs.io)

---

## Customizing Simulations

### Modify Parameters

Edit the main runner script to change simulation parameters:

```python
# In simulations/run_all_simulations.py

# Change photon energy
self.energy_kev = 2.0  # Default: 1.5 keV

# Change mask-resist gap
self.gap_um = 15.0  # Default: 10.0 μm

# Change feature size
self.feature_size_um = 0.3  # Default: 0.5 μm

# Modify sweep ranges
energies = np.linspace(0.8, 3.0, 30)  # More points
gaps = np.linspace(5, 30, 20)  # Different range
```

### Run Specific Parameter Sweeps

```python
from aerial_image import XRayMask, parameter_sweep_energy

# Create mask
mask = XRayMask(
    absorber_material='Ta',
    absorber_thickness=0.8,  # Thicker absorber
    membrane_material='Si3N4',
    membrane_thickness=2.0,
    feature_size=0.3,  # Smaller features
    pitch=0.6
)

# Run energy sweep
import numpy as np
energies = np.linspace(1.0, 2.5, 15)
results = parameter_sweep_energy(mask, gap=15.0, energies=energies)

# Access results
print(f"Optimal contrast: {max(results['contrast'])}")
```

---

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'numpy'"
**Solution:** 
```bash
pip3 install --user numpy scipy matplotlib gdspy
# or
python -m pip install --user numpy scipy matplotlib gdspy
```

### Problem: "Permission denied" when running setup.sh
**Solution:**
```bash
chmod +x setup.sh
./setup.sh
```

### Problem: Plots not displaying
**Solution:**
- Check if matplotlib backend is configured correctly
- Plots are saved to `data/` directory even if display fails
- View saved PNG files directly

### Problem: "RuntimeWarning: invalid value encountered"
**Solution:**
- This is normal for some parameter combinations
- Results with `nan` values indicate invalid conditions
- The code handles these gracefully

### Problem: GDS file won't open in viewer
**Solution:**
- Ensure you're using a GDSII-compatible viewer (KLayout recommended)
- File should be ~150 KB in size
- Check file wasn't corrupted during download

### Problem: Simulations run slowly
**Solution:**
- Expected runtime: 2-3 minutes total
- Reduce number of points in parameter sweeps
- Comment out unused sweeps in run_all_simulations.py

---

## Verifying Results

### Expected Simulation Results:

**Aerial Image Analysis:**
- Contrast values between 0.1 and 1.0
- Resolution values in μm scale
- Clear trends vs energy, gap, thickness

**Resist Response:**
- CD values in 0.3-0.7 μm range (for 0.5 μm features)
- LER values in nm scale
- Dose response curves showing threshold behavior

**Thermal-Mechanical:**
- Temperature rise in K (single digits for <1W)
- Deflection in μm (varies by material)
- Diamond shows lowest deflection

### Comparing with Literature:

Expected values align with published XRL studies:
- Contrast: 0.5-0.9 typical for Ta absorbers
- Resolution: ~0.5× feature size for Fresnel regime
- LER: 3-10 nm for modern resists

---

## File Locations After Running

```
xrl_project/
├── simulations/
│   ├── *.py (source code)
│   └── __pycache__/ (compiled bytecode)
│
├── layouts/
│   ├── generate_layouts.py
│   ├── xrl_test_patterns.gds (NEW)
│   └── xrl_test_patterns_report.txt (NEW)
│
└── data/
    ├── aerial_image_comprehensive.png (NEW)
    ├── resist_response_comprehensive.png (NEW)
    ├── thermal_mechanical_comprehensive.png (NEW)
    └── simulation_summary.txt (NEW)
```

---

## Advanced Usage

### Running in Jupyter Notebook

```bash
# Install Jupyter if not already installed
pip3 install --user jupyter

# Start Jupyter
cd simulations/
jupyter notebook
```

Create new notebook and import modules:
```python
from aerial_image import *
from resist_response import *
from thermal_mechanical import *

# Interactive parameter exploration
mask = XRayMask(absorber_material='Ta', absorber_thickness=0.5, ...)
sim = AerialImageSimulator(mask, gap=10.0)
x, intensity = sim.compute_aerial_image(energy_kev=1.5)

import matplotlib.pyplot as plt
plt.plot(x, intensity)
plt.show()
```

### Batch Processing

Create a script to run multiple configurations:

```python
# batch_run.py
import numpy as np
from run_all_simulations import IntegratedXRLSimulation

# Run for different feature sizes
for feature_size in [0.2, 0.5, 1.0, 2.0]:
    print(f"\n Running for {feature_size} μm features...")
    sim = IntegratedXRLSimulation()
    sim.feature_size_um = feature_size
    sim.run_all()
```

---

## Getting Help

### Documentation Files:
- **README.md** - Comprehensive project overview
- **QUICK_START.md** - Quick reference guide
- **FILES_INCLUDED.txt** - Complete file manifest
- **docs/beamtime_proposal.md** - Experimental details
- **docs/integration_roadmap.md** - Integration with CAM project

### Inline Documentation:
- All Python modules have extensive docstrings
- Each function includes parameter descriptions
- Example usage provided in `if __name__ == "__main__":` blocks

### Common Questions:

**Q: Can I modify the code for my own project?**
A: Yes! All code is provided for educational purposes. Modify as needed.

**Q: What if I want different materials?**
A: Add entries to MATERIALS or MEMBRANE_MATERIALS dictionaries in the respective modules.

**Q: Can I export results to Excel/CSV?**
A: Yes, add code like:
```python
import pandas as pd
df = pd.DataFrame(results)
df.to_csv('results.csv', index=False)
```

**Q: How do I cite this work?**
A: Academic citation:
```
Agarwal, A. (2025). X-Ray Lithography Feasibility Study: 
Modeling, Simulation, and Prototyping. ME6110 Advanced 
Micro/Nanofabrication, IIT Bombay.
```

---

## Next Steps

After successfully running the code:

1. **Review Results** - Check all plots in `data/` directory
2. **Customize Parameters** - Modify for your specific requirements
3. **Prepare for Beamtime** - Use `docs/beamtime_proposal.md` template
4. **Experimental Validation** - Compare simulation predictions with actual exposures
5. **Iterate** - Update models based on experimental data

---

## Support

For technical issues:
- Check troubleshooting section above
- Review inline code comments
- Ensure all dependencies are installed correctly
- Verify Python version compatibility

For scientific questions:
- Consult referenced literature in beamtime proposal
- Review simulation methodology in code comments
- Compare with published XRL studies

---

**Project:** ME6110 X-Ray Lithography Study  
**Author:** Abhineet Agarwal  
**Date:** November 2025  
**Status:** Ready to Run ✓

---

Happy simulating! 🚀
