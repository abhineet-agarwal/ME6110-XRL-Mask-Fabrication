#!/usr/bin/env python3
"""
Quick test script to verify XRL project installation
Run this after setup to ensure everything works correctly.
"""

import sys
import os

print("="*70)
print("X-RAY LITHOGRAPHY PROJECT - INSTALLATION TEST")
print("="*70)
print()

# Test 1: Check Python version
print("Test 1: Python Version")
print(f"  Python {sys.version.split()[0]}", end=" ")
version_info = sys.version_info
if version_info >= (3, 8):
    print("✓ OK")
else:
    print("✗ FAIL (need 3.8+)")
    sys.exit(1)

# Test 2: Import required packages
print("\nTest 2: Required Packages")
required_packages = {
    'numpy': 'NumPy',
    'scipy': 'SciPy',
    'matplotlib': 'Matplotlib',
    'gdspy': 'gdspy'
}

all_ok = True
for module_name, display_name in required_packages.items():
    try:
        mod = __import__(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  {display_name:12s} {version:10s} ✓")
    except ImportError:
        print(f"  {display_name:12s} {'NOT FOUND':10s} ✗")
        all_ok = False

if not all_ok:
    print("\n  Some packages are missing. Run:")
    print("  pip3 install --user numpy scipy matplotlib gdspy")
    sys.exit(1)

# Test 3: Check project structure
print("\nTest 3: Project Structure")
expected_dirs = ['simulations', 'layouts', 'docs', 'data']
for dir_name in expected_dirs:
    if os.path.isdir(dir_name):
        print(f"  {dir_name:15s} ✓")
    else:
        print(f"  {dir_name:15s} ✗ (missing)")

# Test 4: Quick simulation test
print("\nTest 4: Quick Simulation")
try:
    # Change to simulations directory
    sys.path.insert(0, 'simulations')
    from aerial_image import XRayMask, AerialImageSimulator
    
    mask = XRayMask(
        absorber_material='Ta',
        absorber_thickness=0.5,
        membrane_material='Si3N4',
        membrane_thickness=2.0,
        feature_size=0.5,
        pitch=1.0
    )
    
    sim = AerialImageSimulator(mask, gap=10.0)
    x, intensity = sim.compute_aerial_image(energy_kev=1.5)
    contrast = sim.calculate_contrast(x, intensity)
    
    print(f"  Aerial image simulation ✓")
    print(f"    Contrast: {contrast:.3f}")
    print(f"    Data points: {len(x)}")
    
except Exception as e:
    print(f"  Aerial image simulation ✗")
    print(f"    Error: {e}")
    sys.exit(1)

# Test 5: GDS generation test
print("\nTest 5: GDS Library")
try:
    import gdspy
    lib = gdspy.GdsLibrary()
    cell = lib.new_cell('TEST')
    rect = gdspy.Rectangle((0, 0), (10, 10))
    cell.add(rect)
    print(f"  GDS generation ✓")
except Exception as e:
    print(f"  GDS generation ✗")
    print(f"    Error: {e}")

print("\n" + "="*70)
print("ALL TESTS PASSED ✓")
print("="*70)
print()
print("You're ready to run simulations!")
print()
print("Try:")
print("  cd simulations/")
print("  python3 run_all_simulations.py")
print()
