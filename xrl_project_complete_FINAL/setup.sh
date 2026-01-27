#!/bin/bash
################################################################################
# X-Ray Lithography Project Setup Script
# Author: Abhineet Agarwal
# Course: ME6110
# Date: November 2025
################################################################################

echo "========================================================================"
echo "X-RAY LITHOGRAPHY PROJECT - SETUP"
echo "========================================================================"
echo ""

# Check Python version
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed!"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✓ Found Python $PYTHON_VERSION"
echo ""

# Install required packages
echo "Installing required Python packages..."
echo "This may take a few minutes..."
echo ""

pip3 install --user numpy scipy matplotlib gdspy

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ All packages installed successfully"
else
    echo ""
    echo "WARNING: Some packages may have failed to install"
    echo "Try running: pip3 install --user numpy scipy matplotlib gdspy"
fi

echo ""
echo "========================================================================"
echo "SETUP COMPLETE"
echo "========================================================================"
echo ""
echo "Quick Start:"
echo ""
echo "1. Run all simulations:"
echo "   cd simulations/"
echo "   python3 run_all_simulations.py"
echo ""
echo "2. Generate GDS layouts:"
echo "   cd layouts/"
echo "   python3 generate_layouts.py"
echo ""
echo "3. View results:"
echo "   - Plots: data/*.png"
echo "   - Summary: data/simulation_summary.txt"
echo "   - GDS file: layouts/xrl_test_patterns.gds"
echo ""
echo "Documentation:"
echo "   - README.md - Complete project guide"
echo "   - QUICK_START.md - Quick reference"
echo "   - docs/beamtime_proposal.md - Experimental plan"
echo ""
echo "========================================================================"
