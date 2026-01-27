"""
GDS Layout Generator for X-ray Lithography Test Patterns
========================================================

Generates GDSII layouts for XRL exposure test patterns including:
- Dense line/space arrays
- Hole arrays
- Resolution targets
- Alignment marks

Author: Abhineet Agarwal
Course: ME6110
Date: November 2025
"""

import gdspy
import numpy as np
from typing import Tuple, List, Optional


class XRLLayoutDesigner:
    """
    Generates test pattern layouts for X-ray lithography validation.
    """
    
    def __init__(self, library_name: str = "XRL_TEST_PATTERNS"):
        self.lib = gdspy.GdsLibrary()
        self.lib.name = library_name  # Keep as string, gdspy will handle encoding
        self.unit = 1e-6  # 1 micron base unit
        self.precision = 1e-9  # 1 nm precision
        
    def create_line_space_array(self,
                                cell_name: str,
                                pitch: float,
                                linewidth: float,
                                array_width: float = 100.0,
                                array_height: float = 100.0,
                                layer: int = 1) -> gdspy.Cell:
        """
        Create dense line/space test pattern.
        
        Args:
            cell_name: Name of the cell
            pitch: Line pitch (μm)
            linewidth: Line width (μm)
            array_width: Total array width (μm)
            array_height: Total array height (μm)
            layer: GDSII layer number
        
        Returns:
            GDSII cell
        """
        cell = self.lib.new_cell(cell_name)
        
        n_lines = int(array_width / pitch)
        
        for i in range(n_lines):
            x = i * pitch
            rect = gdspy.Rectangle(
                (x, 0),
                (x + linewidth, array_height),
                layer=layer
            )
            cell.add(rect)
        
        return cell
    
    def create_hole_array(self,
                         cell_name: str,
                         hole_diameter: float,
                         pitch: float,
                         array_size: int = 10,
                         layer: int = 1) -> gdspy.Cell:
        """
        Create 2D array of circular holes.
        
        Args:
            cell_name: Name of the cell
            hole_diameter: Hole diameter (μm)
            pitch: Center-to-center spacing (μm)
            array_size: Number of holes in each direction
            layer: GDSII layer number
        
        Returns:
            GDSII cell
        """
        cell = self.lib.new_cell(cell_name)
        
        radius = hole_diameter / 2
        
        for i in range(array_size):
            for j in range(array_size):
                x = i * pitch
                y = j * pitch
                circle = gdspy.Round(
                    (x, y),
                    radius,
                    number_of_points=64,
                    layer=layer
                )
                cell.add(circle)
        
        return cell
    
    def create_resolution_target(self,
                                cell_name: str,
                                min_feature: float = 0.1,
                                max_feature: float = 2.0,
                                n_steps: int = 10,
                                layer: int = 1) -> gdspy.Cell:
        """
        Create resolution test target with varying feature sizes.
        
        Args:
            cell_name: Name of the cell
            min_feature: Minimum feature size (μm)
            max_feature: Maximum feature size (μm)
            n_steps: Number of feature size steps
            layer: GDSII layer number
        
        Returns:
            GDSII cell
        """
        cell = self.lib.new_cell(cell_name)
        
        feature_sizes = np.linspace(min_feature, max_feature, n_steps)
        
        y_offset = 0
        for size in feature_sizes:
            pitch = size * 2
            
            # Create 5 lines at this size
            for i in range(5):
                x = i * pitch
                rect = gdspy.Rectangle(
                    (x, y_offset),
                    (x + size, y_offset + 50),
                    layer=layer
                )
                cell.add(rect)
            
            # Add text label
            text = gdspy.Text(
                f"{size:.2f}um",
                5,
                (60, y_offset + 20),
                layer=layer
            )
            cell.add(text)
            
            y_offset += 60
        
        return cell
    
    def create_alignment_marks(self,
                              cell_name: str,
                              mark_size: float = 20.0,
                              cross_width: float = 2.0,
                              layer: int = 2) -> gdspy.Cell:
        """
        Create alignment cross marks.
        
        Args:
            cell_name: Name of the cell
            mark_size: Size of alignment cross (μm)
            cross_width: Width of cross arms (μm)
            layer: GDSII layer number
        
        Returns:
            GDSII cell
        """
        cell = self.lib.new_cell(cell_name)
        
        # Horizontal arm
        h_arm = gdspy.Rectangle(
            (-mark_size/2, -cross_width/2),
            (mark_size/2, cross_width/2),
            layer=layer
        )
        
        # Vertical arm
        v_arm = gdspy.Rectangle(
            (-cross_width/2, -mark_size/2),
            (cross_width/2, mark_size/2),
            layer=layer
        )
        
        cell.add(h_arm)
        cell.add(v_arm)
        
        return cell
    
    def create_elmore_pattern(self,
                             cell_name: str,
                             base_pitch: float = 1.0,
                             layer: int = 1) -> gdspy.Cell:
        """
        Create Elmore pattern for proximity effect characterization.
        
        Alternating isolated and dense line regions.
        """
        cell = self.lib.new_cell(cell_name)
        
        # Dense region (5 lines)
        for i in range(5):
            x = i * base_pitch
            rect = gdspy.Rectangle(
                (x, 0),
                (x + base_pitch/2, 50),
                layer=layer
            )
            cell.add(rect)
        
        # Isolated line
        iso_x = 10 + base_pitch * 10
        iso_rect = gdspy.Rectangle(
            (iso_x, 0),
            (iso_x + base_pitch/2, 50),
            layer=layer
        )
        cell.add(iso_rect)
        
        # Another dense region
        for i in range(5):
            x = iso_x + 10 + i * base_pitch
            rect = gdspy.Rectangle(
                (x, 0),
                (x + base_pitch/2, 50),
                layer=layer
            )
            cell.add(rect)
        
        return cell
    
    def create_contact_hole_shrink_test(self,
                                       cell_name: str,
                                       nominal_size: float = 0.5,
                                       bias_range: float = 0.2,
                                       n_steps: int = 9,
                                       layer: int = 1) -> gdspy.Cell:
        """
        Create contact hole size variation test.
        """
        cell = self.lib.new_cell(cell_name)
        
        sizes = np.linspace(nominal_size - bias_range, 
                           nominal_size + bias_range, 
                           n_steps)
        
        pitch = 5.0
        
        for i, size in enumerate(sizes):
            x = i * pitch
            circle = gdspy.Round(
                (x, 0),
                size / 2,
                number_of_points=64,
                layer=layer
            )
            cell.add(circle)
            
            # Label
            text = gdspy.Text(
                f"{size:.2f}",
                2,
                (x - 1, -5),
                layer=layer
            )
            cell.add(text)
        
        return cell
    
    def create_master_layout(self,
                            output_file: str,
                            die_size: Tuple[float, float] = (10000, 10000)) -> str:
        """
        Create master layout with all test patterns.
        
        Args:
            output_file: Output GDS file path
            die_size: Die size (width, height) in μm
        
        Returns:
            Path to generated GDS file
        """
        master = self.lib.new_cell("MASTER_DIE")
        
        # Create individual patterns
        print("Creating test patterns...")
        
        # 1. Line/space arrays at different pitches
        pitches = [0.2, 0.5, 1.0, 2.0]
        y_offset = 100
        for pitch in pitches:
            linewidth = pitch / 2
            cell = self.create_line_space_array(
                f"LS_P{pitch:.1f}",
                pitch, linewidth,
                array_width=100, array_height=100,
                layer=1
            )
            ref = gdspy.CellReference(cell, (100, y_offset))
            master.add(ref)
            y_offset += 150
            print(f"  - Line/space array: pitch={pitch} μm")
        
        # 2. Hole arrays
        hole_sizes = [0.3, 0.5, 1.0]
        x_offset = 300
        y_offset = 100
        for hole_size in hole_sizes:
            cell = self.create_hole_array(
                f"HOLES_D{hole_size:.1f}",
                hole_size, hole_size * 2.5,
                array_size=10, layer=1
            )
            ref = gdspy.CellReference(cell, (x_offset, y_offset))
            master.add(ref)
            y_offset += 150
            print(f"  - Hole array: diameter={hole_size} μm")
        
        # 3. Resolution target
        resolution_cell = self.create_resolution_target(
            "RESOLUTION_TARGET",
            min_feature=0.1, max_feature=2.0,
            n_steps=10, layer=1
        )
        resolution_ref = gdspy.CellReference(resolution_cell, (600, 100))
        master.add(resolution_ref)
        print("  - Resolution target")
        
        # 4. Elmore pattern
        elmore_cell = self.create_elmore_pattern(
            "ELMORE_PATTERN", base_pitch=1.0, layer=1
        )
        elmore_ref = gdspy.CellReference(elmore_cell, (800, 100))
        master.add(elmore_ref)
        print("  - Elmore proximity test")
        
        # 5. Contact hole shrink test
        contact_cell = self.create_contact_hole_shrink_test(
            "CONTACT_SHRINK",
            nominal_size=0.5, bias_range=0.2,
            n_steps=9, layer=1
        )
        contact_ref = gdspy.CellReference(contact_cell, (1000, 100))
        master.add(contact_ref)
        print("  - Contact hole shrink test")
        
        # 6. Alignment marks at corners
        align_cell = self.create_alignment_marks(
            "ALIGNMENT", mark_size=20, cross_width=2, layer=2
        )
        
        # Place at four corners
        corners = [
            (50, 50),
            (die_size[0] - 50, 50),
            (50, die_size[1] - 50),
            (die_size[0] - 50, die_size[1] - 50)
        ]
        for corner in corners:
            align_ref = gdspy.CellReference(align_cell, corner)
            master.add(align_ref)
        print("  - Alignment marks (4 corners)")
        
        # Add die outline
        outline = gdspy.Rectangle(
            (0, 0),
            die_size,
            layer=0
        )
        master.add(outline)
        
        # Write to file
        self.lib.write_gds(output_file)
        print(f"\nGDS layout written to: {output_file}")
        
        return output_file
    
    def generate_pattern_report(self, output_file: str) -> str:
        """Generate text report describing all patterns in layout"""
        report_path = output_file.replace('.gds', '_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("X-RAY LITHOGRAPHY TEST PATTERN LAYOUT REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("LAYOUT INFORMATION:\n")
            f.write(f"  Library: {self.lib.name}\n")
            f.write(f"  Number of cells: {len(self.lib.cells)}\n")
            f.write(f"  Unit: {self.unit} m\n")
            f.write(f"  Precision: {self.precision} m\n\n")
            
            f.write("-"*70 + "\n")
            f.write("TEST PATTERNS INCLUDED:\n")
            f.write("-"*70 + "\n\n")
            
            f.write("1. LINE/SPACE ARRAYS:\n")
            f.write("   Purpose: Contrast and resolution characterization\n")
            f.write("   Pitches: 0.2, 0.5, 1.0, 2.0 μm\n")
            f.write("   Duty cycle: 50%\n")
            f.write("   Array size: 100 × 100 μm\n\n")
            
            f.write("2. HOLE ARRAYS:\n")
            f.write("   Purpose: Contact hole pattern fidelity\n")
            f.write("   Diameters: 0.3, 0.5, 1.0 μm\n")
            f.write("   Pitch: 2.5× diameter\n")
            f.write("   Array: 10 × 10 holes\n\n")
            
            f.write("3. RESOLUTION TARGET:\n")
            f.write("   Purpose: Minimum feature size determination\n")
            f.write("   Feature range: 0.1 - 2.0 μm (10 steps)\n")
            f.write("   5 lines per feature size\n\n")
            
            f.write("4. ELMORE PATTERN:\n")
            f.write("   Purpose: Proximity effect characterization\n")
            f.write("   Tests: Dense vs isolated line behavior\n\n")
            
            f.write("5. CONTACT HOLE SHRINK TEST:\n")
            f.write("   Purpose: Process bias characterization\n")
            f.write("   Nominal: 0.5 μm ± 0.2 μm (9 steps)\n\n")
            
            f.write("6. ALIGNMENT MARKS:\n")
            f.write("   Type: Cross marks\n")
            f.write("   Size: 20 × 20 μm\n")
            f.write("   Locations: Four corners\n")
            f.write("   Layer: 2\n\n")
            
            f.write("-"*70 + "\n")
            f.write("RECOMMENDED EXPOSURE STRATEGY:\n")
            f.write("-"*70 + "\n")
            f.write("• Energy: 1.0 - 2.0 keV (optimized from simulations)\n")
            f.write("• Dose range: 0.5× - 2× D₀ (bracket optimum)\n")
            f.write("• Focus: Vary mask-resist gap in 5 μm steps\n")
            f.write("• Expected results:\n")
            f.write("  - Line/space: Measure contrast vs pitch\n")
            f.write("  - Holes: Determine CD bias and circularity\n")
            f.write("  - Resolution: Find minimum resolvable feature\n")
            f.write("  - Elmore: Quantify proximity effect magnitude\n\n")
            
            f.write("="*70 + "\n")
        
        print(f"Pattern report written to: {report_path}")
        return report_path


def generate_xrl_test_layouts(output_dir: str = 'layouts'):
    """
    Main function to generate all XRL test layouts.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("X-RAY LITHOGRAPHY TEST PATTERN GENERATION")
    print("="*70)
    print()
    
    designer = XRLLayoutDesigner("XRL_TEST_PATTERNS")
    
    # Generate master layout
    gds_file = os.path.join(output_dir, 'xrl_test_patterns.gds')
    designer.create_master_layout(gds_file, die_size=(10000, 10000))
    
    # Generate report
    report_file = designer.generate_pattern_report(gds_file)
    
    print("\n" + "="*70)
    print("LAYOUT GENERATION COMPLETE")
    print("="*70)
    print(f"GDS file: {gds_file}")
    print(f"Report: {report_file}")
    print()
    
    return gds_file, report_file


if __name__ == "__main__":
    generate_xrl_test_layouts()
