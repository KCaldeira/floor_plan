#!/usr/bin/env python3
"""
Test script to verify Excel formatting changes.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from output_analyzer import OutputAnalyzer

def test_formatting():
    """Test the Excel formatting functionality."""
    
    # Create test data
    test_distances = pd.DataFrame({
        'First point ID': ['A', 'B', 'C'],
        'Second point ID': ['B', 'C', 'A'],
        'Distance': [5.123456, 3.987654, 4.555555],
        'Optimized Distance': [5.123, 3.988, 4.556],
        'Distance Error': [0.000456, 0.000346, 0.000445]
    })
    
    test_coordinates = pd.DataFrame({
        'Point ID': ['A', 'B', 'C'],
        'x_guess': [0.0, 5.123456, 2.987654],
        'y_guess': [0.0, 0.0, 4.555555],
        'Fixed Point': ['origin', 'y_axis', ''],
        'x_optimized': [0.0, 0.0, 2.987],
        'y_optimized': [0.0, 5.123, 4.556],
        'Distance Count': [2, 2, 2],
        'RMS Error': [0.000456, 0.000346, 0.000445],
        'Movement Distance': [0.0, 0.0, 0.001]
    })
    
    # Test the formatting
    analyzer = OutputAnalyzer()
    
    # Save with formatting
    output_file = "test_formatting_output.xlsx"
    analyzer.save_enhanced_output(test_distances, test_coordinates, output_file)
    
    print(f"Test formatting completed. Check {output_file} for formatting verification.")
    print("Expected formatting:")
    print("- All columns should be 10 units wide")
    print("- Header row should have word wrapping")
    print("- Numeric data should be formatted to 3 decimal places")

if __name__ == '__main__':
    test_formatting() 