#!/usr/bin/env python3
"""
Test script to verify the ignore feature in the data loader.
"""

import pandas as pd
import tempfile
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import ExcelDataLoader


def create_test_excel_file(file_path):
    """Create a test Excel file with some rows marked as 'ignore'."""
    
    # Sheet 1: Distances with Status column
    distances_data = {
        'First point ID': ['A', 'B', 'A', 'C', 'B'],
        'Second point ID': ['B', 'C', 'C', 'D', 'D'],
        'Distance': [5.0, 3.0, 4.0, 2.0, 6.0],
        'Status': ['', 'ignore', '', 'ignore', '']
    }
    distances_df = pd.DataFrame(distances_data)
    
    # Sheet 2: Initial Coordinates
    coordinates_data = {
        'Point ID': ['A', 'B', 'C', 'D'],
        'x_guess': [0.0, 0.0, 3.0, 5.0],
        'y_guess': [0.0, 5.0, 4.0, 2.0],
        'Fixed Point': ['origin', 'y_axis', '', '']
    }
    coordinates_df = pd.DataFrame(coordinates_data)
    
    # Create Excel file
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        distances_df.to_excel(writer, sheet_name='Distances', index=False)
        coordinates_df.to_excel(writer, sheet_name='Initial Coordinates', index=False)
    
    print(f"Created test file: {file_path}")
    print("Distances data:")
    print(distances_df)
    print("\nCoordinates data:")
    print(coordinates_df)


def test_ignore_feature():
    """Test the ignore feature."""
    
    # Create temporary test file
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
        test_file = tmp_file.name
    
    try:
        # Create test Excel file
        create_test_excel_file(test_file)
        
        # Test loading with ignore feature
        print("\n=== Testing Ignore Feature ===")
        loader = ExcelDataLoader(test_file)
        distances, coordinates = loader.load_data()
        
        print(f"\nLoaded {len(distances)} distance measurements after filtering:")
        for (p1, p2), dist in distances.items():
            print(f"  {p1} - {p2}: {dist}")
        
        print(f"\nLoaded {len(coordinates)} coordinates:")
        for point_id, (x, y) in coordinates.items():
            print(f"  {point_id}: ({x}, {y})")
        
        # Verify that ignored rows are not included
        expected_distances = {('A', 'B'): 5.0, ('A', 'C'): 4.0, ('B', 'D'): 6.0}
        assert distances == expected_distances, f"Expected {expected_distances}, got {distances}"
        
        print("\n✅ Test passed! Ignored rows were successfully filtered out.")
        
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.unlink(test_file)


if __name__ == '__main__':
    test_ignore_feature() 