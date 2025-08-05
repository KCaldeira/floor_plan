#!/usr/bin/env python3
"""
Simple test for the ignore feature.
"""

import pandas as pd
import tempfile
import os

def test_ignore():
    """Test the ignore feature with a simple example."""
    
    # Create test data
    distances_data = {
        'First point ID': ['A', 'B', 'A'],
        'Second point ID': ['B', 'C', 'C'],
        'Distance': [5.0, 3.0, 4.0],
        'Status': ['', 'ignore', '']
    }
    distances_df = pd.DataFrame(distances_data)
    
    coordinates_data = {
        'Point ID': ['A', 'B', 'C'],
        'x_guess': [0.0, 0.0, 3.0],
        'y_guess': [0.0, 5.0, 4.0],
        'Fixed Point': ['origin', 'y_axis', '']
    }
    coordinates_df = pd.DataFrame(coordinates_data)
    
    # Create temporary Excel file
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
        test_file = tmp_file.name
    
    try:
        # Write test data to Excel
        with pd.ExcelWriter(test_file, engine='openpyxl') as writer:
            distances_df.to_excel(writer, sheet_name='Distances', index=False)
            coordinates_df.to_excel(writer, sheet_name='Initial Coordinates', index=False)
        
        print(f"Created test file: {test_file}")
        print("Original distances data:")
        print(distances_df)
        
        # Test the data loader
        import sys
        sys.path.insert(0, 'src')
        from data_loader import ExcelDataLoader
        
        loader = ExcelDataLoader(test_file)
        distances, coordinates = loader.load_data()
        
        print(f"\nAfter filtering, loaded {len(distances)} distance measurements:")
        for (p1, p2), dist in distances.items():
            print(f"  {p1} - {p2}: {dist}")
        
        # Should only have A-B and A-C, not B-C (which was ignored)
        expected = {('A', 'B'): 5.0, ('A', 'C'): 4.0}
        if distances == expected:
            print("✅ Test passed! Ignored row was successfully filtered out.")
        else:
            print(f"❌ Test failed! Expected {expected}, got {distances}")
        
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.unlink(test_file)

if __name__ == '__main__':
    test_ignore() 