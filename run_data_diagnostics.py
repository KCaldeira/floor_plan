#!/usr/bin/env python3
"""
Data diagnostics script.

This script performs comprehensive data quality analysis to help identify
bad measurements, mis-entered point identifiers, and other data issues.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import ExcelDataLoader
import pandas as pd


def run_data_diagnostics(data_file: str = None):
    """
    Run comprehensive data quality diagnostics.
    
    Args:
        data_file: Path to Excel file with distance measurements and coordinates
    """
    
    # Use default data file if none provided
    if data_file is None:
        data_file = "data/50_e_1_st_measurements.xlsx"
    
    print("="*80)
    print("DATA QUALITY DIAGNOSTICS")
    print("="*80)
    print(f"Data file: {data_file}")
    print()
    
    # ============================================================================
    # STEP 1: LOAD DATA AND RUN BASIC DIAGNOSTICS
    # ============================================================================
    print("STEP 1: Loading data and running basic diagnostics...")
    
    try:
        loader = ExcelDataLoader(data_file)
        distances, initial_coordinates = loader.load_data()
        fixed_points_info = loader.get_fixed_points_info()
        
        print(f"  Loaded {len(distances)} distance measurements")
        print(f"  Loaded {len(initial_coordinates)} points")
        print(f"  Fixed points: {len([k for k, v in fixed_points_info.items() if v in ['origin', 'y_axis']])}")
        
        # Print comprehensive data quality report
        loader.print_data_quality_report()
        
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return
    
    print()
    
    # ============================================================================
    # STEP 2: DETAILED ISSUE ANALYSIS
    # ============================================================================
    print("STEP 2: Detailed issue analysis...")
    
    # Get detailed analysis
    analysis = loader.analyze_data_quality()
    
    # Analyze specific issues
    if 'duplicate_measurements' in analysis:
        print("\nDUPLICATE MEASUREMENTS:")
        for dup in analysis['duplicate_measurements']:
            print(f"  {dup['point1']}-{dup['point2']}: {dup['distance']:.3f}")
    
    if 'extreme_distances' in analysis:
        print("\nEXTREME DISTANCES (potential errors):")
        for ext in analysis['extreme_distances']:
            print(f"  {ext['point1']}-{ext['point2']}: {ext['distance']:.3f} (z-score: {ext['z_score']:.2f}, type: {ext['type']})")
    
    if 'isolated_points' in analysis:
        print("\nISOLATED POINTS (only one measurement):")
        for point in analysis['isolated_points']:
            print(f"  {point}")
    
    if 'coordinate_outliers' in analysis:
        print("\nCOORDINATE OUTLIERS:")
        for outlier in analysis['coordinate_outliers']:
            print(f"  {outlier['point_id']}: ({outlier['x']:.3f}, {outlier['y']:.3f}) "
                  f"(x-z: {outlier['x_z_score']:.2f}, y-z: {outlier['y_z_score']:.2f})")
    
    print()
    
    # ============================================================================
    # STEP 3: RECOMMENDATIONS
    # ============================================================================
    print("STEP 3: Recommendations...")
    print("\nBASED ON THE ANALYSIS, CONSIDER THE FOLLOWING:")
    
    recommendations = []
    
    # Check for point consistency issues
    consistency = analysis['point_consistency']
    if not consistency['consistent']:
        recommendations.append("• Fix point identifier mismatches between distance and coordinate data")
    
    # Check for duplicate measurements
    if 'duplicate_measurements' in analysis:
        recommendations.append("• Remove or correct duplicate distance measurements")
    
    # Check for extreme distances
    if 'extreme_distances' in analysis:
        recommendations.append("• Re-measure distances that are statistical outliers")
    
    # Check for isolated points
    if 'isolated_points' in analysis:
        recommendations.append("• Add more measurements for isolated points")
    
    # Check for coordinate outliers
    if 'coordinate_outliers' in analysis:
        recommendations.append("• Verify coordinate values for statistical outliers")
    
    if not recommendations:
        recommendations.append("• No obvious data quality issues detected")
        recommendations.append("• Consider running the complete analysis to see optimization results")
    
    for rec in recommendations:
        print(rec)
    
    print("\n" + "="*80)
    print("DIAGNOSTICS COMPLETE")
    print("="*80)
    
    return {
        'loader': loader,
        'analysis': analysis
    }


def main():
    """Main function to run data diagnostics."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run data quality diagnostics")
    parser.add_argument("--data_file", type=str, default=None,
                       help="Path to Excel file with distance measurements and coordinates")
    
    args = parser.parse_args()
    
    try:
        run_data_diagnostics(args.data_file)
    except Exception as e:
        print(f"Error running diagnostics: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 