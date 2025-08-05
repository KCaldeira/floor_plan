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
from src.measurement_sensitivity_analyzer import MeasurementSensitivityAnalyzer
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
    # STEP 3: SENSITIVITY-BASED DIAGNOSTICS
    # ============================================================================
    print("STEP 3: Running sensitivity-based diagnostics...")
    
    try:
        # Create sensitivity analyzer
        sensitivity_analyzer = MeasurementSensitivityAnalyzer(distances, initial_coordinates, fixed_points_info)
        
        # Run baseline optimization
        baseline_result = sensitivity_analyzer.run_baseline_optimization()
        
        # Run leave-one-out analysis to identify problematic measurements
        print("  Running leave-one-out analysis to identify problematic measurements...")
        sensitivity_results = sensitivity_analyzer.analyze_measurement_sensitivity()
        
        # Get measurements that cause optimization failures
        failed_measurements = []
        for pair, result in sensitivity_analyzer.sensitivity_results.items():
            if result['rms_displacement'] == float('inf'):
                failed_measurements.append({
                    'pair': pair,
                    'distance': result['original_distance'],
                    'error': result.get('error', 'Unknown error')
                })
        
        if failed_measurements:
            print("\nMEASUREMENTS CAUSING OPTIMIZATION FAILURES:")
            for failed in failed_measurements:
                print(f"  {failed['pair'][0]}-{failed['pair'][1]}: {failed['distance']:.3f} - {failed['error']}")
        
        # Get measurements with highest impact
        top_measurements = sensitivity_analyzer.get_top_impactful_measurements(10)
        print("\nMEASUREMENTS WITH HIGHEST IMPACT (potential errors):")
        for i, measurement in enumerate(top_measurements[:10], 1):
            pair = measurement['measurement_pair']
            print(f"  {i:2d}. {pair[0]}-{pair[1]}: {measurement['original_distance']:.3f} "
                  f"(RMS displacement: {measurement['rms_displacement']:.6f})")
        
        # Run Jacobian analysis for local sensitivity
        print("\n  Running Jacobian analysis for local sensitivity...")
        jacobian_results = sensitivity_analyzer.compute_jacobian_sensitivity()
        
        # Get measurements with highest local sensitivity
        top_jacobian = jacobian_results['sensitivities'][:10]
        print("\nMEASUREMENTS WITH HIGHEST LOCAL SENSITIVITY:")
        for i, measurement in enumerate(top_jacobian[:10], 1):
            pair = measurement['measurement_pair']
            print(f"  {i:2d}. {pair[0]}-{pair[1]}: {measurement['distance']:.3f} "
                  f"(sensitivity: {measurement['sensitivity']:.6f})")
        
    except Exception as e:
        print(f"❌ Error in sensitivity analysis: {str(e)}")
    
    print()
    
    # ============================================================================
    # STEP 4: RECOMMENDATIONS
    # ============================================================================
    print("STEP 4: Recommendations...")
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
    
    # Check for optimization failures
    if 'failed_measurements' in locals() and failed_measurements:
        recommendations.append("• Re-measure distances that cause optimization failures")
    
    # Check for high-impact measurements
    if 'top_measurements' in locals() and top_measurements:
        recommendations.append("• Focus on re-measuring high-impact measurements first")
    
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
        'analysis': analysis,
        'sensitivity_analyzer': sensitivity_analyzer if 'sensitivity_analyzer' in locals() else None
    }


def main():
    """Main function to run data diagnostics."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run data quality diagnostics')
    parser.add_argument('--data-file', type=str, help='Path to Excel data file')
    
    args = parser.parse_args()
    
    try:
        results = run_data_diagnostics(args.data_file)
        return 0
    except Exception as e:
        print(f"Error during diagnostics: {str(e)}")
        return 1


if __name__ == '__main__':
    exit(main()) 