#!/usr/bin/env python3
"""
Complete sensitivity analysis script.

This script runs both leave-one-out and Jacobian sensitivity analyses
to provide comprehensive measurement sensitivity information.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import ExcelDataLoader
from src.measurement_sensitivity_analyzer import MeasurementSensitivityAnalyzer
from src.visualization import FloorPlanVisualizer
import datetime


def run_complete_sensitivity_analysis(data_file: str = None):
    """
    Run complete sensitivity analysis including both leave-one-out and Jacobian methods.
    
    Args:
        data_file: Path to Excel file with distance measurements and coordinates
    """
    
    # Use default data file if none provided
    if data_file is None:
        data_file = "data/50_e_1_st_measurements.xlsx"
    
    print("="*80)
    print("COMPLETE SENSITIVITY ANALYSIS")
    print("="*80)
    print(f"Data file: {data_file}")
    print()
    
    # Create visualizer to get output directory
    visualizer = FloorPlanVisualizer()
    output_dir = visualizer.get_output_directory()
    print(f"Output directory: {output_dir}")
    print()
    
    # ============================================================================
    # STEP 1: LOAD DATA
    # ============================================================================
    print("STEP 1: Loading data...")
    loader = ExcelDataLoader(data_file)
    distances, initial_coordinates = loader.load_data()
    fixed_points_info = loader.get_fixed_points_info()
    
    print(f"  Loaded {len(distances)} distance measurements")
    print(f"  Loaded {len(initial_coordinates)} points")
    print(f"  Fixed points: {len([k for k, v in fixed_points_info.items() if v in ['origin', 'y_axis']])}")
    print()
    
    # ============================================================================
    # STEP 2: RUN COMPLETE SENSITIVITY ANALYSIS
    # ============================================================================
    print("STEP 2: Running complete sensitivity analysis...")
    
    # Create sensitivity analyzer
    analyzer = MeasurementSensitivityAnalyzer(distances, initial_coordinates, fixed_points_info)
    
    # Run both analyses
    results = analyzer.run_complete_sensitivity_analysis()
    
    print()
    
    # ============================================================================
    # STEP 3: EXPORT RESULTS
    # ============================================================================
    print("STEP 3: Exporting results...")
    
    # Export leave-one-out results
    leave_one_out_file = os.path.join(output_dir, "leave_one_out_sensitivity_analysis.xlsx")
    analyzer.export_sensitivity_results(leave_one_out_file)
    
    # Export Jacobian results
    jacobian_file = os.path.join(output_dir, "jacobian_sensitivity_analysis.xlsx")
    analyzer.export_jacobian_results(results['jacobian_results'], jacobian_file)
    
    print(f"  Leave-one-out analysis saved to: {leave_one_out_file}")
    print(f"  Jacobian analysis saved to: {jacobian_file}")
    print()
    
    # ============================================================================
    # STEP 4: COMPARISON ANALYSIS
    # ============================================================================
    print("STEP 4: Comparing analysis methods...")
    
    # Get top measurements from both methods
    leave_one_out_top = analyzer.get_top_impactful_measurements(10)
    jacobian_top = results['jacobian_results']['sensitivities'][:10]
    
    # Find common measurements in top 10
    leave_one_out_pairs = set(f"{m['measurement_pair'][0]}-{m['measurement_pair'][1]}" for m in leave_one_out_top)
    jacobian_pairs = set(f"{m['measurement_pair'][0]}-{m['measurement_pair'][1]}" for m in jacobian_top)
    
    common_pairs = leave_one_out_pairs.intersection(jacobian_pairs)
    
    print(f"  Measurements in top 10 of both methods: {len(common_pairs)}")
    if common_pairs:
        print("  Common high-impact measurements:")
        for pair in sorted(common_pairs):
            print(f"    - {pair}")
    
    print()
    
    # ============================================================================
    # STEP 5: ANALYSIS COMPLETE
    # ============================================================================
    print("STEP 5: Analysis complete!")
    print("="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print(f"Data file: {data_file}")
    print(f"Output directory: {output_dir}")
    print()
    print("Files generated:")
    print(f"  - leave_one_out_sensitivity_analysis.xlsx (Leave-one-out analysis)")
    print(f"  - jacobian_sensitivity_analysis.xlsx (Jacobian analysis)")
    print()
    print("Key insights:")
    print("  - Leave-one-out: Shows impact of completely removing each measurement")
    print("  - Jacobian: Shows local sensitivity to small changes in measurements")
    print("  - Both methods provide complementary sensitivity information")
    print()
    print("Next steps:")
    print("  - Review both Excel files for detailed analysis")
    print("  - Focus on measurements that rank high in both methods")
    print("  - Consider re-measuring high-impact measurements if accuracy is poor")
    print()
    print("Analysis completed successfully!")
    
    return results


def main():
    """Main function to run the complete sensitivity analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run complete sensitivity analysis')
    parser.add_argument('--data-file', type=str, help='Path to Excel data file')
    
    args = parser.parse_args()
    
    try:
        results = run_complete_sensitivity_analysis(args.data_file)
        return 0
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return 1


if __name__ == '__main__':
    exit(main()) 