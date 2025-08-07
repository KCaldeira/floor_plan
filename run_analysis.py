#!/usr/bin/env python3
"""
Basic Analysis Script

This script runs:
1. Basic coordinate estimation (full optimization)
2. Line orientation constraints (H/V) if specified in data
3. Enhanced output analysis with Excel formatting
4. Visualizations (PNG files)

Usage:
    python run_analysis.py [data_file]
    
Examples:
    python run_analysis.py
    python run_analysis.py data/my_data.xlsx
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from src.data_loader import ExcelDataLoader
from src.coordinate_estimator import CoordinateEstimator
from src.output_analyzer import OutputAnalyzer
from src.visualization import FloorPlanVisualizer
from src.distance_calculator import DistanceCalculator
from src.summary_calculator import SummaryCalculator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run floor plan coordinate estimation analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run with default data file
  %(prog)s data/my_data.xlsx                 # Run with custom data file
  %(prog)s data/my_data.xlsx --line-weight 5.0  # Increase line constraint weighting
        """
    )
    
    parser.add_argument(
        'data_file', 
        nargs='?', 
        default="data/50_e_1_st_measurements.xlsx",
        help="Path to Excel file with distance measurements and coordinates (default: data/50_e_1_st_measurements.xlsx)"
    )
    
    parser.add_argument(
        '--line-weight',
        type=float,
        default=1.0,
        help="Weight multiplier for horizontal (H) and vertical (V) line constraints (default: 1.0)"
    )
    
    return parser.parse_args()


def analyze_initial_guess_discrepancies(distances, initial_coordinates, output_dir):
    """
    Analyze discrepancies between measured distances and distances calculated from initial guesses.
    
    Args:
        distances: List of distance measurements
        initial_coordinates: Dictionary of initial coordinates
        output_dir: Output directory path
    
    Returns:
        DataFrame with analysis results
    """
    print("\n  Analyzing initial guess discrepancies...")
    
    # Create distance calculator
    distance_calc = DistanceCalculator()
    
    # Calculate distances from initial coordinates
    results = []
    total_discrepancy = 0.0
    max_discrepancy = 0.0
    max_discrepancy_measurement = None
    
    for (point1, point2), measured_distance in distances.items():
        
        # Get coordinates for both points
        if point1 in initial_coordinates and point2 in initial_coordinates:
            coord1 = initial_coordinates[point1]
            coord2 = initial_coordinates[point2]
            
            # Calculate distance from initial coordinates
            calculated_distance = distance_calc.euclidean_distance(coord1, coord2)
            
            # Calculate discrepancy
            discrepancy = abs(calculated_distance - measured_distance)
            relative_discrepancy = discrepancy / measured_distance if measured_distance > 0 else 0
            
            total_discrepancy += discrepancy
            
            if discrepancy > max_discrepancy:
                max_discrepancy = discrepancy
                max_discrepancy_measurement = (point1, point2)
            
            results.append({
                'Point1': point1,
                'Point2': point2,
                'Measured_Distance': measured_distance,
                'Calculated_Distance': calculated_distance,
                'Absolute_Discrepancy': discrepancy,
                'Relative_Discrepancy': relative_discrepancy
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by absolute discrepancy (descending)
    df = df.sort_values('Absolute_Discrepancy', ascending=False)
    
    # Save to Excel
    filename = os.path.join(output_dir, "initial_guess_discrepancies.xlsx")
    df.to_excel(filename, index=False)
    
    print(f"    Saved initial guess discrepancies to: {filename}")
    print(f"    Total discrepancy: {total_discrepancy:.6f}")
    print(f"    Max discrepancy: {max_discrepancy:.6f} ({max_discrepancy_measurement[0]}-{max_discrepancy_measurement[1]})")
    print(f"    Average discrepancy: {total_discrepancy/len(distances):.6f}")
    
    return df


def main():
    """Main function to run the analysis."""
    args = parse_arguments()
    
    # Check if data file exists
    if not os.path.exists(args.data_file):
        print(f"Error: Data file not found: {args.data_file}")
        print("Available data files:")
        data_dir = "data"
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith('.xlsx'):
                    print(f"  - data/{file}")
        return 1
    
    print("="*80)
    print("FLOOR PLAN COORDINATE ESTIMATION ANALYSIS")
    print("="*80)
    print(f"Data file: {args.data_file}")
    print()
    
    # Create visualizer to get output directory
    visualizer = FloorPlanVisualizer()
    output_dir = visualizer.get_output_directory()
    print(f"Output directory: {output_dir}")
    print()
    
    try:
        # ============================================================================
        # STEP 1: LOAD DATA
        # ============================================================================
        print("STEP 1: Loading data...")
        loader = ExcelDataLoader(args.data_file)
        distances, initial_coordinates = loader.load_data()
        fixed_points_info = loader.get_fixed_points_info()
        original_distances_df, original_coordinates_df = loader.get_original_dataframes()
        
        print(f"  Loaded {len(distances)} distance measurements")
        print(f"  Loaded {len(initial_coordinates)} points")
        print(f"  Fixed points: {len([k for k, v in fixed_points_info.items() if v in ['origin', 'y_axis']])}")
        print()
        
        # ============================================================================
        # STEP 2: ANALYZE INITIAL GUESS DISCREPANCIES
        # ============================================================================
        print("STEP 2: Analyzing initial guess discrepancies...")
        initial_discrepancies = analyze_initial_guess_discrepancies(
            distances, initial_coordinates, output_dir
        )
        
        # ============================================================================
        # STEP 3: RUN COORDINATE ESTIMATION
        # ============================================================================
        print("STEP 3: Running coordinate estimation...")
        
        # Get line orientations
        line_orientations = loader.get_line_orientations()
        
        # Get weights
        weights = loader.get_weights()
        
        # Count line orientation constraints
        horizontal_constraints = sum(1 for orient in line_orientations.values() if orient == 'H')
        vertical_constraints = sum(1 for orient in line_orientations.values() if orient == 'V')
        
        print(f"  Line orientation constraints: {horizontal_constraints} horizontal, {vertical_constraints} vertical")
        print(f"  Line constraint weight multiplier: {args.line_weight}")
        
        # Report weight statistics
        weight_values = list(weights.values())
        if len(set(weight_values)) > 1:  # If not all weights are the same
            print(f"  Measurement weights: min={min(weight_values):.2f}, max={max(weight_values):.2f}, mean={np.mean(weight_values):.2f}")
        else:
            print(f"  All measurements have equal weight: {weight_values[0]:.2f}")
        
        # Run full optimization
        print("  Running full optimization...")
        estimator = CoordinateEstimator(
            distances, 
            initial_coordinates, 
            fixed_points_info, 
            line_orientations, 
            line_orientation_weight=args.line_weight,
            weights=weights
        )
        result = estimator.optimize()
        
        print(f"    Optimization completed successfully")
        print(f"    Final RMS error: {result['error_metrics']['root_mean_square_error']:.6f}")
        print(f"    Iterations: {result['optimization_info']['nit']}")
        print(f"    Function evaluations: {result['optimization_info']['nfev']}")
        print()
        
        # ============================================================================
        # STEP 4: ENHANCED OUTPUT ANALYSIS
        # ============================================================================
        print("STEP 4: Generating enhanced output analysis...")
        analyzer = OutputAnalyzer()
        enhanced_distances, enhanced_coordinates = analyzer.generate_enhanced_output(
            original_distances_df, original_coordinates_df, 
            result['coordinates'], distances
        )
        
        # Save enhanced output
        enhanced_output_file = os.path.join(output_dir, "enhanced_analysis.xlsx")
        analyzer.save_enhanced_output(enhanced_distances, enhanced_coordinates, enhanced_output_file)
        print(f"  Enhanced analysis saved to: {enhanced_output_file}")
        
        # Print enhanced summary
        analyzer.print_enhanced_summary(enhanced_distances, enhanced_coordinates)
        print()
        
        # ============================================================================
        # STEP 4.5: SUMMARY REPORT GENERATION
        # ============================================================================
        print("STEP 4.5: Generating summary report...")
        summary_calculator = SummaryCalculator()
        summary_report = summary_calculator.generate_summary_report(result['coordinates'])
        
        # Save summary report
        summary_output_file = os.path.join(output_dir, "summary_report.xlsx")
        summary_calculator.save_summary_report(summary_report, summary_output_file)
        print(f"  Summary report saved to: {summary_output_file}")
        
        # Print summary report
        summary_calculator.print_summary_report(summary_report)
        print()
        
        # ============================================================================
        # STEP 5: VISUALIZATION
        # ============================================================================
        print("STEP 5: Creating visualizations...")
        
        # Plot initial coordinates
        fig1 = visualizer.plot_coordinates(initial_coordinates, "Initial Coordinates")
        visualizer.save_plot(fig1, "initial_coordinates.png")
        
        # Plot optimized coordinates
        fig2 = visualizer.plot_coordinates(result['coordinates'], "Optimized Coordinates")
        visualizer.save_plot(fig2, "optimized_coordinates.png")
        
        # Plot comparison
        fig3 = visualizer.plot_comparison(initial_coordinates, result['coordinates'], 
                                         "Initial vs Optimized Coordinates")
        visualizer.save_plot(fig3, "coordinate_comparison.png")
        
        # Plot with distances
        fig4 = visualizer.plot_with_distances(result['coordinates'], distances,
                                             "Optimized Coordinates with Distance Measurements")
        visualizer.save_plot(fig4, "coordinates_with_distances.png")
        
        print(f"  Visualizations saved to: {output_dir}")
        print()
        
        # ============================================================================
        # STEP 6: FINAL SUMMARY
        # ============================================================================
        print("STEP 6: Analysis complete!")
        print("="*80)
        print("ANALYSIS SUMMARY")
        print("="*80)
        print(f"Data file: {args.data_file}")
        print(f"Output directory: {output_dir}")
        print()
        print("Files generated:")
        print(f"  - enhanced_analysis.xlsx (Enhanced distance and coordinate analysis)")
        print(f"  - summary_report.xlsx (Floor plan summary statistics)")
        print(f"  - initial_guess_discrepancies.xlsx (Initial guess analysis)")
        print(f"  - initial_coordinates.png (Initial coordinate visualization)")
        print(f"  - optimized_coordinates.png (Optimized coordinate visualization)")
        print(f"  - coordinate_comparison.png (Before/after comparison)")
        print(f"  - coordinates_with_distances.png (Coordinates with distance measurements)")
        print()
        print("Key results:")
        print(f"  - Final RMS error: {result['error_metrics']['root_mean_square_error']:.6f}")
        print(f"  - Total measurements analyzed: {len(distances)}")
        print(f"  - Total points optimized: {len(result['coordinates'])}")
        print(f"  - Line constraint weight multiplier: {args.line_weight}")
        print()
        print("Next steps:")
        print("  - Review summary_report.xlsx for floor plan width, length, and area statistics")
        print("  - Review enhanced_analysis.xlsx for detailed distance and coordinate analysis")
        print("  - Check initial_guess_discrepancies.xlsx to identify problematic initial guesses")
        print("  - Examine visualizations to understand the coordinate optimization results")
        print("  - Check distance errors in the enhanced analysis to identify problematic measurements")
        print()
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 