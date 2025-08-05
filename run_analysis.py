#!/usr/bin/env python3
"""
Basic Analysis and Leave-One-Out Analysis Script

This script runs:
1. Basic coordinate estimation (full, x_only, y_only)
2. Enhanced output analysis with Excel formatting
3. Visualizations (PNG files)
4. Leave-one-out sensitivity analysis (optional)

Usage:
    python run_basic_and_leave_one_out_analysis.py [data_file] [--no-leave-one-out]
    
Examples:
    python run_basic_and_leave_one_out_analysis.py
    python run_basic_and_leave_one_out_analysis.py data/my_data.xlsx
    python run_basic_and_leave_one_out_analysis.py --no-leave-one-out
    python run_basic_and_leave_one_out_analysis.py data/my_data.xlsx --no-leave-one-out
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
from src.measurement_sensitivity_analyzer import MeasurementSensitivityAnalyzer
from src.distance_calculator import DistanceCalculator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run floor plan coordinate estimation analysis with optional leave-one-out analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run with default data file and leave-one-out analysis
  %(prog)s data/my_data.xlsx                 # Run with custom data file
  %(prog)s --no-leave-one-out               # Skip leave-one-out analysis
  %(prog)s data/my_data.xlsx --no-leave-one-out  # Custom data file, no leave-one-out
        """
    )
    
    parser.add_argument(
        'data_file', 
        nargs='?', 
        default="data/50_e_1_st_measurements.xlsx",
        help="Path to Excel file with distance measurements and coordinates (default: data/50_e_1_st_measurements.xlsx)"
    )
    
    parser.add_argument(
        '--no-leave-one-out',
        action='store_true',
        help="Skip the leave-one-out sensitivity analysis (faster execution)"
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
            relative_discrepancy = (discrepancy / measured_distance) * 100 if measured_distance > 0 else 0
            
            total_discrepancy += discrepancy
            if discrepancy > max_discrepancy:
                max_discrepancy = discrepancy
                max_discrepancy_measurement = f"{point1}-{point2}"
            
            results.append({
                'Point1': point1,
                'Point2': point2,
                'Measured_Distance': measured_distance,
                'Calculated_Distance': calculated_distance,
                'Discrepancy': discrepancy,
                'Relative_Discrepancy_Percent': relative_discrepancy,
                'Coord1_X': coord1[0],
                'Coord1_Y': coord1[1],
                'Coord2_X': coord2[0],
                'Coord2_Y': coord2[1]
            })
        else:
            # Handle missing coordinates
            missing_point = point1 if point1 not in initial_coordinates else point2
            results.append({
                'Point1': point1,
                'Point2': point2,
                'Measured_Distance': measured_distance,
                'Calculated_Distance': np.nan,
                'Discrepancy': np.nan,
                'Relative_Discrepancy_Percent': np.nan,
                'Coord1_X': initial_coordinates.get(point1, [np.nan, np.nan])[0] if point1 in initial_coordinates else np.nan,
                'Coord1_Y': initial_coordinates.get(point1, [np.nan, np.nan])[1] if point1 in initial_coordinates else np.nan,
                'Coord2_X': initial_coordinates.get(point2, [np.nan, np.nan])[0] if point2 in initial_coordinates else np.nan,
                'Coord2_Y': initial_coordinates.get(point2, [np.nan, np.nan])[1] if point2 in initial_coordinates else np.nan,
                'Error': f"Missing coordinates for point: {missing_point}"
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by discrepancy (descending)
    df_sorted = df.sort_values('Discrepancy', ascending=False, na_position='last')
    
    # Calculate summary statistics
    valid_discrepancies = df['Discrepancy'].dropna()
    mean_discrepancy = valid_discrepancies.mean() if len(valid_discrepancies) > 0 else 0
    std_discrepancy = valid_discrepancies.std() if len(valid_discrepancies) > 0 else 0
    
    # Print summary
    print(f"    Total measurements analyzed: {len(distances)}")
    print(f"    Average discrepancy: {mean_discrepancy:.6f}")
    print(f"    Standard deviation of discrepancy: {std_discrepancy:.6f}")
    print(f"    Maximum discrepancy: {max_discrepancy:.6f} ({max_discrepancy_measurement})")
    print(f"    Total discrepancy: {total_discrepancy:.6f}")
    
    # Save to Excel with formatting
    filename = os.path.join(output_dir, "initial_guess_discrepancy_analysis.xlsx")
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Write main data
        df_sorted.to_excel(writer, sheet_name='Discrepancy_Analysis', index=False)
        
        # Write summary statistics
        summary_data = {
            'Metric': [
                'Total Measurements',
                'Valid Measurements',
                'Average Discrepancy',
                'Std Dev Discrepancy',
                'Maximum Discrepancy',
                'Total Discrepancy',
                'Max Discrepancy Measurement'
            ],
            'Value': [
                len(distances),
                len(valid_discrepancies),
                f"{mean_discrepancy:.6f}",
                f"{std_discrepancy:.6f}",
                f"{max_discrepancy:.6f}",
                f"{total_discrepancy:.6f}",
                max_discrepancy_measurement or "N/A"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
        
        # Format the sheets
        workbook = writer.book
        
        # Format main analysis sheet
        worksheet = writer.sheets['Discrepancy_Analysis']
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 20)
            worksheet.column_dimensions[column_letter].width = adjusted_width
        
        # Format summary sheet
        worksheet = writer.sheets['Summary_Statistics']
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 20)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print(f"    Analysis saved to: {filename}")
    
    return df_sorted


def analyze_line_constraints(coordinates: Dict[str, Tuple[float, float]], 
                           line_constraints: List[Dict[str, any]], 
                           output_dir: str) -> pd.DataFrame:
    """
    Analyze how well line constraints are satisfied.
    
    Args:
        coordinates: Dictionary mapping point IDs to (x, y) coordinates
        line_constraints: List of line constraint dictionaries
        output_dir: Output directory path
    
    Returns:
        DataFrame with line constraint analysis results
    """
    print("\n  Analyzing line constraint satisfaction...")
    
    results = []
    
    for constraint in line_constraints:
        line_id = constraint["line_id"]
        point_ids = constraint["point_ids"]
        
        # Extract coordinates for points in this line group
        line_points = []
        valid_point_ids = []
        for point_id in point_ids:
            if point_id in coordinates:
                line_points.append(coordinates[point_id])
                valid_point_ids.append(point_id)
        
        if len(line_points) < 2:
            results.append({
                'Line_ID': line_id,
                'Point_Count': len(valid_point_ids),
                'Line_Type': 'insufficient_points',
                'Max_Deviation': np.nan,
                'Avg_Deviation': np.nan,
                'RMS_Deviation': np.nan,
                'Points': ', '.join(valid_point_ids)
            })
            continue
        
        # Fit a line through the points
        x_coords = [p[0] for p in line_points]
        y_coords = [p[1] for p in line_points]
        x_array = np.array(x_coords)
        y_array = np.array(y_coords)
        
        # Check if this is a vertical line
        if np.std(x_array) < 1e-10:
            # Vertical line: x = constant
            x_constant = np.mean(x_array)
            deviations = [abs(x - x_constant) for x in x_coords]
            line_type = "vertical"
        else:
            # Regular line: fit y = mx + b
            A = np.vstack([x_array, np.ones(len(x_array))]).T
            m, b = np.linalg.lstsq(A, y_array, rcond=None)[0]
            
            # Calculate deviations from the fitted line
            deviations = []
            for x, y in zip(x_coords, y_coords):
                y_fitted = m * x + b
                deviation = abs(y - y_fitted)
                deviations.append(deviation)
            
            line_type = "arbitrary"
        
        # Calculate statistics
        max_deviation = max(deviations)
        avg_deviation = np.mean(deviations)
        rms_deviation = np.sqrt(np.mean(np.array(deviations) ** 2))
        
        results.append({
            'Line_ID': line_id,
            'Point_Count': len(valid_point_ids),
            'Line_Type': line_type,
            'Max_Deviation': max_deviation,
            'Avg_Deviation': avg_deviation,
            'RMS_Deviation': rms_deviation,
            'Points': ', '.join(valid_point_ids)
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print summary
    if not df.empty:
        print(f"    Total line constraints analyzed: {len(line_constraints)}")
        if 'Max_Deviation' in df.columns:
            max_dev = df['Max_Deviation'].max()
            avg_dev = df['Avg_Deviation'].mean()
            print(f"    Maximum deviation from any line: {max_dev:.6f}")
            print(f"    Average deviation across all lines: {avg_dev:.6f}")
    
    # Save to Excel
    filename = os.path.join(output_dir, "line_constraint_analysis.xlsx")
    df.to_excel(filename, index=False)
    print(f"    Analysis saved to: {filename}")
    
    return df


def main():
    """Run basic analysis and leave-one-out analysis."""
    
    # Parse command line arguments
    args = parse_arguments()
    
    print("=" * 80)
    print("BASIC ANALYSIS AND LEAVE-ONE-OUT ANALYSIS")
    print("=" * 80)
    print(f"Data file: {args.data_file}")
    print(f"Leave-one-out analysis: {'Disabled' if args.no_leave_one_out else 'Enabled'}")
    
    # Create visualizer to get output directory
    visualizer = FloorPlanVisualizer()
    output_dir = visualizer.get_output_directory()
    print(f"Output directory: {output_dir}")
    
    # STEP 1: Load data
    print("\nSTEP 1: Loading data...")
    loader = ExcelDataLoader(args.data_file)
    distances, initial_coordinates = loader.load_data()
    fixed_points_info = loader.get_fixed_points_info()
    line_constraints = loader.get_line_constraints()
    original_distances_df, original_coordinates_df = loader.get_original_dataframes()
    
    print(f"  Loaded {len(distances)} distance measurements")
    print(f"  Loaded {len(initial_coordinates)} points")
    print(f"  Fixed points info: {fixed_points_info}")
    print(f"  Line constraints: {len(line_constraints)} line groups")
    for constraint in line_constraints:
        print(f"    - {constraint['line_id']}: {constraint['point_ids']}")
    
    # STEP 2: Analyze initial guess discrepancies
    print("\nSTEP 2: Analyzing initial guess discrepancies...")
    discrepancy_df = analyze_initial_guess_discrepancies(distances, initial_coordinates, output_dir)
    
    # STEP 3: Run coordinate estimations with different constraints
    print("\nSTEP 3: Running coordinate estimations...")
    
    # Run three different optimizations
    optimization_types = [
        ("full", "Full Optimization (X and Y)"),
        ("x_only", "X-Only Optimization (Y fixed)"),
        ("y_only", "Y-Only Optimization (X fixed)")
    ]
    
    results = {}
    
    for constraint_type, description in optimization_types:
        print(f"\n  Running {description}...")
        estimator = CoordinateEstimator(distances, initial_coordinates, fixed_points_info, line_constraints)
        result = estimator.optimize(constraint_type=constraint_type)
        results[constraint_type] = result
        
        print(f"    Optimization completed successfully")
        print(f"    Final RMS error: {result['error_metrics']['root_mean_square_error']:.6f}")
        print(f"    Iterations: {result['optimization_info']['nit']}")
        print(f"    Function evaluations: {result['optimization_info']['nfev']}")
    
    # Use full optimization result for the main analysis
    result = results["full"]
    
    # STEP 4: Analyze line constraints for all optimizations
    print("\nSTEP 4: Analyzing line constraints...")
    line_analysis_results = {}
    
    for constraint_type, description in optimization_types:
        print(f"\n  Analyzing line constraints for {description}...")
        result = results[constraint_type]
        line_df = analyze_line_constraints(result['coordinates'], line_constraints, output_dir)
        line_analysis_results[constraint_type] = line_df
    
    # STEP 5: Generate enhanced output analysis for all optimizations
    print("\nSTEP 5: Generating enhanced output analysis...")
    analyzer = OutputAnalyzer()
    
    for constraint_type, description in optimization_types:
        print(f"\n  Generating output for {description}...")
        result = results[constraint_type]
        
        enhanced_distances, enhanced_coordinates = analyzer.generate_enhanced_output(
            original_distances_df, original_coordinates_df, 
            result['coordinates'], distances
        )
        
        # Save enhanced output with constraint type in filename
        constraint_suffix = constraint_type.replace("_", "_")
        enhanced_filename = os.path.join(output_dir, f"enhanced_analysis_{constraint_suffix}.xlsx")
        analyzer.save_enhanced_output(enhanced_distances, enhanced_coordinates, enhanced_filename)
        print(f"    Enhanced output saved to: {enhanced_filename}")
        
        # Print enhanced analysis summary
        print(f"\n=== ENHANCED OUTPUT ANALYSIS - {description.upper()} ===")
        analyzer.print_enhanced_summary(enhanced_distances, enhanced_coordinates)
    
    # STEP 6: Create visualizations for all optimizations
    print("\nSTEP 6: Creating visualizations...")
    
    for constraint_type, description in optimization_types:
        print(f"\n  Creating visualizations for {description}...")
        result = results[constraint_type]
        constraint_suffix = constraint_type.replace("_", "_")
        
        # Initial coordinates plot (same for all)
        if constraint_type == "full":
            fig1 = visualizer.plot_coordinates(initial_coordinates, "Initial Coordinates")
            visualizer.save_plot(fig1, "initial_coordinates.png")
            print(f"    Plot saved to: {os.path.join(output_dir, 'initial_coordinates.png')}")
        
        # Optimized coordinates plot
        fig2 = visualizer.plot_coordinates(result['coordinates'], f"Optimized Coordinates - {description}")
        visualizer.save_plot(fig2, f"optimized_coordinates_{constraint_suffix}.png")
        print(f"    Plot saved to: {os.path.join(output_dir, f'optimized_coordinates_{constraint_suffix}.png')}")
        
        # Comparison plot
        fig3 = visualizer.plot_comparison(initial_coordinates, result['coordinates'])
        visualizer.save_plot(fig3, f"coordinate_comparison_{constraint_suffix}.png")
        print(f"    Plot saved to: {os.path.join(output_dir, f'coordinate_comparison_{constraint_suffix}.png')}")
        
        # Coordinates with distances plot
        fig4 = visualizer.plot_with_distances(result['coordinates'], distances)
        visualizer.save_plot(fig4, f"coordinates_with_distances_{constraint_suffix}.png")
        print(f"    Plot saved to: {os.path.join(output_dir, f'coordinates_with_distances_{constraint_suffix}.png')}")
    
    print(f"  All visualizations saved to: {output_dir}")
    
    # STEP 7: Run leave-one-out sensitivity analysis (if enabled)
    if not args.no_leave_one_out:
        print("\nSTEP 7: Running leave-one-out sensitivity analysis...")
        print("  This will analyze the impact of removing each measurement...")
        
        sensitivity_analyzer = MeasurementSensitivityAnalyzer(
            distances, 
            initial_coordinates, 
            fixed_points_info
        )
        
        # Run baseline optimization
        baseline_result = sensitivity_analyzer.run_baseline_optimization()
        
        # Run leave-one-out analysis only
        print("\nRunning leave-one-out sensitivity analysis...")
        leave_one_out_results = sensitivity_analyzer.analyze_measurement_sensitivity()
        
        # Print leave-one-out summary
        sensitivity_analyzer.print_impact_summary(top_n=15)
        
        # Export leave-one-out results
        leave_one_out_filename = os.path.join(output_dir, "leave_one_out_sensitivity_analysis.xlsx")
        sensitivity_analyzer.export_sensitivity_results(leave_one_out_filename)
        print(f"  Leave-one-out analysis saved to: {leave_one_out_filename}")
        print("  Note: Excel file now includes point-specific movement data for N, M4, O, X0, M3")
    else:
        print("\nSTEP 7: Skipping leave-one-out sensitivity analysis (--no-leave-one-out flag specified)")
    
    # STEP 8: Final summary
    print("\nSTEP 8: Analysis complete!")
    print("=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Data file: {args.data_file}")
    print(f"Output directory: {output_dir}")
    print()
    print("Files generated:")
    print("  Initial Guess Analysis:")
    print(f"    - initial_guess_discrepancy_analysis.xlsx (compares measured vs calculated distances)")
    print("  Line Constraint Analysis:")
    print(f"    - line_constraint_analysis.xlsx (analyzes how well points lie on specified lines)")
    print("  Enhanced Analysis Excel Files:")
    for constraint_type, description in optimization_types:
        constraint_suffix = constraint_type.replace("_", "_")
        print(f"    - enhanced_analysis_{constraint_suffix}.xlsx ({description})")
    if not args.no_leave_one_out:
        print("  Leave-One-Out Analysis:")
        print(f"    - leave_one_out_sensitivity_analysis.xlsx (with point-specific movement data)")
    print("  Visualizations:")
    print(f"    - initial_coordinates.png (Initial coordinate visualization)")
    for constraint_type, description in optimization_types:
        constraint_suffix = constraint_type.replace("_", "_")
        print(f"    - optimized_coordinates_{constraint_suffix}.png ({description})")
        print(f"    - coordinate_comparison_{constraint_suffix}.png ({description})")
        print(f"    - coordinates_with_distances_{constraint_suffix}.png ({description})")
    print()
    print("Key results:")
    # Show top discrepancy measurements
    if not discrepancy_df.empty:
        top_discrepancy = discrepancy_df.iloc[0]
        print(f"  - Largest initial discrepancy: {top_discrepancy['Point1']}-{top_discrepancy['Point2']} ({top_discrepancy['Discrepancy']:.6f})")
    
    # Show line constraint summary
    if line_constraints:
        print(f"  - Line constraints: {len(line_constraints)} line groups specified")
        if 'line_analysis_results' in locals() and 'full' in line_analysis_results:
            full_line_df = line_analysis_results['full']
            if not full_line_df.empty and 'Max_Deviation' in full_line_df.columns:
                max_dev = full_line_df['Max_Deviation'].max()
                print(f"  - Maximum line deviation: {max_dev:.6f}")
    for constraint_type, description in optimization_types:
        result = results[constraint_type]
        print(f"  - {description}: RMS error = {result['error_metrics']['root_mean_square_error']:.6f}")
    print(f"  - Total measurements analyzed: {len(distances)}")
    print(f"  - Total points optimized: {len(result['coordinates'])}")
    
    if not args.no_leave_one_out and 'sensitivity_analyzer' in locals():
        # Get top leave-one-out measurement
        if sensitivity_analyzer.measurement_impact_ranking:
            top_measurement = sensitivity_analyzer.measurement_impact_ranking[0]
            print(f"  - Top leave-one-out measurement: {top_measurement['point1']}-{top_measurement['point2']} (RMS displacement: {top_measurement['rms_displacement']:.6f})")
    
    print()
    print("Next steps:")
    print("  - Review initial_guess_discrepancy_analysis.xlsx to identify problematic measurements")
    print("  - Review line_constraint_analysis.xlsx to see how well line constraints are satisfied")
    print("  - Compare RMS errors between full, x-only, and y-only optimizations")
    print("  - Review enhanced_analysis_*.xlsx files for detailed distance and coordinate analysis")
    if not args.no_leave_one_out:
        print("  - Review leave_one_out_sensitivity_analysis.xlsx for measurement impact (includes point-specific movement data)")
    print("  - Use visualizations to compare coordinate results between optimization types")
    if not args.no_leave_one_out:
        print("  - Focus on measurements that rank high in leave-one-out analysis")
    print()
    print("Analysis completed successfully!")


if __name__ == "__main__":
    main() 