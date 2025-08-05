#!/usr/bin/env python3
"""
Basic Analysis and Jacobian Analysis Script

This script runs:
1. Basic coordinate estimation
2. Enhanced output analysis with Excel formatting
3. Visualizations (PNG files)
4. Jacobian sensitivity analysis (without leave-one-out)

It skips the computationally expensive leave-one-out analysis.
"""

import sys
import os
from datetime import datetime
from src.data_loader import ExcelDataLoader
from src.coordinate_estimator import CoordinateEstimator
from src.output_analyzer import OutputAnalyzer
from src.visualization import FloorPlanVisualizer
from src.measurement_sensitivity_analyzer import MeasurementSensitivityAnalyzer


def main():
    """Run basic analysis and Jacobian analysis."""
    
    print("=" * 80)
    print("BASIC ANALYSIS AND JACOBIAN ANALYSIS")
    print("=" * 80)
    
    # Get data file from command line or use default
    data_file = sys.argv[1] if len(sys.argv) > 1 else "data/50_e_1_st_measurements.xlsx"
    print(f"Data file: {data_file}")
    
    # Create visualizer to get output directory
    visualizer = FloorPlanVisualizer()
    output_dir = visualizer.get_output_directory()
    print(f"Output directory: {output_dir}")
    
    # STEP 1: Load data
    print("\nSTEP 1: Loading data...")
    loader = ExcelDataLoader(data_file)
    distances, initial_coordinates = loader.load_data()
    fixed_points_info = loader.get_fixed_points_info()
    original_distances_df, original_coordinates_df = loader.get_original_dataframes()
    
    print(f"  Loaded {len(distances)} distance measurements")
    print(f"  Loaded {len(initial_coordinates)} points")
    print(f"  Fixed points info: {fixed_points_info}")
    
    # STEP 2: Run main coordinate estimation
    print("\nSTEP 2: Running main coordinate estimation...")
    estimator = CoordinateEstimator(distances, initial_coordinates, fixed_points_info)
    result = estimator.optimize()
    
    print(f"  Optimization completed successfully")
    print(f"  Final RMS error: {result['error_metrics']['root_mean_square_error']:.6f}")
    print(f"  Iterations: {result['optimization_info']['nit']}")
    print(f"  Function evaluations: {result['optimization_info']['nfev']}")
    
    # STEP 3: Generate enhanced output analysis
    print("\nSTEP 3: Generating enhanced output analysis...")
    analyzer = OutputAnalyzer()
    enhanced_distances, enhanced_coordinates = analyzer.generate_enhanced_output(
        original_distances_df, original_coordinates_df, 
        result['coordinates'], distances
    )
    
    # Save enhanced output
    enhanced_filename = os.path.join(output_dir, "enhanced_analysis.xlsx")
    analyzer.save_enhanced_output(enhanced_distances, enhanced_coordinates, enhanced_filename)
    print(f"Enhanced output saved to: {enhanced_filename}")
    
    # Print enhanced analysis summary
    print("\n=== ENHANCED OUTPUT ANALYSIS ===")
    analyzer.print_enhanced_summary(enhanced_distances, enhanced_coordinates)
    
    # STEP 4: Create visualizations
    print("\nSTEP 4: Creating visualizations...")
    
    # Initial coordinates plot
    fig1 = visualizer.plot_coordinates(initial_coordinates, "Initial Coordinates")
    visualizer.save_plot(fig1, "initial_coordinates.png")
    print(f"Plot saved to: {os.path.join(output_dir, 'initial_coordinates.png')}")
    
    # Optimized coordinates plot
    fig2 = visualizer.plot_coordinates(result['coordinates'], "Optimized Coordinates")
    visualizer.save_plot(fig2, "optimized_coordinates.png")
    print(f"Plot saved to: {os.path.join(output_dir, 'optimized_coordinates.png')}")
    
    # Comparison plot
    fig3 = visualizer.plot_comparison(initial_coordinates, result['coordinates'])
    visualizer.save_plot(fig3, "coordinate_comparison.png")
    print(f"Plot saved to: {os.path.join(output_dir, 'coordinate_comparison.png')}")
    
    # Coordinates with distances plot
    fig4 = visualizer.plot_with_distances(result['coordinates'], distances)
    visualizer.save_plot(fig4, "coordinates_with_distances.png")
    print(f"Plot saved to: {os.path.join(output_dir, 'coordinates_with_distances.png')}")
    
    print(f"  Visualizations saved to: {output_dir}")
    
    # STEP 5: Run Jacobian sensitivity analysis
    print("\nSTEP 5: Running Jacobian sensitivity analysis...")
    print("  This will analyze local sensitivity of coordinates to distance changes...")
    
    sensitivity_analyzer = MeasurementSensitivityAnalyzer(
        distances, 
        initial_coordinates, 
        fixed_points_info
    )
    
    # Run baseline optimization
    baseline_result = sensitivity_analyzer.run_baseline_optimization()
    
    # Run Jacobian analysis only
    print("\nRunning Jacobian sensitivity analysis...")
    jacobian_results = sensitivity_analyzer.compute_jacobian_sensitivity()
    
    # Print Jacobian summary
    sensitivity_analyzer.print_jacobian_summary(jacobian_results, top_n=15)
    
    # Export Jacobian results
    jacobian_filename = os.path.join(output_dir, "jacobian_sensitivity_analysis.xlsx")
    sensitivity_analyzer.export_jacobian_results(jacobian_results, jacobian_filename)
    print(f"  Jacobian analysis saved to: {jacobian_filename}")
    
    # STEP 6: Final summary
    print("\nSTEP 6: Analysis complete!")
    print("=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Data file: {data_file}")
    print(f"Output directory: {output_dir}")
    print()
    print("Files generated:")
    print(f"  - enhanced_analysis.xlsx (Enhanced distance and coordinate analysis)")
    print(f"  - jacobian_sensitivity_analysis.xlsx (Jacobian sensitivity analysis)")
    print(f"  - initial_coordinates.png (Initial coordinate visualization)")
    print(f"  - optimized_coordinates.png (Optimized coordinate visualization)")
    print(f"  - coordinate_comparison.png (Before/after comparison)")
    print(f"  - coordinates_with_distances.png (Coordinates with distance measurements)")
    print()
    print("Key results:")
    print(f"  - Final RMS error: {result['error_metrics']['root_mean_square_error']:.6f}")
    print(f"  - Total measurements analyzed: {len(distances)}")
    print(f"  - Total points optimized: {len(result['coordinates'])}")
    
    # Get top Jacobian measurement
    if jacobian_results['sensitivities']:
        top_jacobian = jacobian_results['sensitivities'][0]
        print(f"  - Top Jacobian measurement: {top_jacobian['measurement_pair'][0]}-{top_jacobian['measurement_pair'][1]} (Sensitivity: {top_jacobian['sensitivity']:.6f})")
    
    print()
    print("Next steps:")
    print("  - Review enhanced_analysis.xlsx for detailed distance and coordinate analysis")
    print("  - Review jacobian_sensitivity_analysis.xlsx for local measurement sensitivity")
    print("  - Focus on measurements that rank high in Jacobian analysis")
    print("  - Use visualizations to verify coordinate results")
    print()
    print("Analysis completed successfully!")


if __name__ == "__main__":
    main() 