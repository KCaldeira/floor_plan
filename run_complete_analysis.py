"""
Complete analysis wrapper script.

This script runs both the main coordinate estimation analysis and the sensitivity analysis
in sequence, providing a complete analysis workflow.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import ExcelDataLoader
from src.coordinate_estimator import CoordinateEstimator
from src.output_analyzer import OutputAnalyzer
from src.measurement_sensitivity_analyzer import MeasurementSensitivityAnalyzer
from src.visualization import FloorPlanVisualizer
import datetime


def run_complete_analysis(data_file: str = None):
    """
    Run complete analysis including coordinate estimation and sensitivity analysis.
    
    Args:
        data_file: Path to Excel file with distance measurements and coordinates
    """
    
    # Use default data file if none provided
    if data_file is None:
        data_file = "data/50_e_1_st_measurements.xlsx"
    
    print("="*80)
    print("COMPLETE FLOOR PLAN ANALYSIS")
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
    original_distances_df, original_coordinates_df = loader.get_original_dataframes()
    
    print(f"  Loaded {len(distances)} distance measurements")
    print(f"  Loaded {len(initial_coordinates)} points")
    print(f"  Fixed points: {len([k for k, v in fixed_points_info.items() if v in ['origin', 'y_axis']])}")
    print()
    
    # ============================================================================
    # STEP 2: MAIN COORDINATE ESTIMATION
    # ============================================================================
    print("STEP 2: Running main coordinate estimation...")
    estimator = CoordinateEstimator(distances, initial_coordinates, fixed_points_info)
    result = estimator.optimize()
    
    print(f"  Optimization completed successfully")
    print(f"  Final RMS error: {result['error_metrics']['root_mean_square_error']:.6f}")
    print(f"  Iterations: {result['optimization_info']['nit']}")
    print(f"  Function evaluations: {result['optimization_info']['nfev']}")
    print()
    
    # ============================================================================
    # STEP 3: ENHANCED OUTPUT ANALYSIS
    # ============================================================================
    print("STEP 3: Generating enhanced output analysis...")
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
    # STEP 4: VISUALIZATION
    # ============================================================================
    print("STEP 4: Creating visualizations...")
    
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
    # STEP 5: SENSITIVITY ANALYSIS
    # ============================================================================
    print("STEP 5: Running sensitivity analysis...")
    print("  This will analyze each measurement's impact on coordinate positions...")
    print("  Running both leave-one-out and Jacobian sensitivity analyses...")
    print()
    
    sensitivity_analyzer = MeasurementSensitivityAnalyzer(
        distances, 
        initial_coordinates, 
        fixed_points_info
    )
    
    # Run complete sensitivity analysis (both leave-one-out and Jacobian)
    sensitivity_results = sensitivity_analyzer.run_complete_sensitivity_analysis()
    
    # Export leave-one-out results
    leave_one_out_file = os.path.join(output_dir, "leave_one_out_sensitivity_analysis.xlsx")
    sensitivity_analyzer.export_sensitivity_results(leave_one_out_file)
    
    # Export Jacobian results
    jacobian_file = os.path.join(output_dir, "jacobian_sensitivity_analysis.xlsx")
    sensitivity_analyzer.export_jacobian_results(sensitivity_results['jacobian_results'], jacobian_file)
    
    print(f"  Leave-one-out analysis saved to: {leave_one_out_file}")
    print(f"  Jacobian analysis saved to: {jacobian_file}")
    print()
    
    # Compare analysis methods
    print("  Comparing analysis methods...")
    leave_one_out_top = sensitivity_analyzer.get_top_impactful_measurements(10)
    jacobian_top = sensitivity_results['jacobian_results']['sensitivities'][:10]
    
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
    # STEP 6: FINAL SUMMARY
    # ============================================================================
    print("STEP 6: Analysis complete!")
    print("="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print(f"Data file: {data_file}")
    print(f"Output directory: {output_dir}")
    print()
    print("Files generated:")
    print(f"  - enhanced_analysis.xlsx (Enhanced distance and coordinate analysis)")
    print(f"  - leave_one_out_sensitivity_analysis.xlsx (Leave-one-out sensitivity analysis)")
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
    
    # Get top impactful measurements from both methods
    leave_one_out_top = sensitivity_analyzer.get_top_impactful_measurements(5)
    jacobian_top = sensitivity_results['jacobian_results']['sensitivities'][:5]
    
    print(f"  - Top leave-one-out measurement: {leave_one_out_top[0]['point1']}-{leave_one_out_top[0]['point2']} "
          f"(RMS displacement: {leave_one_out_top[0]['rms_displacement']:.6f})")
    print(f"  - Top Jacobian measurement: {jacobian_top[0]['measurement_pair'][0]}-{jacobian_top[0]['measurement_pair'][1]} "
          f"(Sensitivity: {jacobian_top[0]['sensitivity']:.6f})")
    print()
    print("Next steps:")
    print("  - Review enhanced_analysis.xlsx for detailed distance and coordinate analysis")
    print("  - Review leave_one_out_sensitivity_analysis.xlsx for global measurement impact")
    print("  - Review jacobian_sensitivity_analysis.xlsx for local measurement sensitivity")
    print("  - Focus on measurements that rank high in both analyses")
    print("  - Consider re-measuring high-impact measurements if accuracy is poor")
    print("  - Use visualizations to verify coordinate results")
    print()
    print("Analysis completed successfully!")
    
    return {
        'result': result,
        'enhanced_distances': enhanced_distances,
        'enhanced_coordinates': enhanced_coordinates,
        'sensitivity_analyzer': sensitivity_analyzer,
        'output_dir': output_dir
    }


def main():
    """Main function to run complete analysis."""
    
    # Check for command line argument
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        # Use default data file
        data_file = "data/50_e_1_st_measurements.xlsx"
    
    # Check if data file exists
    if not os.path.exists(data_file):
        print(f"Error: Data file not found: {data_file}")
        print("Available data files:")
        data_dir = "data"
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith('.xlsx'):
                    print(f"  - data/{file}")
        return
    
    # Run complete analysis
    try:
        results = run_complete_analysis(data_file)
        print("Analysis completed successfully!")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 