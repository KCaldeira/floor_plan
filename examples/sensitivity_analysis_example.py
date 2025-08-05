"""
Example script demonstrating measurement sensitivity analysis.

This script shows how to use the MeasurementSensitivityAnalyzer to identify
which distance measurements have the greatest impact on final coordinate positions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import ExcelDataLoader
from src.measurement_sensitivity_analyzer import MeasurementSensitivityAnalyzer
from src.visualization import FloorPlanVisualizer
import datetime


def run_sensitivity_analysis(data_file: str, output_dir: str = None):
    """
    Run sensitivity analysis on a dataset.
    
    Args:
        data_file: Path to Excel file with distance measurements and coordinates
        output_dir: Directory for output files (optional)
    """
    print("="*60)
    print("MEASUREMENT SENSITIVITY ANALYSIS")
    print("="*60)
    print(f"Data file: {data_file}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load data
    print("Loading data...")
    loader = ExcelDataLoader(data_file)
    distances, initial_coordinates = loader.load_data()
    fixed_points_info = loader.get_fixed_points_info()
    
    print(f"Loaded {len(distances)} distance measurements")
    print(f"Loaded {len(initial_coordinates)} points")
    print(f"Fixed points: {len([k for k, v in fixed_points_info.items() if v in ['origin', 'y_axis']])}")
    print()
    
    # Create sensitivity analyzer
    analyzer = MeasurementSensitivityAnalyzer(
        distances, 
        initial_coordinates, 
        fixed_points_info
    )
    
    # Run baseline optimization
    baseline_result = analyzer.run_baseline_optimization()
    print(f"Baseline optimization completed successfully")
    print(f"Final RMS error: {baseline_result['error_metrics']['rms_error']:.6f}")
    print()
    
    # Run sensitivity analysis
    print("Starting sensitivity analysis...")
    sensitivity_results = analyzer.analyze_measurement_sensitivity()
    print("Sensitivity analysis completed!")
    print()
    
    # Print summary
    analyzer.print_impact_summary(top_n=15)
    
    # Export results
    if output_dir:
        output_file = os.path.join(output_dir, "sensitivity_analysis.xlsx")
        analyzer.export_sensitivity_results(output_file)
        print(f"\nResults exported to: {output_file}")
    
    # Get detailed information about top measurements
    print("\n" + "="*60)
    print("DETAILED ANALYSIS OF TOP 5 MEASUREMENTS")
    print("="*60)
    
    top_measurements = analyzer.get_top_impactful_measurements(5)
    for i, measurement in enumerate(top_measurements, 1):
        pair = measurement['measurement_pair']
        point1, point2 = pair
        
        print(f"\n{i}. Measurement {point1}-{point2} (Distance: {measurement['original_distance']:.3f})")
        print(f"   RMS Displacement: {measurement['rms_displacement']:.6f}")
        print(f"   Max Displacement: {measurement['max_displacement']:.6f}")
        print(f"   Total Displacement: {measurement['total_displacement']:.6f}")
        
        # Get detailed point displacement information
        result = analyzer.sensitivity_results[pair]
        point_displacements = result['point_displacements']
        
        # Find points with largest displacements
        sorted_points = sorted(
            point_displacements.items(), 
            key=lambda x: x[1]['displacement'], 
            reverse=True
        )
        
        print(f"   Top 3 most affected points:")
        for j, (point_id, displacement_info) in enumerate(sorted_points[:3], 1):
            disp = displacement_info['displacement']
            x_change = displacement_info['x_change']
            y_change = displacement_info['y_change']
            print(f"      {j}. Point {point_id}: displacement={disp:.6f}, "
                  f"dx={x_change:.6f}, dy={y_change:.6f}")
    
    return analyzer


def main():
    """Main function to run sensitivity analysis example."""
    
    # Example with sample data
    data_file = "data/sample_measurements.xlsx"
    
    # Check if sample data exists, otherwise use flexible example
    if not os.path.exists(data_file):
        data_file = "data/flexible_example.xlsx"
    
    if not os.path.exists(data_file):
        print("No sample data files found. Please ensure you have:")
        print("- data/sample_measurements.xlsx")
        print("- data/flexible_example.xlsx")
        print("- data/50_e_1_st_measurements.xlsx")
        return
    
    # Use the visualization module's output directory
    visualizer = FloorPlanVisualizer()
    output_dir = visualizer.get_output_directory()
    
    # Run analysis
    analyzer = run_sensitivity_analysis(data_file, output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("The sensitivity analysis has identified which distance measurements")
    print("have the greatest impact on final coordinate positions.")
    print()
    print("Key findings:")
    print("- Measurements with high RMS displacement cause the most movement")
    print("- These measurements may be critical for coordinate accuracy")
    print("- Consider re-measuring high-impact measurements if accuracy is poor")
    print("- Low-impact measurements may be redundant or less critical")
    print()
    print(f"Detailed results saved to: {output_dir}")


if __name__ == "__main__":
    main() 