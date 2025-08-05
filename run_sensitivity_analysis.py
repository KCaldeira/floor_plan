"""
Simple script to run sensitivity analysis on real data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import ExcelDataLoader
from src.measurement_sensitivity_analyzer import MeasurementSensitivityAnalyzer
from src.visualization import FloorPlanVisualizer
import datetime


def main():
    """Run sensitivity analysis on real data."""
    
    # Use the real data file
    data_file = "data/50_e_1_st_measurements.xlsx"
    
    print("="*60)
    print("SENSITIVITY ANALYSIS ON REAL DATA")
    print("="*60)
    print(f"Data file: {data_file}")
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
    print("Running baseline optimization...")
    baseline_result = analyzer.run_baseline_optimization()
    print(f"Baseline optimization completed successfully")
    print(f"Final RMS error: {baseline_result['error_metrics']['root_mean_square_error']:.6f}")
    print()
    
    # Run sensitivity analysis
    print("Starting sensitivity analysis...")
    print("This will analyze each of the 55 measurements individually...")
    sensitivity_results = analyzer.analyze_measurement_sensitivity()
    print("Sensitivity analysis completed!")
    print()
    
    # Print summary
    analyzer.print_impact_summary(top_n=20)
    
    # Create visualizer to get output directory
    visualizer = FloorPlanVisualizer()
    output_dir = visualizer.get_output_directory()
    
    # Export results to the same output directory
    output_file = f"sensitivity_analysis_real_data.xlsx"
    output_path = os.path.join(output_dir, output_file)
    analyzer.export_sensitivity_results(output_path)
    print(f"\nResults exported to: {output_path}")
    
    # Get detailed information about top measurements
    print("\n" + "="*60)
    print("DETAILED ANALYSIS OF TOP 10 MEASUREMENTS")
    print("="*60)
    
    top_measurements = analyzer.get_top_impactful_measurements(10)
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
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("The sensitivity analysis has identified which of the 55 distance")
    print("measurements have the greatest impact on final coordinate positions.")
    print()
    print("Key findings:")
    print("- Measurements with high RMS displacement cause the most movement")
    print("- These measurements may be critical for coordinate accuracy")
    print("- Consider re-measuring high-impact measurements if accuracy is poor")
    print("- Low-impact measurements may be redundant or less critical")
    print()
    print(f"Detailed results saved to: {output_path}")


if __name__ == "__main__":
    main() 