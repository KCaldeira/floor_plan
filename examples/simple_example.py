#!/usr/bin/env python3
"""
Simple example demonstrating the floor plan coordinate estimation system.

This example:
1. Loads data from the sample Excel file
2. Performs coordinate optimization
3. Displays results and creates visualizations
"""

import sys
import os

# Add the parent directory to the Python path so we can import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import modules
from src.data_loader import ExcelDataLoader
from src.coordinate_estimator import CoordinateEstimator
from src.distance_calculator import DistanceCalculator
from src.visualization import FloorPlanVisualizer


def main():
    """Run the simple example."""
    print("=== Floor Plan Coordinate Estimation - Simple Example ===\n")
    
    # Step 1: Load data from Excel file
    print("Step 1: Loading data from Excel file...")
    try:
        loader = ExcelDataLoader("data/sample_measurements.xlsx")
        distances, initial_coords = loader.load_data()
        
        print(f"✓ Loaded {len(distances)} distance measurements")
        print(f"✓ Loaded {len(initial_coords)} initial coordinates")
        
        # Check connectivity
        connected_components, is_fully_connected = loader.get_connectivity_info()
        if is_fully_connected:
            print("✓ All points are connected")
        else:
            print(f"⚠️  Warning: Found {len(connected_components)} disconnected components")
            for i, component in enumerate(connected_components):
                print(f"   Component {i+1}: {component}")
        
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return
    
    print()
    
    # Step 2: Validate distance measurements
    print("Step 2: Validating distance measurements...")
    calculator = DistanceCalculator()
    validation_issues = calculator.validate_distances(distances)
    
    if validation_issues:
        print("⚠️  Validation warnings:")
        for issue in validation_issues:
            print(f"   - {issue}")
    else:
        print("✓ All distance measurements are valid")
    
    # Get distance statistics
    stats = calculator.get_distance_statistics(distances)
    print(f"   Distance statistics:")
    print(f"     Min: {stats['min_distance']:.2f}")
    print(f"     Max: {stats['max_distance']:.2f}")
    print(f"     Mean: {stats['mean_distance']:.2f}")
    
    print()
    
    # Step 3: Perform coordinate optimization
    print("Step 3: Performing coordinate optimization...")
    try:
        # Get fixed points information from the data loader
        fixed_points_info = loader.get_fixed_points_info()
        print(f"Fixed points: {fixed_points_info}")
        
        estimator = CoordinateEstimator(distances, initial_coords, fixed_points_info)
        result = estimator.optimize()
        
        if result['success']:
            print("✓ Optimization completed successfully")
        else:
            print("⚠️  Optimization completed with warnings")
        
        # Display optimization summary
        summary = estimator.get_optimization_summary(result)
        print("\n" + summary)
        
    except Exception as e:
        print(f"❌ Error during optimization: {str(e)}")
        return
    
    print()
    
    # Step 4: Calculate final error metrics
    print("Step 4: Calculating final error metrics...")
    final_coords = result['coordinates']
    error_metrics = result['error_metrics']
    
    print("Final error metrics:")
    print(f"  Mean Absolute Error: {error_metrics['mean_absolute_error']:.6f}")
    print(f"  Root Mean Square Error: {error_metrics['root_mean_square_error']:.6f}")
    print(f"  Maximum Error: {error_metrics['max_error']:.6f}")
    
    print()
    
    # Step 5: Create visualizations
    print("Step 5: Creating visualizations...")
    visualizer = FloorPlanVisualizer()
    
    # Plot initial coordinates
    fig1 = visualizer.plot_coordinates(initial_coords, "Initial Coordinates")
    visualizer.save_plot(fig1, "initial_coordinates.png")
    
    # Plot optimized coordinates with distances
    fig2 = visualizer.plot_with_distances(final_coords, distances, "Optimized Coordinates with Distances")
    visualizer.save_plot(fig2, "optimized_coordinates.png")
    
    # Plot comparison
    fig3 = visualizer.plot_comparison(initial_coords, final_coords)
    visualizer.save_plot(fig3, "coordinate_comparison.png")
    
    # Plot error analysis
    errors = calculator.calculate_errors(final_coords, distances)
    fig4 = visualizer.plot_error_analysis(errors)
    visualizer.save_plot(fig4, "error_analysis.png")
    
    output_dir = visualizer.get_output_directory()
    print("✓ Visualizations saved in organized output directory:")
    print(f"  Output directory: {output_dir}")
    print("  - initial_coordinates.png")
    print("  - optimized_coordinates.png")
    print("  - coordinate_comparison.png")
    print("  - error_analysis.png")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main() 