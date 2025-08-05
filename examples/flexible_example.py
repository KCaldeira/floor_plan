#!/usr/bin/env python3
"""
Flexible example demonstrating the floor plan coordinate estimation system
with explicit fixed point designation.

This example shows how to use the "Fixed Point" column to designate
which points should be fixed at (0,0) and (0,y) without relying on row order.
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
    """Run the flexible example with explicit fixed point designation."""
    print("=== Floor Plan Coordinate Estimation - Flexible Example ===\n")
    
    # Step 1: Load data from Excel file with flexible fixed point designation
    print("Step 1: Loading data from Excel file with flexible fixed point designation...")
    try:
        loader = ExcelDataLoader("data/flexible_example.xlsx")
        distances, initial_coords = loader.load_data()
        
        print(f"✓ Loaded {len(distances)} distance measurements")
        print(f"✓ Loaded {len(initial_coords)} initial coordinates")
        
        # Get and display fixed points information
        fixed_points_info = loader.get_fixed_points_info()
        print(f"✓ Fixed points designation: {fixed_points_info}")
        
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
    
    # Step 3: Perform coordinate optimization with explicit fixed point info
    print("Step 3: Performing coordinate optimization with flexible fixed point designation...")
    try:
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
    fig1 = visualizer.plot_coordinates(initial_coords, "Initial Coordinates (Flexible)")
    visualizer.save_plot(fig1, "flexible_initial_coordinates.png")
    
    # Plot optimized coordinates with distances
    fig2 = visualizer.plot_with_distances(final_coords, distances, "Flexible Optimized Coordinates")
    visualizer.save_plot(fig2, "flexible_optimized_coordinates.png")
    
    # Plot comparison
    fig3 = visualizer.plot_comparison(initial_coords, final_coords)
    visualizer.save_plot(fig3, "flexible_coordinate_comparison.png")
    
    # Plot error analysis
    errors = calculator.calculate_errors(final_coords, distances)
    fig4 = visualizer.plot_error_analysis(errors)
    visualizer.save_plot(fig4, "flexible_error_analysis.png")
    
    output_dir = visualizer.get_output_directory()
    print("✓ Visualizations saved in organized output directory:")
    print(f"  Output directory: {output_dir}")
    print("  - flexible_initial_coordinates.png")
    print("  - flexible_optimized_coordinates.png")
    print("  - flexible_coordinate_comparison.png")
    print("  - flexible_error_analysis.png")
    
    print("\n=== Flexible Example completed successfully! ===")
    print("\nKey benefits of flexible fixed point designation:")
    print("- Row order doesn't matter")
    print("- Explicit control over which points are fixed")
    print("- Clear documentation of constraints")
    print("- Easy to modify without reordering data")


if __name__ == "__main__":
    main() 