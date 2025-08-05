"""
Visualization module for plotting coordinate estimation results.

This module provides:
- Basic plotting of point locations
- Visualization of distance measurements
- Comparison of initial vs optimized coordinates
- Error visualization
- Organized output directory management
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple, List
import matplotlib.patches as patches
import os
import datetime


class FloorPlanVisualizer:
    """
    Visualizes floor plan coordinate estimation results.
    
    This class provides methods to create plots showing:
    - Point locations (initial and optimized)
    - Distance measurements between points
    - Error analysis
    - Optimization convergence
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory for saving plots. If None, creates timestamped directory.
        """
        # Set up matplotlib style for better plots
        plt.style.use('default')
        
        # Set up output directory
        if output_dir is None:
            # Create timestamped output directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"output/run_{timestamp}"
        else:
            self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Output directory: {self.output_dir}")
    
    def plot_coordinates(self, coordinates: Dict[str, Tuple[float, float]], 
                        title: str = "Point Coordinates",
                        show_labels: bool = True,
                        figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Create a basic plot of point coordinates.
        
        Args:
            coordinates: Dictionary mapping point IDs to (x, y) coordinates
            title: Title for the plot
            show_labels: Whether to show point ID labels
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract coordinates for plotting
        point_ids = list(coordinates.keys())
        x_coords = [coordinates[pid][0] for pid in point_ids]
        y_coords = [coordinates[pid][1] for pid in point_ids]
        
        # Plot points
        ax.scatter(x_coords, y_coords, c='blue', s=100, alpha=0.7, edgecolors='black')
        
        # Add point labels
        if show_labels:
            for i, point_id in enumerate(point_ids):
                ax.annotate(point_id, (x_coords[i], y_coords[i]), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=12, fontweight='bold')
        
        # Set plot properties
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add origin lines
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        return fig
    
    def plot_with_distances(self, coordinates: Dict[str, Tuple[float, float]], 
                           distances: Dict[Tuple[str, str], float],
                           title: str = "Points with Distance Measurements",
                           figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot points with lines showing distance measurements.
        
        Args:
            coordinates: Dictionary mapping point IDs to (x, y) coordinates
            distances: Dictionary mapping (point_id1, point_id2) to distances
            title: Title for the plot
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract coordinates for plotting
        point_ids = list(coordinates.keys())
        x_coords = [coordinates[pid][0] for pid in point_ids]
        y_coords = [coordinates[pid][1] for pid in point_ids]
        
        # Plot points
        ax.scatter(x_coords, y_coords, c='blue', s=100, alpha=0.7, edgecolors='black')
        
        # Add point labels
        for i, point_id in enumerate(point_ids):
            ax.annotate(point_id, (x_coords[i], y_coords[i]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=12, fontweight='bold')
        
        # Draw distance lines
        for pair, distance in distances.items():
            point1_id, point2_id = pair
            
            # Get coordinates for both points
            x1, y1 = coordinates[point1_id]
            x2, y2 = coordinates[point2_id]
            
            # Draw line between points
            ax.plot([x1, x2], [y1, y2], 'r-', alpha=0.6, linewidth=2)
            
            # Add distance label at midpoint
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            ax.annotate(f'{distance:.2f}', (mid_x, mid_y), 
                       xytext=(0, 5), textcoords='offset points',
                       ha='center', fontsize=10, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Set plot properties
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add origin lines
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        return fig
    
    def plot_comparison(self, initial_coords: Dict[str, Tuple[float, float]], 
                       optimized_coords: Dict[str, Tuple[float, float]],
                       title: str = "Initial vs Optimized Coordinates",
                       figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot comparison between initial and optimized coordinates.
        
        Args:
            initial_coords: Dictionary of initial coordinates
            optimized_coords: Dictionary of optimized coordinates
            title: Title for the plot
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract coordinates
        point_ids = list(optimized_coords.keys())
        
        # Initial coordinates
        x_init = [initial_coords[pid][0] for pid in point_ids]
        y_init = [initial_coords[pid][1] for pid in point_ids]
        
        # Optimized coordinates
        x_opt = [optimized_coords[pid][0] for pid in point_ids]
        y_opt = [optimized_coords[pid][1] for pid in point_ids]
        
        # Plot initial points
        ax.scatter(x_init, y_init, c='red', s=100, alpha=0.7, 
                  edgecolors='black', label='Initial', marker='o')
        
        # Plot optimized points
        ax.scatter(x_opt, y_opt, c='green', s=100, alpha=0.7, 
                  edgecolors='black', label='Optimized', marker='s')
        
        # Draw arrows from initial to optimized positions
        for i, point_id in enumerate(point_ids):
            ax.annotate('', xy=(x_opt[i], y_opt[i]), xytext=(x_init[i], y_init[i]),
                       arrowprops=dict(arrowstyle='->', color='blue', alpha=0.6, lw=1.5))
        
        # Add point labels
        for i, point_id in enumerate(point_ids):
            ax.annotate(point_id, (x_opt[i], y_opt[i]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=12, fontweight='bold')
        
        # Set plot properties
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.legend()
        
        # Add origin lines
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        return fig
    
    def plot_error_analysis(self, errors: Dict[Tuple[str, str], float],
                           title: str = "Distance Error Analysis",
                           figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Create error analysis plots.
        
        Args:
            errors: Dictionary mapping (point_id1, point_id2) to error values
            title: Title for the plot
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Extract error values
        error_values = list(errors.values())
        error_pairs = list(errors.keys())
        
        # Histogram of errors
        ax1.hist(error_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Error Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Error Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Bar plot of individual errors
        pair_labels = [f"{pair[0]}-{pair[1]}" for pair in error_pairs]
        bars = ax2.bar(range(len(error_values)), error_values, 
                       color=['red' if e < 0 else 'green' for e in error_values],
                       alpha=0.7)
        ax2.set_xlabel('Point Pairs')
        ax2.set_ylabel('Error Value')
        ax2.set_title('Individual Distance Errors')
        ax2.set_xticks(range(len(pair_labels)))
        ax2.set_xticklabels(pair_labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add zero line
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        fig.suptitle(title, y=1.02)
        
        return fig
    
    def save_plot(self, fig: plt.Figure, filename: str, dpi: int = 300):
        """
        Save a plot to file in the output directory.
        
        Args:
            fig: Matplotlib figure to save
            filename: Output filename (will be saved in output directory)
            dpi: Resolution for the saved image
        """
        # Create full path in output directory
        full_path = os.path.join(self.output_dir, filename)
        fig.savefig(full_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to: {full_path}")
    
    def get_output_directory(self) -> str:
        """
        Get the current output directory.
        
        Returns:
            Path to the output directory
        """
        return self.output_dir
    
    def show_plot(self, fig: plt.Figure):
        """
        Display a plot.
        
        Args:
            fig: Matplotlib figure to display
        """
        plt.show() 