"""
Coordinate estimator module for solving the floor plan coordinate estimation problem.

This module implements:
- Nonlinear least squares optimization to find optimal coordinates
- Constraint handling for fixed points
- Objective function based on distance errors
- Coordinate reconstruction from optimization variables
"""

import numpy as np
import scipy.optimize as optimize
from typing import Dict, Tuple, List, Optional
from .distance_calculator import DistanceCalculator


class CoordinateEstimator:
    """
    Estimates optimal coordinates for points using distance measurements.
    
    This class implements the core optimization algorithm:
    1. Formulates the problem as nonlinear least squares
    2. Handles fixed point constraints (one at origin, one at (0,y))
    3. Optimizes coordinates to minimize distance errors
    4. Provides detailed results and error metrics
    """
    
    def __init__(self, distances: Dict[Tuple[str, str], float], 
                 initial_coordinates: Dict[str, Tuple[float, float]],
                 fixed_points_info: Optional[Dict[str, str]] = None,
                 line_orientations: Optional[Dict[Tuple[str, str], str]] = None,
                 line_orientation_weight: float = 1.0,
                 enable_line_orientation: bool = True,
                 weights: Optional[Dict[Tuple[str, str], float]] = None):
        """
        Initialize the coordinate estimator.
        
        Args:
            distances: Dictionary mapping (point_id1, point_id2) to measured distances
            initial_coordinates: Dictionary mapping point IDs to initial (x, y) coordinates
            fixed_points_info: Optional dictionary mapping point IDs to constraint types:
                              "origin", "y_axis", or "free"
            line_orientations: Optional dictionary mapping (point_id1, point_id2) to line orientation:
                              "H" for horizontal, "V" for vertical, empty string for none
            line_orientation_weight: Weight for line orientation penalties (default: 1.0)
            enable_line_orientation: Whether to enable line orientation constraints (default: True)
            weights: Optional dictionary mapping (point_id1, point_id2) to weight values.
                    Default weight is 1.0 if not specified.
        """
        self.distances = distances
        self.initial_coordinates = initial_coordinates
        self.distance_calculator = DistanceCalculator()
        
        # Line orientation parameters
        self.line_orientations = line_orientations or {}
        self.line_orientation_weight = line_orientation_weight
        self.enable_line_orientation = enable_line_orientation
        
        # Weights for measurements
        self.weights = weights or {}
        # Set default weight of 1.0 for any measurement not in weights
        for pair in self.distances.keys():
            if pair not in self.weights:
                self.weights[pair] = 1.0
        
        # Extract point IDs and determine fixed points
        self.point_ids = self._extract_point_ids()
        self.fixed_points, self.y_axis_points = self.extract_fixed_points(initial_coordinates, fixed_points_info)
        self.free_points = self._identify_free_points()
        
        # Validate the setup
        self._validate_setup()
    
    def _extract_point_ids(self) -> List[str]:
        """
        Extract all unique point IDs from the data.
        
        Returns:
            List of all point IDs
        """
        point_ids = set()
        
        # Add points from distance measurements
        for pair in self.distances.keys():
            point_ids.add(pair[0])
            point_ids.add(pair[1])
        
        # Add points from initial coordinates
        point_ids.update(self.initial_coordinates.keys())
        
        return sorted(list(point_ids))
    
    @staticmethod
    def extract_fixed_points(initial_coordinates: Dict[str, Tuple[float, float]], 
                           fixed_points_info: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, Tuple[float, float]], List[str]]:
        """
        Extract fixed points and y-axis points from initial coordinates and fixed points info.
        
        Args:
            initial_coordinates: Dictionary mapping point IDs to initial (x, y) coordinates
            fixed_points_info: Optional dictionary mapping point IDs to constraint types
            
        Returns:
            Tuple of (fixed_points_dict, y_axis_points_list)
        """
        fixed_points = {}
        y_axis_points = []
        
        if fixed_points_info is not None:
            for point_id, constraint_type in fixed_points_info.items():
                if constraint_type.lower() in ["origin", "o"]:
                    fixed_points[point_id] = (0.0, 0.0)
                elif constraint_type.lower() in ["y_axis", "y"]:
                    # Validate the initial position
                    x, y = initial_coordinates[point_id]
                    if abs(x) >= 1e-10:
                        raise ValueError(f"Y-axis point {point_id} must have x=0, got x={x}")
                    if y <= 0:
                        raise ValueError(f"Y-axis point {point_id} must have y>0, got y={y}")
                    fixed_points[point_id] = (0.0, y)
                    y_axis_points.append(point_id)
        
        return fixed_points, y_axis_points
    
    def _identify_free_points(self) -> List[str]:
        """
        Identify points that are free to optimize (not fixed).
        
        Returns:
            List of point IDs that are free to optimize
        """
        free_points = []
        for point_id in self.point_ids:
            if point_id not in self.fixed_points:
                free_points.append(point_id)
        return sorted(free_points)
    
    def _validate_setup(self):
        """
        Validate the optimization setup.
        
        Raises:
            ValueError: If setup is invalid
        """
        # Check that we have at least one origin point and one y-axis point
        origin_points = [pid for pid, constraint in self.fixed_points.items() 
                        if constraint == (0.0, 0.0)]
        y_axis_points = [pid for pid in self.y_axis_points]
        
        if not origin_points:
            raise ValueError("At least one point must be fixed at origin (0,0)")
        if not y_axis_points:
            raise ValueError("At least one point must be fixed on y-axis (0,y) where y>0")
        
        # Check that all points in distance measurements have coordinates
        missing_points = []
        for pair in self.distances.keys():
            point1, point2 = pair
            if point1 not in self.initial_coordinates:
                missing_points.append(point1)
            if point2 not in self.initial_coordinates:
                missing_points.append(point2)
        
        if missing_points:
            raise ValueError(f"Points in distance measurements missing from coordinates: {missing_points}")
        
        # Check that we have at least one free point to optimize
        if not self.free_points:
            raise ValueError("No free points to optimize")
    
    def _objective_function(self, coords_vector: np.ndarray) -> np.ndarray:
        """
        Objective function for the optimization.
        
        This function calculates the errors between measured and calculated distances
        for all distance measurements, plus line orientation penalties.
        The optimization minimizes the sum of squared errors.
        
        Args:
            coords_vector: Flattened array of free coordinates [x1, y1, x2, y2, ...]
            
        Returns:
            Array of errors for each measured distance plus line orientation penalties
        """
        # Reconstruct the full coordinate dictionary
        coordinates = self._reconstruct_coordinates(coords_vector)
        
        # Calculate errors for all measured distances
        errors = []
        for pair, measured_distance in self.distances.items():
            point1_id, point2_id = pair
            
            # Get coordinates for both points
            point1_coords = coordinates[point1_id]
            point2_coords = coordinates[point2_id]
            
            # Calculate the distance between these points
            calculated_distance = self.distance_calculator.euclidean_distance(
                point1_coords, point2_coords
            )
            
            # Calculate the error (measured - calculated)
            error = measured_distance - calculated_distance
            
            # Apply weight to the error
            weight = self.weights.get(pair, 1.0)
            weighted_error = error * weight
            errors.append(weighted_error)
        
        # Add line orientation penalties if enabled
        if self.enable_line_orientation and self.line_orientations:
            for pair, orientation in self.line_orientations.items():
                if orientation in ['H', 'V']:
                    point1_id, point2_id = pair
                    
                    # Get coordinates for both points
                    point1_coords = coordinates[point1_id]
                    point2_coords = coordinates[point2_id]
                    
                    # Calculate line orientation penalty
                    if orientation == 'H':
                        # Horizontal line: penalize y-coordinate differences
                        penalty = self.line_orientation_weight * (point1_coords[1] - point2_coords[1])
                    else:  # orientation == 'V'
                        # Vertical line: penalize x-coordinate differences
                        penalty = self.line_orientation_weight * (point1_coords[0] - point2_coords[0])
                    
                    # Apply weight to the line orientation penalty
                    # For structural lines without distance measurements, use default weight of 1.0
                    weight = self.weights.get(pair, 1.0)
                    weighted_penalty = penalty * weight
                    errors.append(weighted_penalty)
        
        return np.array(errors)
    
    def _get_initial_vector(self) -> np.ndarray:
        """
        Create the initial optimization vector from free point coordinates.
        
        Returns:
            Flattened array of initial coordinates for free points
        """
        initial_vector = []
        
        for point_id in self.free_points:
            x, y = self.initial_coordinates[point_id]
            
            # For y_axis points, only include y-coordinate
            if point_id in self.y_axis_points:
                initial_vector.append(y)
            else:
                # For regular free points, both x and y are in vector
                initial_vector.extend([x, y])
        
        return np.array(initial_vector)
    
    def _reconstruct_coordinates(self, coords_vector: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """
        Reconstruct the full coordinate dictionary from the optimization vector.
        
        Args:
            coords_vector: Flattened array of free coordinates [x1, y1, x2, y2, ...]
            
        Returns:
            Complete coordinate dictionary including fixed and free points
        """
        # Start with fixed points
        coordinates = self.fixed_points.copy()
        
        # Add free points from the optimization vector
        vector_idx = 0
        for point_id in self.free_points:
            # For y_axis points, only y-coordinate is in vector
            if point_id in self.y_axis_points:
                y = coords_vector[vector_idx]
                coordinates[point_id] = (0.0, y)  # x=0 is fixed
                vector_idx += 1
            else:
                # For regular free points, both x and y are in vector
                x = coords_vector[vector_idx]
                y = coords_vector[vector_idx + 1]
                coordinates[point_id] = (x, y)
                vector_idx += 2
        
        return coordinates
    
    def optimize(self, max_iterations: int = 1000, tolerance: float = 1e-8) -> Dict:
        """
        Optimize coordinates to minimize distance errors.
        
        Args:
            max_iterations: Maximum number of iterations for optimization
            tolerance: Convergence tolerance for optimization
        
        Returns:
            Dictionary containing optimization results:
            - 'coordinates': Final optimized coordinates
            - 'error_metrics': Various error metrics
            - 'optimization_info': Optimization algorithm information
        """
        # Get initial vector for optimization
        initial_vector = self._get_initial_vector()
        
        if len(initial_vector) == 0:
            raise ValueError("No free coordinates to optimize")
        
        # Run optimization
        result = optimize.least_squares(
            self._objective_function,
            initial_vector,
            method='trf',  # Trust Region Reflective algorithm
            max_nfev=max_iterations,
            ftol=tolerance,
            xtol=tolerance
        )
        
        # Reconstruct final coordinates
        final_coordinates = self._reconstruct_coordinates(result.x)
        
        # Calculate error metrics
        error_metrics = self._calculate_error_metrics(final_coordinates)
        
        # Handle different optimization methods that have different attributes
        optimization_info = {
            'success': result.success,
            'nfev': result.nfev,
            'cost': result.cost,
            'message': result.message
        }
        
        # Add iteration count if available (not available for 'trf' method)
        if hasattr(result, 'nit'):
            optimization_info['nit'] = result.nit
        else:
            optimization_info['nit'] = 'N/A'  # Not available for TRF method
        
        return {
            'coordinates': final_coordinates,
            'error_metrics': error_metrics,
            'optimization_info': optimization_info
        }
    
    def _calculate_error_metrics(self, coordinates: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """
        Calculate various error metrics for the optimized coordinates.
        
        Args:
            coordinates: Dictionary mapping point IDs to (x, y) coordinates
            
        Returns:
            Dictionary containing error metrics
        """
        errors = []
        for pair, measured_distance in self.distances.items():
            point1_id, point2_id = pair
            
            # Get coordinates for both points
            point1_coords = coordinates[point1_id]
            point2_coords = coordinates[point2_id]
            
            # Calculate the distance between these points
            calculated_distance = self.distance_calculator.euclidean_distance(
                point1_coords, point2_coords
            )
            
            # Calculate the error
            error = measured_distance - calculated_distance
            errors.append(error)
        
        errors = np.array(errors)
        
        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'root_mean_square_error': np.sqrt(np.mean(errors**2)),
            'max_error': np.max(np.abs(errors)),
            'min_error': np.min(errors),
            'total_error': np.sum(np.abs(errors))
        }
    
    def get_optimization_summary(self, result: Dict) -> str:
        """
        Generate a summary of the optimization results.
        
        Args:
            result: Dictionary containing optimization results
            
        Returns:
            Formatted summary string
        """
        coords = result['coordinates']
        metrics = result['error_metrics']
        info = result['optimization_info']
        
        summary = f"""
Optimization Summary:
===================
Success: {info['success']}
Iterations: {info['nit']}
Function evaluations: {info['nfev']}
Final cost: {info['cost']:.6f}

Error Metrics:
=============
Mean error: {metrics['mean_error']:.6f}
Std error: {metrics['std_error']:.6f}
RMS error: {metrics['root_mean_square_error']:.6f}
Max absolute error: {metrics['max_error']:.6f}
Total absolute error: {metrics['total_error']:.6f}

Coordinates:
===========
"""
        for point_id in sorted(coords.keys()):
            x, y = coords[point_id]
            summary += f"{point_id}: ({x:.6f}, {y:.6f})\n"
        
        return summary 