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
                 line_constraints: Optional[List[Dict[str, any]]] = None,
                 line_weight: float = 1.0):
        """
        Initialize the coordinate estimator.
        
        Args:
            distances: Dictionary mapping (point_id1, point_id2) to measured distances
            initial_coordinates: Dictionary mapping point IDs to initial (x, y) coordinates
            fixed_points_info: Optional dictionary mapping point IDs to constraint types:
                              "origin", "y_axis", or "free"
            line_constraints: Optional list of line constraint dictionaries
            line_weight: Weight factor for line constraint penalties (default: 1.0)
        """
        self.distances = distances
        self.initial_coordinates = initial_coordinates
        self.distance_calculator = DistanceCalculator()
        self.line_constraints = line_constraints or []
        self.line_weight = line_weight
        
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
                        raise ValueError(f"Point {point_id} designated as y_axis but not at x=0")
                    if y <= 0:
                        raise ValueError(f"Point {point_id} designated as y_axis but y <= 0")
                    y_axis_points.append(point_id)
        else:
            # Fall back to coordinate-based detection
            for point_id, coords in initial_coordinates.items():
                x, y = coords
                
                # Point at origin (0, 0) is fixed
                if abs(x) < 1e-10 and abs(y) < 1e-10:
                    fixed_points[point_id] = (0.0, 0.0)
                
                # Point at (0, y) where y > 0 is fixed (legacy behavior)
                elif abs(x) < 1e-10 and y > 0:
                    fixed_points[point_id] = (0.0, y)
        
        return fixed_points, y_axis_points


    
    def _identify_free_points(self) -> List[str]:
        """
        Identify points that are not fixed and need optimization.
        
        Returns:
            List of point IDs that are free to move
        """
        free_points = []
        for point_id in self.point_ids:
            if point_id not in self.fixed_points:
                free_points.append(point_id)
        
        return free_points
    
    def _validate_setup(self):
        """
        Validate the optimization setup.
        
        Raises:
            ValueError: If setup is invalid
        """
        # Check that we have at least 1 fixed point (origin)
        if len(self.fixed_points) < 1:
            raise ValueError(
                f"Need at least 1 fixed point at (0,0), but found {len(self.fixed_points)}."
            )
        
        # Check that we have at least 1 y_axis point
        if len(self.y_axis_points) < 1:
            raise ValueError(
                f"Need at least 1 y_axis point, but found {len(self.y_axis_points)}."
            )
        
        # Check that we have at least one free point (including y_axis points)
        if len(self.free_points) == 0:
            raise ValueError("No free points to optimize. All points are fixed.")
        
        # Check that all points in distances have coordinates
        distance_points = set()
        for pair in self.distances.keys():
            distance_points.add(pair[0])
            distance_points.add(pair[1])
        
        missing_coords = distance_points - set(self.initial_coordinates.keys())
        if missing_coords:
            raise ValueError(f"Missing initial coordinates for points: {missing_coords}")
    
    def _objective_function(self, coords_vector: np.ndarray) -> np.ndarray:
        """
        Objective function for the optimization.
        
        This function calculates the errors between measured and calculated distances
        for all distance measurements. The optimization minimizes the sum of squared errors.
        
        Args:
            coords_vector: Flattened array of free coordinates [x1, y1, x2, y2, ...]
            
        Returns:
            Array of errors for each measured distance
        """
        # Reconstruct the full coordinate dictionary
        coordinates = self._reconstruct_coordinates(coords_vector, self.constraint_type)
        
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
            errors.append(error)
        
        # Add line constraint penalties
        for line_constraint in self.line_constraints:
            line_penalties = self._calculate_line_penalty(coordinates, line_constraint)
            errors.extend(line_penalties)
        
        return np.array(errors)
    
    def _get_initial_vector(self, constraint_type: str) -> np.ndarray:
        """
        Create the initial optimization vector from free point coordinates.
        
        Args:
            constraint_type: Type of optimization to perform:
                           - "full": Optimize both x and y coordinates
                           - "x_only": Keep y values fixed, optimize x only
                           - "y_only": Keep x values fixed, optimize y only
        
        Returns:
            Flattened array of initial coordinates for free points
        """
        initial_vector = []
        
        for point_id in self.free_points:
            x, y = self.initial_coordinates[point_id]
            
            if constraint_type == "x_only":
                # Only include x-coordinate for x-only optimization
                if point_id in self.y_axis_points:
                    # y_axis points don't have x-coordinate in vector (x=0 is fixed)
                    pass
                else:
                    initial_vector.append(x)
            elif constraint_type == "y_only":
                # Only include y-coordinate for y-only optimization
                if point_id in self.y_axis_points:
                    initial_vector.append(y)
                else:
                    initial_vector.append(y)
            else:  # "full" optimization
                # For y_axis points, only include y-coordinate
                if point_id in self.y_axis_points:
                    initial_vector.append(y)
                else:
                    # For regular free points, both x and y are in vector
                    initial_vector.extend([x, y])
        
        return np.array(initial_vector)
    
    def _calculate_line_penalty(self, coordinates: Dict[str, Tuple[float, float]], 
                               line_constraint: Dict[str, any]) -> List[float]:
        """
        Calculate penalty for points not lying on the specified line.
        
        Args:
            coordinates: Dictionary mapping point IDs to (x, y) coordinates
            line_constraint: Dictionary with keys "line_id", "point_ids", "type"
            
        Returns:
            List of penalty values (one per point in the line group)
        """
        point_ids = line_constraint["point_ids"]
        
        if len(point_ids) < 2:
            return []
        
        # Extract coordinates for points in this line group
        line_points = []
        for point_id in point_ids:
            if point_id in coordinates:
                line_points.append(coordinates[point_id])
            else:
                # Skip points that don't have coordinates
                continue
        
        if len(line_points) < 2:
            return []
        
        # Fit a line through the points using least squares
        x_coords = [p[0] for p in line_points]
        y_coords = [p[1] for p in line_points]
        
        # Use linear regression to fit y = mx + b
        # For vertical lines, we'll handle this as a special case
        x_array = np.array(x_coords)
        y_array = np.array(y_coords)
        
        # Check if this is a vertical line (all x coordinates are nearly equal)
        if np.std(x_array) < 1e-10:
            # Vertical line: x = constant
            x_constant = np.mean(x_array)
            penalties = []
            for point_id in point_ids:
                if point_id in coordinates:
                    x, y = coordinates[point_id]
                    # Penalty is the squared distance from the vertical line
                    penalty = (x - x_constant) ** 2
                    penalties.append(penalty * self.line_weight)
        else:
            # Regular line: fit y = mx + b
            A = np.vstack([x_array, np.ones(len(x_array))]).T
            m, b = np.linalg.lstsq(A, y_array, rcond=None)[0]
            
            penalties = []
            for point_id in point_ids:
                if point_id in coordinates:
                    x, y = coordinates[point_id]
                    # Calculate the y-coordinate on the fitted line
                    y_fitted = m * x + b
                    # Penalty is the squared distance from the line
                    penalty = (y - y_fitted) ** 2
                    penalties.append(penalty * self.line_weight)
        
        return penalties
    
    def _reconstruct_coordinates(self, coords_vector: np.ndarray, constraint_type: str) -> Dict[str, Tuple[float, float]]:
        """
        Reconstruct the full coordinate dictionary from the optimization vector.
        
        Args:
            coords_vector: Flattened array of free coordinates [x1, y1, x2, y2, ...]
            constraint_type: Type of optimization to perform:
                           - "full": Optimize both x and y coordinates
                           - "x_only": Keep y values fixed, optimize x only
                           - "y_only": Keep x values fixed, optimize y only
            
        Returns:
            Complete coordinate dictionary including fixed and free points
        """
        # Start with fixed points
        coordinates = self.fixed_points.copy()
        
        # Add free points from the optimization vector
        vector_idx = 0
        for point_id in self.free_points:
            if constraint_type == "x_only":
                # Only x-coordinate is optimized, y comes from initial coordinates
                if point_id in self.y_axis_points:
                    # y_axis points don't have x-coordinate in vector (x=0 is fixed)
                    y = self.initial_coordinates[point_id][1]
                    coordinates[point_id] = (0.0, y)
                else:
                    x = coords_vector[vector_idx]
                    y = self.initial_coordinates[point_id][1]  # Keep initial y
                    coordinates[point_id] = (x, y)
                    vector_idx += 1
            elif constraint_type == "y_only":
                # Only y-coordinate is optimized, x comes from initial coordinates
                if point_id in self.y_axis_points:
                    y = coords_vector[vector_idx]
                    x = self.initial_coordinates[point_id][0]  # Keep initial x
                    coordinates[point_id] = (x, y)
                    vector_idx += 1
                else:
                    y = coords_vector[vector_idx]
                    x = self.initial_coordinates[point_id][0]  # Keep initial x
                    coordinates[point_id] = (x, y)
                    vector_idx += 1
            else:  # "full" optimization
                # Both x and y coordinates are optimized
                if point_id in self.y_axis_points:
                    # For y_axis points, only y-coordinate is in vector
                    y = coords_vector[vector_idx]
                    coordinates[point_id] = (0.0, y)
                    vector_idx += 1
                else:
                    # For regular free points, both x and y are in vector
                    x = coords_vector[vector_idx]
                    y = coords_vector[vector_idx + 1]
                    coordinates[point_id] = (x, y)
                    vector_idx += 2
        
        return coordinates
    
    def optimize(self, max_iterations: int = 1000, tolerance: float = 1e-8, 
                constraint_type: str = "full") -> Dict:
        """
        Perform the coordinate optimization.
        
        Args:
            max_iterations: Maximum number of optimization iterations
            tolerance: Convergence tolerance for the optimization
            constraint_type: Type of optimization to perform:
                           - "full": Optimize both x and y coordinates
                           - "x_only": Keep y values fixed, optimize x only
                           - "y_only": Keep x values fixed, optimize y only
            
        Returns:
            Dictionary containing:
            - 'coordinates': Optimized coordinates for all points
            - 'success': Whether optimization was successful
            - 'error_metrics': Various error metrics
            - 'optimization_info': Information about the optimization process
        """
        # Store constraint type for use in objective function
        self.constraint_type = constraint_type
        
        # Get initial vector for free points based on constraint type
        initial_vector = self._get_initial_vector(constraint_type)
        
        # Perform the optimization
        try:
            result = optimize.least_squares(
                self._objective_function,
                initial_vector,
                method='trf',  # trust-region-reflective
                max_nfev=max_iterations,
                ftol=tolerance,
                xtol=tolerance
            )
            
            # Reconstruct the final coordinates
            final_coordinates = self._reconstruct_coordinates(result.x, constraint_type)
            
            # Calculate error metrics
            error_metrics = self.distance_calculator.calculate_error_metrics(
                final_coordinates, self.distances
            )
            
            # Prepare optimization info
            optimization_info = {
                'success': result.success,
                'message': result.message,
                'nfev': result.nfev,  # Number of function evaluations
                'nit': getattr(result, 'nit', 0),  # Number of iterations (may not exist)
                'cost': result.cost,   # Final cost (sum of squared errors)
                'optimality': result.optimality,  # Optimality measure
                'constraint_type': constraint_type
            }
            
            return {
                'coordinates': final_coordinates,
                'success': result.success,
                'error_metrics': error_metrics,
                'optimization_info': optimization_info
            }
            
        except Exception as e:
            # Handle optimization failures
            raise RuntimeError(f"Optimization failed: {str(e)}")
    
    def get_optimization_summary(self, result: Dict) -> str:
        """
        Generate a human-readable summary of the optimization results.
        
        Args:
            result: Result dictionary from the optimize() method
            
        Returns:
            Formatted string summary
        """
        coords = result['coordinates']
        error_metrics = result['error_metrics']
        opt_info = result['optimization_info']
        
        summary = []
        summary.append("=== COORDINATE OPTIMIZATION RESULTS ===")
        summary.append("")
        
        # Optimization status
        summary.append("OPTIMIZATION STATUS:")
        summary.append(f"  Success: {result['success']}")
        summary.append(f"  Message: {opt_info['message']}")
        summary.append(f"  Iterations: {opt_info['nit']}")
        summary.append(f"  Function evaluations: {opt_info['nfev']}")
        summary.append(f"  Final cost: {opt_info['cost']:.6f}")
        summary.append("")
        
        # Fixed points
        summary.append("FIXED POINTS:")
        for point_id, coords_tuple in self.fixed_points.items():
            x, y = coords_tuple
            summary.append(f"  {point_id}: ({x:.3f}, {y:.3f})")
        summary.append("")
        
        # Optimized coordinates
        summary.append("OPTIMIZED COORDINATES:")
        for point_id in sorted(coords.keys()):
            if point_id not in self.fixed_points:
                x, y = coords[point_id]
                summary.append(f"  {point_id}: ({x:.3f}, {y:.3f})")
        summary.append("")
        
        # Error metrics
        summary.append("ERROR METRICS:")
        summary.append(f"  Mean Absolute Error: {error_metrics['mean_absolute_error']:.6f}")
        summary.append(f"  Root Mean Square Error: {error_metrics['root_mean_square_error']:.6f}")
        summary.append(f"  Maximum Error: {error_metrics['max_error']:.6f}")
        summary.append(f"  Mean Error: {error_metrics['mean_error']:.6f}")
        summary.append(f"  Error Standard Deviation: {error_metrics['error_std']:.6f}")
        
        return "\n".join(summary) 