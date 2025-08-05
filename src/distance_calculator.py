"""
Distance calculator module for computing Euclidean distances and error metrics.

This module handles:
- Computing Euclidean distances between points
- Calculating error metrics between measured and calculated distances
- Validating distance measurements
- Providing statistical analysis of errors
"""

import numpy as np
from typing import Dict, Tuple, List
import math


class DistanceCalculator:
    """
    Calculates Euclidean distances and error metrics for coordinate estimation.
    
    This class provides methods to:
    - Compute distances between points given their coordinates
    - Calculate errors between measured and calculated distances
    - Provide statistical analysis of distance errors
    """
    
    def __init__(self):
        """Initialize the distance calculator."""
        pass
    
    def euclidean_distance(self, point1: Tuple[float, float], 
                          point2: Tuple[float, float]) -> float:
        """
        Calculate the Euclidean distance between two points.
        
        Args:
            point1: Tuple of (x, y) coordinates for first point
            point2: Tuple of (x, y) coordinates for second point
            
        Returns:
            Euclidean distance between the points
            
        Raises:
            ValueError: If coordinates are not valid numbers
        """
        try:
            x1, y1 = point1
            x2, y2 = point2
            
            # Validate that coordinates are numbers
            if not all(isinstance(coord, (int, float)) for coord in [x1, y1, x2, y2]):
                raise ValueError("All coordinates must be numeric values")
            
            # Calculate Euclidean distance using the distance formula
            # distance = sqrt((x2 - x1)^2 + (y2 - y1)^2)
            dx = x2 - x1
            dy = y2 - y1
            distance = math.sqrt(dx * dx + dy * dy)
            
            return distance
            
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid coordinates: {point1}, {point2}. Error: {str(e)}")
    
    def calculate_distances(self, coordinates: Dict[str, Tuple[float, float]]) -> Dict[Tuple[str, str], float]:
        """
        Calculate all pairwise distances between points.
        
        Args:
            coordinates: Dictionary mapping point IDs to (x, y) coordinates
            
        Returns:
            Dictionary mapping (point_id1, point_id2) tuples to calculated distances
        """
        distances = {}
        point_ids = list(coordinates.keys())
        
        # Calculate distances for all pairs
        for i in range(len(point_ids)):
            for j in range(i + 1, len(point_ids)):
                point1_id = point_ids[i]
                point2_id = point_ids[j]
                
                # Get coordinates for both points
                point1_coords = coordinates[point1_id]
                point2_coords = coordinates[point2_id]
                
                # Calculate distance
                distance = self.euclidean_distance(point1_coords, point2_coords)
                
                # Store with consistent ordering (smaller ID first)
                if point1_id < point2_id:
                    key = (point1_id, point2_id)
                else:
                    key = (point2_id, point1_id)
                
                distances[key] = distance
        
        return distances
    
    def calculate_errors(self, coordinates: Dict[str, Tuple[float, float]], 
                        measured_distances: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
        """
        Calculate errors between measured and calculated distances.
        
        Args:
            coordinates: Dictionary mapping point IDs to (x, y) coordinates
            measured_distances: Dictionary mapping (point_id1, point_id2) to measured distances
            
        Returns:
            Dictionary mapping (point_id1, point_id2) to error values
        """
        # Calculate all pairwise distances for the given coordinates
        calculated_distances = self.calculate_distances(coordinates)
        
        # Calculate errors for measured distances only
        errors = {}
        for pair, measured_dist in measured_distances.items():
            if pair in calculated_distances:
                calculated_dist = calculated_distances[pair]
                error = measured_dist - calculated_dist
                errors[pair] = error
            else:
                # This shouldn't happen if data validation worked correctly
                raise ValueError(f"Distance measurement for {pair} not found in calculated distances")
        
        return errors
    
    def calculate_error_metrics(self, coordinates: Dict[str, Tuple[float, float]], 
                              measured_distances: Dict[Tuple[str, str], float]) -> Dict[str, float]:
        """
        Calculate comprehensive error metrics for the coordinate estimation.
        
        Args:
            coordinates: Dictionary mapping point IDs to (x, y) coordinates
            measured_distances: Dictionary mapping (point_id1, point_id2) to measured distances
            
        Returns:
            Dictionary containing various error metrics:
            - 'mean_absolute_error': Average absolute error
            - 'root_mean_square_error': RMS error
            - 'max_error': Maximum absolute error
            - 'mean_error': Average error (can be positive or negative)
            - 'error_std': Standard deviation of errors
        """
        # Calculate individual errors
        errors = self.calculate_errors(coordinates, measured_distances)
        error_values = list(errors.values())
        
        if not error_values:
            return {
                'mean_absolute_error': 0.0,
                'root_mean_square_error': 0.0,
                'max_error': 0.0,
                'mean_error': 0.0,
                'error_std': 0.0
            }
        
        # Convert to numpy array for efficient calculations
        errors_array = np.array(error_values)
        
        # Calculate various error metrics
        mean_absolute_error = np.mean(np.abs(errors_array))
        root_mean_square_error = np.sqrt(np.mean(errors_array ** 2))
        max_error = np.max(np.abs(errors_array))
        mean_error = np.mean(errors_array)
        error_std = np.std(errors_array)
        
        return {
            'mean_absolute_error': float(mean_absolute_error),
            'root_mean_square_error': float(root_mean_square_error),
            'max_error': float(max_error),
            'mean_error': float(mean_error),
            'error_std': float(error_std)
        }
    
    def validate_distances(self, distances: Dict[Tuple[str, str], float]) -> List[str]:
        """
        Validate distance measurements for common issues.
        
        Args:
            distances: Dictionary mapping (point_id1, point_id2) to distances
            
        Returns:
            List of validation warnings/errors
        """
        issues = []
        
        # Check for negative or zero distances
        for pair, distance in distances.items():
            if distance <= 0:
                issues.append(f"Invalid distance for {pair}: {distance} (must be positive)")
        
        # Check for triangle inequality violations
        # This is a basic check - more sophisticated checks could be added
        point_ids = set()
        for pair in distances.keys():
            point_ids.add(pair[0])
            point_ids.add(pair[1])
        
        point_ids = list(point_ids)
        
        # Check triangle inequality for all triangles
        for i in range(len(point_ids)):
            for j in range(i + 1, len(point_ids)):
                for k in range(j + 1, len(point_ids)):
                    point_a = point_ids[i]
                    point_b = point_ids[j]
                    point_c = point_ids[k]
                    
                    # Get the three sides of the triangle
                    sides = []
                    for pair in [(point_a, point_b), (point_b, point_c), (point_a, point_c)]:
                        ordered_pair = (min(pair), max(pair))
                        if ordered_pair in distances:
                            sides.append(distances[ordered_pair])
                    
                    # If we have all three sides, check triangle inequality
                    if len(sides) == 3:
                        a, b, c = sides
                        if a + b <= c or a + c <= b or b + c <= a:
                            issues.append(
                                f"Triangle inequality violated for triangle {point_a}-{point_b}-{point_c}: "
                                f"sides {a}, {b}, {c}"
                            )
        
        return issues
    
    def get_distance_statistics(self, distances: Dict[Tuple[str, str], float]) -> Dict[str, float]:
        """
        Calculate statistics for the distance measurements.
        
        Args:
            distances: Dictionary mapping (point_id1, point_id2) to distances
            
        Returns:
            Dictionary containing distance statistics:
            - 'min_distance': Minimum distance
            - 'max_distance': Maximum distance
            - 'mean_distance': Average distance
            - 'distance_std': Standard deviation of distances
            - 'total_distances': Number of distance measurements
        """
        if not distances:
            return {
                'min_distance': 0.0,
                'max_distance': 0.0,
                'mean_distance': 0.0,
                'distance_std': 0.0,
                'total_distances': 0
            }
        
        distance_values = list(distances.values())
        distances_array = np.array(distance_values)
        
        return {
            'min_distance': float(np.min(distances_array)),
            'max_distance': float(np.max(distances_array)),
            'mean_distance': float(np.mean(distances_array)),
            'distance_std': float(np.std(distances_array)),
            'total_distances': len(distances)
        } 