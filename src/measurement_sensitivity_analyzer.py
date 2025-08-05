"""
Measurement sensitivity analyzer module.

This module implements sensitivity analysis by systematically removing each distance
measurement and analyzing the impact on final coordinate positions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from .coordinate_estimator import CoordinateEstimator
from .distance_calculator import DistanceCalculator
import copy
import openpyxl


class MeasurementSensitivityAnalyzer:
    """
    Analyzes the sensitivity of coordinate estimation to individual distance measurements.
    
    This class implements sensitivity analysis by:
    1. Running baseline optimization with all measurements
    2. Systematically removing each measurement and re-optimizing
    3. Calculating coordinate changes and ranking measurement impact
    4. Providing detailed analysis of which measurements cause the most movement
    """
    
    def __init__(self, distances: Dict[Tuple[str, str], float], 
                 initial_coordinates: Dict[str, Tuple[float, float]],
                 fixed_points_info: Optional[Dict[str, str]] = None):
        """
        Initialize the sensitivity analyzer.
        
        Args:
            distances: Dictionary mapping (point_id1, point_id2) to measured distances
            initial_coordinates: Dictionary mapping point IDs to initial (x, y) coordinates
            fixed_points_info: Optional dictionary mapping point IDs to constraint types
        """
        self.original_distances = distances
        self.initial_coordinates = initial_coordinates
        self.fixed_points_info = fixed_points_info
        
        # Extract fixed points from the coordinate estimator
        self._extract_fixed_points()
        
        # Store baseline results
        self.baseline_result = None
        self.baseline_coordinates = None
        
        # Store sensitivity analysis results
        self.sensitivity_results = {}
        self.measurement_impact_ranking = None
    
    def _extract_fixed_points(self):
        """Extract fixed points information using the shared utility method."""
        self.fixed_points, self.y_axis_points = CoordinateEstimator.extract_fixed_points(
            self.initial_coordinates, 
            self.fixed_points_info
        )
    
    def run_baseline_optimization(self) -> Dict:
        """
        Run the baseline optimization with all measurements.
        
        Returns:
            Dictionary containing optimization results
        """
        print("Running baseline optimization with all measurements...")
        
        estimator = CoordinateEstimator(
            self.original_distances, 
            self.initial_coordinates, 
            self.fixed_points_info
        )
        
        self.baseline_result = estimator.optimize()
        self.baseline_coordinates = self.baseline_result['coordinates']
        
        print(f"Baseline optimization completed. Final RMS error: {self.baseline_result['error_metrics']['root_mean_square_error']:.6f}")
        
        return self.baseline_result
    
    def analyze_measurement_sensitivity(self, max_iterations: int = 1000, 
                                     tolerance: float = 1e-8) -> Dict:
        """
        Perform sensitivity analysis by removing each measurement and re-optimizing.
        
        Args:
            max_iterations: Maximum iterations for optimization
            tolerance: Convergence tolerance for optimization
            
        Returns:
            Dictionary containing sensitivity analysis results
        """
        if self.baseline_coordinates is None:
            raise ValueError("Must run baseline optimization first. Call run_baseline_optimization().")
        
        print(f"Starting sensitivity analysis for {len(self.original_distances)} measurements...")
        
        # Get all measurement pairs
        measurement_pairs = list(self.original_distances.keys())
        
        # Store results for each measurement
        self.sensitivity_results = {}
        
        for i, (point1, point2) in enumerate(measurement_pairs):
            print(f"Analyzing measurement {i+1}/{len(measurement_pairs)}: {point1}-{point2}")
            
            # Create reduced distances dictionary (remove one measurement)
            reduced_distances = copy.deepcopy(self.original_distances)
            del reduced_distances[(point1, point2)]
            
            # Run optimization with reduced dataset
            try:
                estimator = CoordinateEstimator(
                    reduced_distances, 
                    self.initial_coordinates, 
                    self.fixed_points_info
                )
                
                result = estimator.optimize(max_iterations=max_iterations, tolerance=tolerance)
                
                # Calculate coordinate changes
                coordinate_changes = self._calculate_coordinate_changes(
                    self.baseline_coordinates, 
                    result['coordinates']
                )
                
                # Find the point with maximum displacement
                max_displacement_point = None
                max_displacement_value = 0
                for point_id, point_data in coordinate_changes['point_displacements'].items():
                    if point_data['displacement'] > max_displacement_value:
                        max_displacement_value = point_data['displacement']
                        max_displacement_point = point_id
                
                # Store results
                self.sensitivity_results[(point1, point2)] = {
                    'removed_measurement': (point1, point2),
                    'original_distance': self.original_distances[(point1, point2)],
                    'optimization_result': result,
                    'coordinate_changes': coordinate_changes,
                    'rms_displacement': coordinate_changes['rms_displacement'],
                    'max_displacement': coordinate_changes['max_displacement'],
                    'total_displacement': coordinate_changes['total_displacement'],
                    'point_displacements': coordinate_changes['point_displacements'],
                    'max_displacement_point': max_displacement_point,
                    'max_displacement_value': max_displacement_value
                }
                
            except Exception as e:
                print(f"Warning: Optimization failed for measurement {point1}-{point2}: {e}")
                # Store failure result
                self.sensitivity_results[(point1, point2)] = {
                    'removed_measurement': (point1, point2),
                    'original_distance': self.original_distances[(point1, point2)],
                    'optimization_result': None,
                    'coordinate_changes': None,
                    'rms_displacement': float('inf'),
                    'max_displacement': float('inf'),
                    'total_displacement': float('inf'),
                    'point_displacements': {},
                    'error': str(e)
                }
        
        # Rank measurements by impact
        self._rank_measurements_by_impact()
        
        print("Sensitivity analysis completed.")
        return self.sensitivity_results
    
    def _calculate_coordinate_changes(self, baseline_coords: Dict[str, Tuple[float, float]], 
                                   new_coords: Dict[str, Tuple[float, float]]) -> Dict:
        """
        Calculate coordinate changes between baseline and new coordinates.
        
        Args:
            baseline_coords: Baseline coordinate dictionary
            new_coords: New coordinate dictionary
            
        Returns:
            Dictionary containing various displacement metrics
        """
        point_displacements = {}
        displacements = []
        
        for point_id in baseline_coords.keys():
            if point_id in new_coords:
                # Calculate displacement for this point
                x1, y1 = baseline_coords[point_id]
                x2, y2 = new_coords[point_id]
                
                displacement = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                x_change = x2 - x1
                y_change = y2 - y1
                
                # Calculate movement direction
                if displacement > 1e-10:
                    angle_rad = np.arctan2(y_change, x_change)
                    angle_deg = np.degrees(angle_rad)
                    
                    # Determine cardinal direction
                    if -22.5 <= angle_deg <= 22.5:
                        direction = "East"
                    elif 22.5 < angle_deg <= 67.5:
                        direction = "Northeast"
                    elif 67.5 < angle_deg <= 112.5:
                        direction = "North"
                    elif 112.5 < angle_deg <= 157.5:
                        direction = "Northwest"
                    elif 157.5 < angle_deg <= 180 or -180 <= angle_deg <= -157.5:
                        direction = "West"
                    elif -157.5 < angle_deg <= -112.5:
                        direction = "Southwest"
                    elif -112.5 < angle_deg <= -67.5:
                        direction = "South"
                    elif -67.5 < angle_deg <= -22.5:
                        direction = "Southeast"
                else:
                    direction = "No movement"
                    angle_deg = 0
                
                point_displacements[point_id] = {
                    'displacement': displacement,
                    'x_change': x_change,
                    'y_change': y_change,
                    'direction': direction,
                    'angle_degrees': angle_deg,
                    'baseline_coords': (x1, y1),
                    'new_coords': (x2, y2)
                }
                displacements.append(displacement)
            else:
                # Point not in new coordinates (shouldn't happen in normal case)
                point_displacements[point_id] = {
                    'displacement': float('inf'),
                    'x_change': float('inf'),
                    'y_change': float('inf'),
                    'direction': "No movement",
                    'angle_degrees': 0,
                    'baseline_coords': baseline_coords[point_id],
                    'new_coords': None
                }
                displacements.append(float('inf'))
        
        # Calculate aggregate metrics
        valid_displacements = [d for d in displacements if d != float('inf')]
        
        if valid_displacements:
            rms_displacement = np.sqrt(np.mean(np.array(valid_displacements)**2))
            max_displacement = max(valid_displacements)
            total_displacement = sum(valid_displacements)
        else:
            rms_displacement = float('inf')
            max_displacement = float('inf')
            total_displacement = float('inf')
        
        return {
            'rms_displacement': rms_displacement,
            'max_displacement': max_displacement,
            'total_displacement': total_displacement,
            'point_displacements': point_displacements,
            'displacement_list': displacements
        }
    
    def _rank_measurements_by_impact(self):
        """
        Rank measurements by their impact on coordinate changes.
        """
        if not self.sensitivity_results:
            raise ValueError("No sensitivity results available. Run analyze_measurement_sensitivity() first.")
        
        # Create ranking based on RMS displacement
        ranking_data = []
        
        for measurement_pair, result in self.sensitivity_results.items():
            if result['rms_displacement'] != float('inf'):
                ranking_data.append({
                    'measurement_pair': measurement_pair,
                    'point1': measurement_pair[0],
                    'point2': measurement_pair[1],
                    'original_distance': result['original_distance'],
                    'rms_displacement': result['rms_displacement'],
                    'max_displacement': result['max_displacement'],
                    'total_displacement': result['total_displacement'],
                    'max_displacement_point': result.get('max_displacement_point', None),
                    'max_displacement_value': result.get('max_displacement_value', 0)
                })
        
        # Sort by RMS displacement (highest impact first)
        ranking_data.sort(key=lambda x: x['rms_displacement'], reverse=True)
        
        self.measurement_impact_ranking = ranking_data
    
    def get_measurement_ranking(self) -> List[Dict]:
        """
        Get the ranking of measurements by their impact.
        
        Returns:
            List of dictionaries containing measurement impact data, sorted by impact
        """
        if self.measurement_impact_ranking is None:
            raise ValueError("No ranking available. Run analyze_measurement_sensitivity() first.")
        
        return self.measurement_impact_ranking
    
    def get_top_impactful_measurements(self, top_n: int = 10) -> List[Dict]:
        """
        Get the top N most impactful measurements.
        
        Args:
            top_n: Number of top measurements to return
            
        Returns:
            List of top N measurement impact dictionaries
        """
        ranking = self.get_measurement_ranking()
        return ranking[:top_n]
    
    def get_measurement_impact_summary(self) -> Dict:
        """
        Get a summary of measurement impact analysis.
        
        Returns:
            Dictionary containing summary statistics
        """
        if self.measurement_impact_ranking is None:
            raise ValueError("No ranking available. Run analyze_measurement_sensitivity() first.")
        
        if not self.measurement_impact_ranking:
            return {
                'total_measurements': 0,
                'successful_analyses': 0,
                'failed_analyses': 0,
                'max_rms_displacement': 0,
                'min_rms_displacement': 0,
                'mean_rms_displacement': 0,
                'median_rms_displacement': 0
            }
        
        rms_displacements = [item['rms_displacement'] for item in self.measurement_impact_ranking]
        
        return {
            'total_measurements': len(self.original_distances),
            'successful_analyses': len(self.measurement_impact_ranking),
            'failed_analyses': len(self.sensitivity_results) - len(self.measurement_impact_ranking),
            'max_rms_displacement': max(rms_displacements),
            'min_rms_displacement': min(rms_displacements),
            'mean_rms_displacement': np.mean(rms_displacements),
            'median_rms_displacement': np.median(rms_displacements)
        }
    
    def print_impact_summary(self, top_n: int = 10):
        """
        Print a summary of the most impactful measurements.
        
        Args:
            top_n: Number of top measurements to display
        """
        if self.measurement_impact_ranking is None:
            print("No sensitivity analysis results available.")
            return
        
        summary = self.get_measurement_impact_summary()
        
        print("\n" + "="*60)
        print("MEASUREMENT SENSITIVITY ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total measurements analyzed: {summary['total_measurements']}")
        print(f"Successful analyses: {summary['successful_analyses']}")
        print(f"Failed analyses: {summary['failed_analyses']}")
        print(f"RMS displacement range: {summary['min_rms_displacement']:.6f} - {summary['max_rms_displacement']:.6f}")
        print(f"Mean RMS displacement: {summary['mean_rms_displacement']:.6f}")
        print(f"Median RMS displacement: {summary['median_rms_displacement']:.6f}")
        
        print(f"\nTop {top_n} Most Impactful Measurements:")
        print("-" * 70)
        print(f"{'Rank':<4} {'Measurement':<15} {'Distance':<10} {'RMS Disp':<10} {'Max Disp':<10} {'Max Point':<10}")
        print("-" * 70)
        
        top_measurements = self.get_top_impactful_measurements(top_n)
        for i, measurement in enumerate(top_measurements, 1):
            pair = measurement['measurement_pair']
            measurement_str = f"{pair[0]}-{pair[1]}"
            max_point = measurement.get('max_displacement_point', 'N/A')
            print(f"{i:<4} {measurement_str:<15} {measurement['original_distance']:<10.3f} "
                  f"{measurement['rms_displacement']:<10.6f} {measurement['max_displacement']:<10.6f} {max_point:<10}")
    
    def export_sensitivity_results(self, filename: str = "measurement_sensitivity_analysis.xlsx"):
        """
        Export sensitivity analysis results to Excel file.
        
        Args:
            filename: Output filename
        """
        if self.measurement_impact_ranking is None:
            raise ValueError("No sensitivity analysis results available.")
        
        # Key points to track
        key_points = ['N', 'M4', 'O', 'X0', 'M3']
        
        # Create detailed results DataFrame
        detailed_data = []
        
        for measurement_pair, result in self.sensitivity_results.items():
            if result['rms_displacement'] != float('inf'):
                # Basic measurement data
                row_data = {
                    'Point1': measurement_pair[0],
                    'Point2': measurement_pair[1],
                    'Original_Distance': result['original_distance'],
                    'RMS_Displacement': result['rms_displacement'],
                    'Max_Displacement': result['max_displacement'],
                    'Total_Displacement': result['total_displacement'],
                    'Max_Displacement_Point': result.get('max_displacement_point', None),
                    'Max_Displacement_Value': result.get('max_displacement_value', 0),
                    'Optimization_Success': True,
                    'Error': None
                }
                
                # Add point-specific data for key points
                if result['point_displacements']:
                    for point_id in key_points:
                        if point_id in result['point_displacements']:
                            point_data = result['point_displacements'][point_id]
                            row_data[f'{point_id}_Movement'] = point_data['displacement']
                            row_data[f'{point_id}_X_Change'] = point_data['x_change']
                            row_data[f'{point_id}_Y_Change'] = point_data['y_change']
                            row_data[f'{point_id}_Direction'] = point_data['direction']
                        else:
                            row_data[f'{point_id}_Movement'] = 0
                            row_data[f'{point_id}_X_Change'] = 0
                            row_data[f'{point_id}_Y_Change'] = 0
                            row_data[f'{point_id}_Direction'] = "No movement"
                
                detailed_data.append(row_data)
            else:
                # Failed optimization
                row_data = {
                    'Point1': measurement_pair[0],
                    'Point2': measurement_pair[1],
                    'Original_Distance': result['original_distance'],
                    'RMS_Displacement': None,
                    'Max_Displacement': None,
                    'Total_Displacement': None,
                    'Optimization_Success': False,
                    'Error': result.get('error', 'Unknown error')
                }
                
                # Add empty point-specific data for key points
                for point_id in key_points:
                    row_data[f'{point_id}_Movement'] = None
                    row_data[f'{point_id}_X_Change'] = None
                    row_data[f'{point_id}_Y_Change'] = None
                    row_data[f'{point_id}_Direction'] = None
                
                detailed_data.append(row_data)
        
        detailed_df = pd.DataFrame(detailed_data)
        
        # Sort detailed results by RMS displacement (descending)
        detailed_df = detailed_df.sort_values('RMS_Displacement', ascending=False, na_position='last')
        
        # Create ranking DataFrame
        ranking_data = []
        for i, measurement in enumerate(self.measurement_impact_ranking, 1):
            ranking_data.append({
                'Rank': i,
                'Point1': measurement['point1'],
                'Point2': measurement['point2'],
                'Original_Distance': measurement['original_distance'],
                'RMS_Displacement': measurement['rms_displacement'],
                'Max_Displacement': measurement['max_displacement'],
                'Total_Displacement': measurement['total_displacement'],
                'Max_Displacement_Point': measurement.get('max_displacement_point', None),
                'Max_Displacement_Value': measurement.get('max_displacement_value', 0)
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        
        # Create summary DataFrame
        summary = self.get_measurement_impact_summary()
        summary_data = [{
            'Metric': key.replace('_', ' ').title(),
            'Value': value
        } for key, value in summary.items()]
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to Excel
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            detailed_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
            ranking_df.to_excel(writer, sheet_name='Impact_Ranking', index=False)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Apply formatting to all sheets
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                self._format_excel_sheet(worksheet, sheet_name)
        
        print(f"Sensitivity analysis results exported to: {filename}")
    
    def _format_excel_sheet(self, worksheet, sheet_name):
        """
        Format an Excel worksheet with word wrapping for the first row and numeric formatting.
        
        Args:
            worksheet: The openpyxl worksheet object
            sheet_name: Name of the sheet for reference
        """
        from openpyxl.styles import Alignment
        
        # Format the first row (headers) with word wrapping
        for cell in worksheet[1]:
            cell.alignment = Alignment(wrap_text=True, vertical='top')
        
        # Set all column widths to 10 units
        for column in worksheet.columns:
            column_letter = column[0].column_letter
            worksheet.column_dimensions[column_letter].width = 10
        
        # Format numeric data to 3 decimal places
        for row in worksheet.iter_rows(min_row=2):  # Skip header row
            for cell in row:
                if cell.value is not None:
                    # Check if the cell contains a numeric value
                    try:
                        float_val = float(cell.value)
                        # Format to 3 decimal places
                        cell.number_format = '0.000'
                    except (ValueError, TypeError):
                        # Not a numeric value, leave as is
                        pass 

    def compute_jacobian_sensitivity(self) -> Dict:
        """
        Compute Jacobian-based sensitivity analysis.
        
        This method computes the Jacobian matrix ∂coordinates/∂distances at the optimal solution
        to provide local sensitivity information for each distance measurement.
        
        Returns:
            Dictionary containing Jacobian sensitivity analysis results
        """
        if self.baseline_coordinates is None:
            raise ValueError("Must run baseline optimization first. Call run_baseline_optimization().")
        
        print("Computing Jacobian-based sensitivity analysis...")
        
        # Get optimal coordinates and distances
        optimal_coords = self.baseline_coordinates
        distance_pairs = list(self.original_distances.keys())
        
        # Compute Jacobian matrix: ∂coordinates/∂distances
        jacobian = self._compute_jacobian(optimal_coords, distance_pairs)
        
        # Analyze sensitivity for each distance measurement
        jacobian_sensitivities = []
        for i, (pair, distance) in enumerate(self.original_distances.items()):
            # Compute sensitivity: ||∂coordinates/∂distance_i||
            sensitivity = np.linalg.norm(jacobian[:, i])
            
            # Compute coordinate-specific sensitivities
            coord_sensitivities = {}
            for j, point_id in enumerate(self._get_all_point_ids()):
                if point_id not in self.fixed_points:
                    # Get indices for this point's coordinates in the optimization vector
                    if point_id in self.y_axis_points:
                        # Only y-coordinate for y_axis points
                        coord_sensitivities[f"{point_id}_y"] = abs(jacobian[j*2+1, i]) if j*2+1 < jacobian.shape[0] else 0.0
                    else:
                        # Both x and y coordinates for regular points
                        coord_sensitivities[f"{point_id}_x"] = abs(jacobian[j*2, i]) if j*2 < jacobian.shape[0] else 0.0
                        coord_sensitivities[f"{point_id}_y"] = abs(jacobian[j*2+1, i]) if j*2+1 < jacobian.shape[0] else 0.0
            
            # Find most sensitive coordinate
            max_sensitive_coord = max(coord_sensitivities.items(), key=lambda x: x[1])
            
            jacobian_sensitivities.append({
                'measurement_pair': pair,
                'distance': distance,
                'sensitivity': sensitivity,
                'coordinate_sensitivities': coord_sensitivities,
                'most_sensitive_coordinate': max_sensitive_coord[0],
                'max_coordinate_sensitivity': max_sensitive_coord[1]
            })
        
        # Sort by sensitivity (descending)
        jacobian_sensitivities.sort(key=lambda x: x['sensitivity'], reverse=True)
        
        return {
            'jacobian_matrix': jacobian,
            'sensitivities': jacobian_sensitivities,
            'distance_pairs': distance_pairs
        }
    
    def _compute_jacobian(self, coordinates: Dict[str, Tuple[float, float]], 
                         distance_pairs: List[Tuple[str, str]]) -> np.ndarray:
        """
        Compute the Jacobian matrix ∂coordinates/∂distances.
        
        Args:
            coordinates: Optimal coordinate dictionary
            distance_pairs: List of distance measurement pairs
            
        Returns:
            Jacobian matrix with shape (n_coordinates, n_distances)
        """
        # Get free point IDs (excluding fixed points)
        free_points = [pid for pid in self._get_all_point_ids() if pid not in self.fixed_points]
        n_free_coords = sum(2 if pid not in self.y_axis_points else 1 for pid in free_points)
        n_distances = len(distance_pairs)
        
        # Initialize Jacobian matrix
        jacobian = np.zeros((n_free_coords, n_distances))
        
        # Compute Jacobian using finite differences
        for i, (pair, original_distance) in enumerate(self.original_distances.items()):
            # Use a perturbation that's proportional to the distance value
            # This ensures the perturbation is meaningful relative to the scale of the problem
            epsilon = max(original_distance * 0.01, 0.1)  # 1% of distance or minimum 0.1
            
            # Perturb the distance
            perturbed_distances = self.original_distances.copy()
            perturbed_distances[pair] = original_distance + epsilon
            
            # Re-optimize with perturbed distance
            try:
                estimator = CoordinateEstimator(
                    perturbed_distances, 
                    self.initial_coordinates, 
                    self.fixed_points_info
                )
                perturbed_result = estimator.optimize(max_iterations=200, tolerance=1e-8)
                perturbed_coords = perturbed_result['coordinates']
                
                # Compute finite difference
                row_idx = 0
                for point_id in free_points:
                    if point_id in self.y_axis_points:
                        # Only y-coordinate for y_axis points
                        original_y = coordinates[point_id][1]
                        perturbed_y = perturbed_coords[point_id][1]
                        jacobian[row_idx, i] = (perturbed_y - original_y) / epsilon
                        row_idx += 1
                    else:
                        # Both x and y coordinates for regular points
                        original_x, original_y = coordinates[point_id]
                        perturbed_x, perturbed_y = perturbed_coords[point_id]
                        jacobian[row_idx, i] = (perturbed_x - original_x) / epsilon
                        jacobian[row_idx + 1, i] = (perturbed_y - original_y) / epsilon
                        row_idx += 2
                        
            except Exception as e:
                print(f"Warning: Could not compute Jacobian for {pair}: {str(e)}")
                # Set column to zero if optimization fails
                jacobian[:, i] = 0.0
        
        return jacobian
    
    def _get_all_point_ids(self) -> List[str]:
        """Get all point IDs from the data."""
        point_ids = set()
        
        # Add points from distance measurements
        for pair in self.original_distances.keys():
            point_ids.add(pair[0])
            point_ids.add(pair[1])
        
        # Add points from initial coordinates
        point_ids.update(self.initial_coordinates.keys())
        
        return sorted(list(point_ids)) 

    def print_jacobian_summary(self, jacobian_results: Dict, top_n: int = 15):
        """
        Print a summary of the Jacobian sensitivity analysis.
        
        Args:
            jacobian_results: Results from compute_jacobian_sensitivity()
            top_n: Number of top measurements to display
        """
        sensitivities = jacobian_results['sensitivities']
        
        print("\n" + "="*70)
        print("JACOBIAN SENSITIVITY ANALYSIS SUMMARY")
        print("="*70)
        print(f"Total measurements analyzed: {len(sensitivities)}")
        
        if sensitivities:
            max_sensitivity = max(s['sensitivity'] for s in sensitivities)
            min_sensitivity = min(s['sensitivity'] for s in sensitivities)
            mean_sensitivity = np.mean([s['sensitivity'] for s in sensitivities])
            
            print(f"Sensitivity range: {min_sensitivity:.6f} - {max_sensitivity:.6f}")
            print(f"Mean sensitivity: {mean_sensitivity:.6f}")
        
        print(f"\nTop {top_n} Most Sensitive Measurements (Jacobian):")
        print("-" * 80)
        print(f"{'Rank':<4} {'Measurement':<15} {'Distance':<10} {'Sensitivity':<12} {'Max Coord':<15} {'Coord Sens':<12}")
        print("-" * 80)
        
        for i, measurement in enumerate(sensitivities[:top_n], 1):
            pair = measurement['measurement_pair']
            measurement_str = f"{pair[0]}-{pair[1]}"
            distance = measurement['distance']
            sensitivity = measurement['sensitivity']
            max_coord = measurement['most_sensitive_coordinate']
            coord_sens = measurement['max_coordinate_sensitivity']
            
            print(f"{i:<4} {measurement_str:<15} {distance:<10.3f} {sensitivity:<12.6f} {max_coord:<15} {coord_sens:<12.6f}")
    
    def export_jacobian_results(self, jacobian_results: Dict, filename: str = "jacobian_sensitivity_analysis.xlsx"):
        """
        Export Jacobian sensitivity analysis results to Excel file.
        
        Args:
            jacobian_results: Results from compute_jacobian_sensitivity()
            filename: Output filename
        """
        sensitivities = jacobian_results['sensitivities']
        
        # Create detailed results DataFrame
        detailed_data = []
        
        for measurement in sensitivities:
            detailed_data.append({
                'Point1': measurement['measurement_pair'][0],
                'Point2': measurement['measurement_pair'][1],
                'Distance': measurement['distance'],
                'Sensitivity': measurement['sensitivity'],
                'Most_Sensitive_Coordinate': measurement['most_sensitive_coordinate'],
                'Max_Coordinate_Sensitivity': measurement['max_coordinate_sensitivity']
            })
        
        detailed_df = pd.DataFrame(detailed_data)
        
        # Sort by sensitivity (descending)
        detailed_df = detailed_df.sort_values('Sensitivity', ascending=False)
        
        # Create ranking DataFrame
        ranking_data = []
        for i, measurement in enumerate(sensitivities, 1):
            ranking_data.append({
                'Rank': i,
                'Point1': measurement['measurement_pair'][0],
                'Point2': measurement['measurement_pair'][1],
                'Distance': measurement['distance'],
                'Sensitivity': measurement['sensitivity'],
                'Most_Sensitive_Coordinate': measurement['most_sensitive_coordinate'],
                'Max_Coordinate_Sensitivity': measurement['max_coordinate_sensitivity']
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        
        # Create coordinate sensitivity DataFrame
        coord_data = []
        for measurement in sensitivities:
            for coord, sens in measurement['coordinate_sensitivities'].items():
                coord_data.append({
                    'Point1': measurement['measurement_pair'][0],
                    'Point2': measurement['measurement_pair'][1],
                    'Distance': measurement['distance'],
                    'Coordinate': coord,
                    'Sensitivity': sens
                })
        
        coord_df = pd.DataFrame(coord_data)
        
        # Save to Excel
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            detailed_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
            ranking_df.to_excel(writer, sheet_name='Sensitivity_Ranking', index=False)
            coord_df.to_excel(writer, sheet_name='Coordinate_Sensitivities', index=False)
            
            # Apply formatting to all sheets
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                self._format_excel_sheet(worksheet, sheet_name)
        
        print(f"Jacobian sensitivity analysis exported to: {filename}")
    
    def run_complete_sensitivity_analysis(self, max_iterations: int = 1000, 
                                        tolerance: float = 1e-8) -> Dict:
        """
        Run both leave-one-out and Jacobian sensitivity analyses.
        
        Args:
            max_iterations: Maximum iterations for optimization
            tolerance: Convergence tolerance for optimization
            
        Returns:
            Dictionary containing both types of sensitivity analysis results
        """
        print("Running complete sensitivity analysis (leave-one-out + Jacobian)...")
        
        # Run baseline optimization
        baseline_result = self.run_baseline_optimization()
        
        # Run leave-one-out analysis
        print("\nRunning leave-one-out sensitivity analysis...")
        leave_one_out_results = self.analyze_measurement_sensitivity(max_iterations, tolerance)
        
        # Run Jacobian analysis
        print("\nRunning Jacobian sensitivity analysis...")
        jacobian_results = self.compute_jacobian_sensitivity()
        
        # Print summaries
        self.print_impact_summary(top_n=15)
        self.print_jacobian_summary(jacobian_results, top_n=15)
        
        return {
            'baseline_result': baseline_result,
            'leave_one_out_results': leave_one_out_results,
            'jacobian_results': jacobian_results
        } 

    def analyze_point_specific_impact(self, target_points: List[str] = None) -> Dict:
        """
        Analyze how each measurement affects specific points.
        
        This method tracks the impact of removing each measurement on the movement
        of specific points, including both magnitude and direction of movement.
        
        Args:
            target_points: List of point IDs to analyze. If None, analyzes all points.
            
        Returns:
            Dictionary containing point-specific impact analysis
        """
        if self.baseline_coordinates is None:
            raise ValueError("Must run baseline optimization first. Call run_baseline_optimization().")
        
        if target_points is None:
            target_points = list(self.baseline_coordinates.keys())
        
        print(f"Analyzing point-specific impact for {len(target_points)} points...")
        
        # Store results for each measurement and point
        point_impact_results = {}
        
        # Get all measurement pairs
        measurement_pairs = list(self.original_distances.keys())
        
        for i, (point1, point2) in enumerate(measurement_pairs):
            print(f"Analyzing measurement {i+1}/{len(measurement_pairs)}: {point1}-{point2}")
            
            # Create reduced distances dictionary (remove one measurement)
            reduced_distances = copy.deepcopy(self.original_distances)
            del reduced_distances[(point1, point2)]
            
            # Run optimization with reduced dataset
            try:
                estimator = CoordinateEstimator(
                    reduced_distances, 
                    self.initial_coordinates, 
                    self.fixed_points_info
                )
                
                result = estimator.optimize(max_iterations=1000, tolerance=1e-8)
                
                # Analyze impact on each target point
                point_impacts = {}
                for point_id in target_points:
                    if point_id in self.baseline_coordinates and point_id in result['coordinates']:
                        baseline_x, baseline_y = self.baseline_coordinates[point_id]
                        new_x, new_y = result['coordinates'][point_id]
                        
                        # Calculate movement
                        x_change = new_x - baseline_x
                        y_change = new_y - baseline_y
                        total_movement = np.sqrt(x_change**2 + y_change**2)
                        
                        # Determine movement direction
                        if total_movement > 1e-10:  # Only if there's significant movement
                            # Calculate angle in degrees
                            angle_rad = np.arctan2(y_change, x_change)
                            angle_deg = np.degrees(angle_rad)
                            
                            # Determine cardinal direction
                            if -22.5 <= angle_deg <= 22.5:
                                direction = "East"
                            elif 22.5 < angle_deg <= 67.5:
                                direction = "Northeast"
                            elif 67.5 < angle_deg <= 112.5:
                                direction = "North"
                            elif 112.5 < angle_deg <= 157.5:
                                direction = "Northwest"
                            elif 157.5 < angle_deg <= 180 or -180 <= angle_deg <= -157.5:
                                direction = "West"
                            elif -157.5 < angle_deg <= -112.5:
                                direction = "Southwest"
                            elif -112.5 < angle_deg <= -67.5:
                                direction = "South"
                            elif -67.5 < angle_deg <= -22.5:
                                direction = "Southeast"
                        else:
                            direction = "No movement"
                            angle_deg = 0
                        
                        point_impacts[point_id] = {
                            'baseline_coords': (baseline_x, baseline_y),
                            'new_coords': (new_x, new_y),
                            'x_change': x_change,
                            'y_change': y_change,
                            'total_movement': total_movement,
                            'direction': direction,
                            'angle_degrees': angle_deg,
                            'movement_magnitude': total_movement
                        }
                    else:
                        point_impacts[point_id] = {
                            'baseline_coords': self.baseline_coordinates.get(point_id, (0, 0)),
                            'new_coords': result['coordinates'].get(point_id, (0, 0)),
                            'x_change': 0,
                            'y_change': 0,
                            'total_movement': 0,
                            'direction': "No movement",
                            'angle_degrees': 0,
                            'movement_magnitude': 0
                        }
                
                point_impact_results[(point1, point2)] = {
                    'removed_measurement': (point1, point2),
                    'original_distance': self.original_distances[(point1, point2)],
                    'point_impacts': point_impacts,
                    'optimization_success': True
                }
                
            except Exception as e:
                print(f"Warning: Optimization failed for measurement {point1}-{point2}: {e}")
                # Store failure result
                point_impact_results[(point1, point2)] = {
                    'removed_measurement': (point1, point2),
                    'original_distance': self.original_distances[(point1, point2)],
                    'point_impacts': {},
                    'optimization_success': False,
                    'error': str(e)
                }
        
        return point_impact_results
    
    def get_top_measurements_for_point(self, point_id: str, point_impact_results: Dict, top_n: int = 10) -> List[Dict]:
        """
        Get the top measurements that most affect a specific point.
        
        Args:
            point_id: The point ID to analyze
            point_impact_results: Results from analyze_point_specific_impact()
            top_n: Number of top measurements to return
            
        Returns:
            List of measurement impact dictionaries, sorted by impact on the point
        """
        point_measurements = []
        
        for measurement_pair, result in point_impact_results.items():
            if result['optimization_success'] and point_id in result['point_impacts']:
                impact = result['point_impacts'][point_id]
                point_measurements.append({
                    'measurement_pair': measurement_pair,
                    'point1': measurement_pair[0],
                    'point2': measurement_pair[1],
                    'original_distance': result['original_distance'],
                    'movement_magnitude': impact['movement_magnitude'],
                    'x_change': impact['x_change'],
                    'y_change': impact['y_change'],
                    'direction': impact['direction'],
                    'angle_degrees': impact['angle_degrees'],
                    'baseline_coords': impact['baseline_coords'],
                    'new_coords': impact['new_coords']
                })
        
        # Sort by movement magnitude (descending)
        point_measurements.sort(key=lambda x: x['movement_magnitude'], reverse=True)
        
        return point_measurements[:top_n]
    
    def print_point_impact_summary(self, point_id: str, point_impact_results: Dict, top_n: int = 10):
        """
        Print a summary of how measurements affect a specific point.
        
        Args:
            point_id: The point ID to analyze
            point_impact_results: Results from analyze_point_specific_impact()
            top_n: Number of top measurements to display
        """
        top_measurements = self.get_top_measurements_for_point(point_id, point_impact_results, top_n)
        
        if not top_measurements:
            print(f"No impact data available for point {point_id}")
            return
        
        print(f"\n{'='*70}")
        print(f"POINT-SPECIFIC IMPACT ANALYSIS FOR POINT {point_id}")
        print(f"{'='*70}")
        print(f"Top {top_n} measurements affecting point {point_id}:")
        print("-" * 80)
        print(f"{'Rank':<4} {'Measurement':<15} {'Distance':<10} {'Movement':<10} {'Direction':<12} {'X Change':<10} {'Y Change':<10}")
        print("-" * 80)
        
        for i, measurement in enumerate(top_measurements, 1):
            pair = measurement['measurement_pair']
            measurement_str = f"{pair[0]}-{pair[1]}"
            distance = measurement['original_distance']
            movement = measurement['movement_magnitude']
            direction = measurement['direction']
            x_change = measurement['x_change']
            y_change = measurement['y_change']
            
            print(f"{i:<4} {measurement_str:<15} {distance:<10.3f} {movement:<10.3f} {direction:<12} {x_change:<10.3f} {y_change:<10.3f}")
        
        # Summary statistics
        movements = [m['movement_magnitude'] for m in top_measurements]
        x_changes = [m['x_change'] for m in top_measurements]
        y_changes = [m['y_change'] for m in top_measurements]
        
        print(f"\nSummary for point {point_id}:")
        print(f"  Average movement: {np.mean(movements):.3f}")
        print(f"  Maximum movement: {max(movements):.3f}")
        print(f"  Average X change: {np.mean(x_changes):.3f}")
        print(f"  Average Y change: {np.mean(y_changes):.3f}")
        
        # Direction analysis
        directions = [m['direction'] for m in top_measurements]
        direction_counts = {}
        for direction in directions:
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        
        print(f"  Most common movement direction: {max(direction_counts.items(), key=lambda x: x[1])[0]}")
    
    def export_point_impact_results(self, point_impact_results: Dict, filename: str = "point_specific_impact_analysis.xlsx"):
        """
        Export point-specific impact analysis results to Excel file.
        
        Args:
            point_impact_results: Results from analyze_point_specific_impact()
            filename: Output filename
        """
        # Create detailed results DataFrame
        detailed_data = []
        
        for measurement_pair, result in point_impact_results.items():
            if result['optimization_success']:
                for point_id, impact in result['point_impacts'].items():
                    detailed_data.append({
                        'Point1': measurement_pair[0],
                        'Point2': measurement_pair[1],
                        'Target_Point': point_id,
                        'Original_Distance': result['original_distance'],
                        'Movement_Magnitude': impact['movement_magnitude'],
                        'X_Change': impact['x_change'],
                        'Y_Change': impact['y_change'],
                        'Direction': impact['direction'],
                        'Angle_Degrees': impact['angle_degrees'],
                        'Baseline_X': impact['baseline_coords'][0],
                        'Baseline_Y': impact['baseline_coords'][1],
                        'New_X': impact['new_coords'][0],
                        'New_Y': impact['new_coords'][1]
                    })
        
        detailed_df = pd.DataFrame(detailed_data)
        
        # Sort by movement magnitude (descending)
        detailed_df = detailed_df.sort_values('Movement_Magnitude', ascending=False)
        
        # Create summary by point
        point_summary_data = []
        for point_id in set(detailed_df['Target_Point']):
            point_data = detailed_df[detailed_df['Target_Point'] == point_id]
            if not point_data.empty:
                point_summary_data.append({
                    'Point_ID': point_id,
                    'Total_Measurements_Affecting': len(point_data),
                    'Average_Movement': point_data['Movement_Magnitude'].mean(),
                    'Max_Movement': point_data['Movement_Magnitude'].max(),
                    'Average_X_Change': point_data['X_Change'].mean(),
                    'Average_Y_Change': point_data['Y_Change'].mean(),
                    'Most_Common_Direction': point_data['Direction'].mode().iloc[0] if len(point_data['Direction'].mode()) > 0 else 'Unknown'
                })
        
        point_summary_df = pd.DataFrame(point_summary_data)
        point_summary_df = point_summary_df.sort_values('Average_Movement', ascending=False)
        
        # Save to Excel
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            detailed_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
            point_summary_df.to_excel(writer, sheet_name='Point_Summary', index=False)
            
            # Apply formatting to all sheets
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                self._format_excel_sheet(worksheet, sheet_name)
        
        print(f"Point-specific impact analysis exported to: {filename}") 