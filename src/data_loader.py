"""
Data loader module for reading Excel files with distance measurements and initial coordinates.

This module provides:
- Excel file reading with validation
- Data format checking
- Connectivity analysis
- Flexible fixed point designation
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Set, Optional
import os


class ExcelDataLoader:
    """
    Loads and validates data from Excel files for coordinate estimation.
    
    This class handles:
    - Reading distance measurements and initial coordinates
    - Validating data format and consistency
    - Analyzing point connectivity
    - Supporting flexible fixed point designation
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the data loader.
        
        Args:
            file_path: Path to the Excel file
        """
        self.file_path = file_path
        
        # Expected headers for distance measurements sheet
        self.expected_distance_headers = ["First point ID", "Second point ID", "Distance", "Status"]
        
        # Expected headers for coordinates sheet (with fixed point and line group columns)
        self.expected_coordinate_headers = ["Point ID", "x_guess", "y_guess", "Fixed Point", "Line Group"]
        
        # Data storage
        self.distances_df = None
        self.coordinates_df = None
        
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Excel file not found: {file_path}")
    
    def load_data(self) -> Tuple[Dict[Tuple[str, str], float], Dict[str, Tuple[float, float]]]:
        """
        Load and validate all data from the Excel file.
        
        Returns:
            Tuple of (distances_dict, coordinates_dict)
            
        Raises:
            ValueError: If data format is invalid
        """
        # Load both sheets
        self.distances_df = self._load_distances_sheet()
        self.coordinates_df = self._load_coordinates_sheet()
        
        # Validate consistency between sheets
        self._validate_data_consistency()
        
        # Convert to dictionary format
        distances = self._convert_distances_to_dict()
        coordinates = self._convert_coordinates_to_dict()
        
        return distances, coordinates
    
    def get_original_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get the original DataFrames for enhanced output analysis.
        
        Returns:
            Tuple of (distances_df, coordinates_df)
            
        Raises:
            ValueError: If data not loaded yet
        """
        if self.distances_df is None or self.coordinates_df is None:
            raise ValueError("Data not loaded yet. Call load_data() first.")
        
        return self.distances_df.copy(), self.coordinates_df.copy()
    
    def get_fixed_points_info(self) -> Dict[str, str]:
        """
        Get information about which points are fixed and how.
        
        Returns:
            Dictionary mapping point IDs to their constraint type:
            - "origin": Fixed at (0,0)
            - "y_axis": Fixed at (0,y) where y > 0
            - "free": Free to optimize
        """
        if self.coordinates_df is None:
            raise ValueError("Data not loaded yet. Call load_data() first.")
        
        fixed_points = {}
        
        # Use the explicit fixed point designation from the "Fixed Point" column
        for _, row in self.coordinates_df.iterrows():
            point_id = row['Point ID']
            
            # Handle missing data and convert to string for comparison
            fixed_type_raw = row['Fixed Point']
            if pd.isna(fixed_type_raw):
                # Missing data is considered free
                fixed_points[point_id] = "free"
            else:
                # Convert to lowercase string and strip whitespace
                fixed_type = str(fixed_type_raw).lower().strip()
                
                if fixed_type in ['origin', 'o', '0,0']:
                    fixed_points[point_id] = "origin"
                elif fixed_type in ['y_axis', 'y', 'y-axis', '0,y']:
                    fixed_points[point_id] = "y_axis"
                else:
                    # Anything else (including empty strings, other values) is considered free
                    fixed_points[point_id] = "free"
        
        return fixed_points
    
    def get_line_constraints(self) -> List[Dict[str, any]]:
        """
        Get information about line constraints from the Line Group column.
        
        Returns:
            List of line constraint dictionaries with keys:
            - "line_id": The line group identifier
            - "point_ids": List of point IDs that should lie on the same line
            - "type": "arbitrary" (for now, could be extended to "horizontal", "vertical")
        """
        if self.coordinates_df is None:
            raise ValueError("Data not loaded yet. Call load_data() first.")
        
        line_constraints = []
        line_groups = {}
        
        # Group points by line group
        for _, row in self.coordinates_df.iterrows():
            point_id = row['Point ID']
            line_group = row['Line Group']
            
            if pd.notna(line_group) and str(line_group).strip():
                line_group_str = str(line_group).strip()
                # Split by comma and handle multiple line groups per point
                individual_groups = [group.strip() for group in line_group_str.split(',') if group.strip()]
                
                for group in individual_groups:
                    if group not in line_groups:
                        line_groups[group] = []
                    line_groups[group].append(point_id)
        
        # Create line constraints for groups with 2 or more points
        for line_id, point_ids in line_groups.items():
            if len(point_ids) >= 2:
                line_constraints.append({
                    "line_id": line_id,
                    "point_ids": point_ids,
                    "type": "arbitrary"  # Could be extended to "horizontal", "vertical"
                })
        
        return line_constraints
    
    def _load_distances_sheet(self) -> pd.DataFrame:
        """
        Load and validate the distances sheet.
        
        Returns:
            DataFrame with distance measurements (excluding rows with "ignore" in Status column)
            
        Raises:
            ValueError: If headers or data format is invalid
        """
        try:
            # Read the first sheet (distances)
            df = pd.read_excel(self.file_path, sheet_name=0)
            
            # Only use the first 4 columns (ignore any additional columns)
            if len(df.columns) >= 4:
                df = df.iloc[:, :4]
                # Rename columns to expected headers
                df.columns = self.expected_distance_headers
            else:
                raise ValueError(f"Distances sheet must have at least 4 columns, got {len(df.columns)}")
            
            # Filter out rows with "ignore" in the Status column (case-insensitive)
            if 'Status' in df.columns:
                # Convert Status column to string and handle missing values
                df['Status'] = df['Status'].fillna('').astype(str)
                # Count rows to be filtered out
                ignored_rows = df[df['Status'].str.lower().str.contains('ignore', na=False)]
                ignored_count = len(ignored_rows)
                # Filter out rows where Status contains "ignore" (case-insensitive)
                df = df[~df['Status'].str.lower().str.contains('ignore', na=False)]
                if ignored_count > 0:
                    print(f"Filtered out {ignored_count} rows with 'ignore' in Status column")
            
            # Validate data types
            if not df['Distance'].dtype in ['float64', 'int64']:
                raise ValueError("Distance column must contain numeric values")
            
            # Check for missing values in required columns (excluding Status column)
            required_columns = ['First point ID', 'Second point ID', 'Distance']
            if df[required_columns].isnull().any().any():
                raise ValueError("Missing values found in required columns of distances sheet")
            
            # Check for negative distances
            if (df['Distance'] <= 0).any():
                raise ValueError("All distances must be positive")
            
            return df
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Error reading distances sheet: {str(e)}")
    
    def _load_coordinates_sheet(self) -> pd.DataFrame:
        """
        Load and validate the coordinates sheet.
        
        Returns:
            DataFrame with initial coordinates
            
        Raises:
            ValueError: If headers or data format is invalid
        """
        try:
            # Read the second sheet (coordinates)
            df = pd.read_excel(self.file_path, sheet_name=1)
            
            # Use the first 5 columns (including Line Group)
            if len(df.columns) >= 5:
                df = df.iloc[:, :5]
                # Rename columns to expected headers
                df.columns = self.expected_coordinate_headers
            elif len(df.columns) >= 4:
                # Handle case where Line Group column is missing (backward compatibility)
                df = df.iloc[:, :4]
                df.columns = self.expected_coordinate_headers[:4]
                # Add empty Line Group column
                df['Line Group'] = ''
            else:
                raise ValueError(f"Coordinates sheet must have at least 4 columns, got {len(df.columns)}")
            
            # Validate data types
            if not df['x_guess'].dtype in ['float64', 'int64']:
                raise ValueError("x_guess column must contain numeric values")
            if not df['y_guess'].dtype in ['float64', 'int64']:
                raise ValueError("y_guess column must contain numeric values")
            
            # Check for missing values (allow missing values in Fixed Point and Line Group columns)
            required_columns = ['Point ID', 'x_guess', 'y_guess']
            if df[required_columns].isnull().any().any():
                raise ValueError("Missing values found in required columns (Point ID, x_guess, y_guess)")
            
            # Fill missing values in optional columns
            df['Fixed Point'] = df['Fixed Point'].fillna('')
            df['Line Group'] = df['Line Group'].fillna('')
            
            return df
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Error reading coordinates sheet: {str(e)}")
    
    def _validate_data_consistency(self):
        """
        Validate consistency between distance and coordinate data.
        
        Raises:
            ValueError: If point IDs don't match between sheets
        """
        # Get all point IDs from both sheets
        distance_points = set()
        for _, row in self.distances_df.iterrows():
            distance_points.add(row['First point ID'])
            distance_points.add(row['Second point ID'])
        
        coordinate_points = set(self.coordinates_df['Point ID'])
        
        # Check if all points in distances have corresponding coordinates
        missing_in_coordinates = distance_points - coordinate_points
        if missing_in_coordinates:
            raise ValueError(
                f"Points found in distances but missing from coordinates: {missing_in_coordinates}"
            )
        
        # Check if all points in coordinates are used in distances
        unused_in_distances = coordinate_points - distance_points
        if unused_in_distances:
            print(f"Warning: Points in coordinates but not used in distances: {unused_in_distances}")
    
    def _convert_distances_to_dict(self) -> Dict[Tuple[str, str], float]:
        """
        Convert distances DataFrame to dictionary format.
        
        Returns:
            Dictionary mapping (point_id1, point_id2) tuples to distances
        """
        distances = {}
        for _, row in self.distances_df.iterrows():
            point1 = row['First point ID']
            point2 = row['Second point ID']
            distance = float(row['Distance'])
            
            # Ensure consistent ordering of point IDs
            if point1 < point2:
                key = (point1, point2)
            else:
                key = (point2, point1)
            
            distances[key] = distance
        
        return distances
    
    def _convert_coordinates_to_dict(self) -> Dict[str, Tuple[float, float]]:
        """
        Convert coordinates DataFrame to dictionary format.
        
        Returns:
            Dictionary mapping point IDs to (x, y) coordinate tuples
        """
        coordinates = {}
        for _, row in self.coordinates_df.iterrows():
            point_id = row['Point ID']
            x = float(row['x_guess'])
            y = float(row['y_guess'])
            coordinates[point_id] = (x, y)
        
        return coordinates
    
    def get_all_point_ids(self) -> List[str]:
        """
        Get list of all point IDs from the data.
        
        Returns:
            List of all point IDs
        """
        if self.coordinates_df is None:
            raise ValueError("Data not loaded yet. Call load_data() first.")
        
        return list(self.coordinates_df['Point ID'])
    
    def get_connectivity_info(self) -> Tuple[List[Set[str]], bool]:
        """
        Analyze connectivity of the distance measurements.
        
        Returns:
            Tuple of (connected_components, is_fully_connected)
        """
        if self.distances_df is None:
            raise ValueError("Data not loaded yet. Call load_data() first.")
        
        # Build adjacency list
        adjacency = {}
        all_points = set()
        
        for _, row in self.distances_df.iterrows():
            point1 = row['First point ID']
            point2 = row['Second point ID']
            
            all_points.add(point1)
            all_points.add(point2)
            
            if point1 not in adjacency:
                adjacency[point1] = set()
            if point2 not in adjacency:
                adjacency[point2] = set()
            
            adjacency[point1].add(point2)
            adjacency[point2].add(point1)
        
        # Find connected components using DFS
        visited = set()
        connected_components = []
        
        for point in all_points:
            if point not in visited:
                component = set()
                self._dfs(point, adjacency, visited, component)
                connected_components.append(component)
        
        is_fully_connected = len(connected_components) == 1
        
        return connected_components, is_fully_connected
    
    def _dfs(self, node: str, adjacency: Dict[str, Set[str]], 
             visited: Set[str], component: Set[str]):
        """
        Depth-first search to find connected component.
        
        Args:
            node: Current node to visit
            adjacency: Adjacency list representation of the graph
            visited: Set of already visited nodes
            component: Set to collect nodes in current component
        """
        visited.add(node)
        component.add(node)
        
        for neighbor in adjacency.get(node, set()):
            if neighbor not in visited:
                self._dfs(neighbor, adjacency, visited, component) 

    def analyze_data_quality(self) -> Dict:
        """
        Perform comprehensive data quality analysis to identify potential issues.
        
        Returns:
            Dictionary containing data quality analysis results
        """
        if self.distances_df is None or self.coordinates_df is None:
            raise ValueError("Data not loaded yet. Call load_data() first.")
        
        analysis = {
            'point_consistency': self._check_point_consistency(),
            'distance_statistics': self._analyze_distance_statistics(),
            'coordinate_analysis': self._analyze_coordinates(),
            'connectivity_analysis': self._analyze_connectivity(),
            'potential_issues': []
        }
        
        # Identify potential issues
        issues = []
        
        # Check for duplicate measurements
        duplicates = self._find_duplicate_measurements()
        if duplicates:
            issues.append(f"Found {len(duplicates)} duplicate distance measurements")
            analysis['duplicate_measurements'] = duplicates
        
        # Check for very short or very long distances
        extreme_distances = self._find_extreme_distances()
        if extreme_distances:
            issues.append(f"Found {len(extreme_distances)} potentially problematic distances")
            analysis['extreme_distances'] = extreme_distances
        
        # Check for isolated points
        isolated_points = self._find_isolated_points()
        if isolated_points:
            issues.append(f"Found {len(isolated_points)} points with only one measurement")
            analysis['isolated_points'] = isolated_points
        
        # Check for coordinate outliers
        coordinate_outliers = self._find_coordinate_outliers()
        if coordinate_outliers:
            issues.append(f"Found {len(coordinate_outliers)} coordinate outliers")
            analysis['coordinate_outliers'] = coordinate_outliers
        
        analysis['potential_issues'] = issues
        return analysis 

    def _check_point_consistency(self) -> Dict:
        """Check consistency between distance measurements and coordinate data."""
        distance_points = set()
        for _, row in self.distances_df.iterrows():
            distance_points.add(row['First point ID'])
            distance_points.add(row['Second point ID'])
        
        coordinate_points = set(self.coordinates_df['Point ID'])
        
        missing_in_coordinates = distance_points - coordinate_points
        missing_in_distances = coordinate_points - distance_points
        
        return {
            'points_in_distances_only': list(missing_in_coordinates),
            'points_in_coordinates_only': list(missing_in_distances),
            'total_distance_points': len(distance_points),
            'total_coordinate_points': len(coordinate_points),
            'consistent': len(missing_in_coordinates) == 0 and len(missing_in_distances) == 0
        }
    
    def _analyze_distance_statistics(self) -> Dict:
        """Analyze distance measurement statistics."""
        distances = self.distances_df['Distance']
        
        return {
            'count': len(distances),
            'min': float(distances.min()),
            'max': float(distances.max()),
            'mean': float(distances.mean()),
            'median': float(distances.median()),
            'std': float(distances.std()),
            'range': float(distances.max() - distances.min())
        }
    
    def _analyze_coordinates(self) -> Dict:
        """Analyze coordinate data."""
        x_coords = self.coordinates_df['x_guess']
        y_coords = self.coordinates_df['y_guess']
        
        return {
            'x_stats': {
                'min': float(x_coords.min()),
                'max': float(x_coords.max()),
                'mean': float(x_coords.mean()),
                'std': float(x_coords.std())
            },
            'y_stats': {
                'min': float(y_coords.min()),
                'max': float(y_coords.max()),
                'mean': float(y_coords.mean()),
                'std': float(y_coords.std())
            },
            'coordinate_range': {
                'x_range': float(x_coords.max() - x_coords.min()),
                'y_range': float(y_coords.max() - y_coords.min())
            }
        }
    
    def _analyze_connectivity(self) -> Dict:
        """Analyze the connectivity graph of measurements."""
        # Build adjacency list
        adjacency = {}
        for _, row in self.distances_df.iterrows():
            point1 = row['First point ID']
            point2 = row['Second point ID']
            
            if point1 not in adjacency:
                adjacency[point1] = set()
            if point2 not in adjacency:
                adjacency[point2] = set()
            
            adjacency[point1].add(point2)
            adjacency[point2].add(point1)
        
        # Analyze connectivity
        point_degrees = {point: len(neighbors) for point, neighbors in adjacency.items()}
        
        return {
            'total_points': len(adjacency),
            'total_measurements': len(self.distances_df),
            'average_degree': sum(point_degrees.values()) / len(point_degrees) if point_degrees else 0,
            'min_degree': min(point_degrees.values()) if point_degrees else 0,
            'max_degree': max(point_degrees.values()) if point_degrees else 0,
            'point_degrees': point_degrees
        }
    
    def _find_duplicate_measurements(self) -> List[Dict]:
        """Find duplicate or near-duplicate distance measurements."""
        duplicates = []
        
        # Check for exact duplicates
        seen_pairs = set()
        for _, row in self.distances_df.iterrows():
            point1, point2 = row['First point ID'], row['Second point ID']
            distance = row['Distance']
            
            # Normalize pair order
            if point1 > point2:
                point1, point2 = point2, point1
            
            pair_key = (point1, point2, distance)
            if pair_key in seen_pairs:
                duplicates.append({
                    'type': 'exact_duplicate',
                    'point1': point1,
                    'point2': point2,
                    'distance': distance
                })
            seen_pairs.add(pair_key)
        
        return duplicates
    
    def _find_extreme_distances(self) -> List[Dict]:
        """Find distances that are unusually short or long."""
        distances = self.distances_df['Distance']
        mean_dist = distances.mean()
        std_dist = distances.std()
        
        extreme_distances = []
        
        for _, row in self.distances_df.iterrows():
            distance = row['Distance']
            point1, point2 = row['First point ID'], row['Second point ID']
            
            # Check if distance is more than 3 standard deviations from mean
            z_score = abs(distance - mean_dist) / std_dist if std_dist > 0 else 0
            
            if z_score > 3:
                extreme_distances.append({
                    'point1': point1,
                    'point2': point2,
                    'distance': distance,
                    'z_score': z_score,
                    'type': 'very_short' if distance < mean_dist else 'very_long'
                })
        
        return extreme_distances
    
    def _find_isolated_points(self) -> List[str]:
        """Find points that have only one measurement."""
        point_counts = {}
        
        for _, row in self.distances_df.iterrows():
            point1 = row['First point ID']
            point2 = row['Second point ID']
            
            point_counts[point1] = point_counts.get(point1, 0) + 1
            point_counts[point2] = point_counts.get(point2, 0) + 1
        
        return [point for point, count in point_counts.items() if count == 1]
    
    def _find_coordinate_outliers(self) -> List[Dict]:
        """Find coordinate values that are statistical outliers."""
        x_coords = self.coordinates_df['x_guess']
        y_coords = self.coordinates_df['y_guess']
        
        x_mean, x_std = x_coords.mean(), x_coords.std()
        y_mean, y_std = y_coords.mean(), y_coords.std()
        
        outliers = []
        
        for _, row in self.coordinates_df.iterrows():
            point_id = row['Point ID']
            x, y = row['x_guess'], row['y_guess']
            
            x_z_score = abs(x - x_mean) / x_std if x_std > 0 else 0
            y_z_score = abs(y - y_mean) / y_std if y_std > 0 else 0
            
            if x_z_score > 3 or y_z_score > 3:
                outliers.append({
                    'point_id': point_id,
                    'x': x,
                    'y': y,
                    'x_z_score': x_z_score,
                    'y_z_score': y_z_score
                })
        
        return outliers
    
    def print_data_quality_report(self):
        """Print a comprehensive data quality report."""
        analysis = self.analyze_data_quality()
        
        print("\n" + "="*80)
        print("DATA QUALITY ANALYSIS REPORT")
        print("="*80)
        
        # Point consistency
        consistency = analysis['point_consistency']
        print("\nPOINT CONSISTENCY:")
        if consistency['consistent']:
            print("  ✅ All points are consistent between distance and coordinate data")
        else:
            print("  ❌ Point consistency issues found:")
            if consistency['points_in_distances_only']:
                print(f"    Points in distances but missing from coordinates: {consistency['points_in_distances_only']}")
            if consistency['points_in_coordinates_only']:
                print(f"    Points in coordinates but not used in distances: {consistency['points_in_coordinates_only']}")
        
        # Distance statistics
        dist_stats = analysis['distance_statistics']
        print(f"\nDISTANCE STATISTICS:")
        print(f"  Total measurements: {dist_stats['count']}")
        print(f"  Range: {dist_stats['min']:.3f} - {dist_stats['max']:.3f}")
        print(f"  Mean: {dist_stats['mean']:.3f}")
        print(f"  Standard deviation: {dist_stats['std']:.3f}")
        
        # Connectivity analysis
        connectivity = analysis['connectivity_analysis']
        print(f"\nCONNECTIVITY ANALYSIS:")
        print(f"  Total points: {connectivity['total_points']}")
        print(f"  Average connections per point: {connectivity['average_degree']:.1f}")
        print(f"  Min connections: {connectivity['min_degree']}")
        print(f"  Max connections: {connectivity['max_degree']}")
        
        # Potential issues
        if analysis['potential_issues']:
            print(f"\nPOTENTIAL ISSUES FOUND:")
            for issue in analysis['potential_issues']:
                print(f"  ⚠️  {issue}")
        else:
            print(f"\n✅ No obvious data quality issues detected")
        
        print("\n" + "="*80) 