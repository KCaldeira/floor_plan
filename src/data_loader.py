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
        self.expected_distance_headers = ["First point ID", "Second point ID", "Distance", "Status", "Line Orientation", "Weight"]
        
        # Expected headers for coordinates sheet
        self.expected_coordinate_headers = ["Point ID", "x_guess", "y_guess", "Fixed Point"]
        
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
        
        fixed_points_info = {}
        
        for _, row in self.coordinates_df.iterrows():
            point_id = row['Point ID']
            fixed_point = row['Fixed Point']
            
            if pd.notna(fixed_point) and str(fixed_point).strip():
                fixed_point_str = str(fixed_point).strip().lower()
                if fixed_point_str in ['origin', 'o']:
                    fixed_points_info[point_id] = 'origin'
                elif fixed_point_str in ['y_axis', 'y']:
                    fixed_points_info[point_id] = 'y_axis'
                else:
                    fixed_points_info[point_id] = 'free'
            else:
                fixed_points_info[point_id] = 'free'
        
        return fixed_points_info
    
    def get_line_orientations(self) -> Dict[Tuple[str, str], str]:
        """
        Get line orientation information for all measurements and structural lines.
        
        Returns:
            Dictionary mapping (point_id1, point_id2) to line orientation:
            - "H": Horizontal line constraint
            - "V": Vertical line constraint
            - Empty string: No line orientation constraint
            
        Note: This includes both distance measurements and structural lines without distances.
        """
        if self.distances_df is None:
            raise ValueError("Data not loaded yet. Call load_data() first.")
        
        # Load the original distances sheet to get all rows (including structural lines)
        try:
            original_df = pd.read_excel(self.file_path, sheet_name=0)
            
            # Handle columns - need at least 4, optionally 5 or 6
            if len(original_df.columns) >= 4:
                if len(original_df.columns) >= 6:
                    # Use first 6 columns (including Line Orientation and Weight)
                    original_df = original_df.iloc[:, :6]
                    original_df.columns = self.expected_distance_headers
                elif len(original_df.columns) >= 5:
                    # Use first 5 columns (including Line Orientation, no Weight)
                    original_df = original_df.iloc[:, :5]
                    original_df.columns = self.expected_distance_headers[:5]
                    # Add default Weight column
                    original_df['Weight'] = 1.0
                else:
                    # Use first 4 columns (no Line Orientation, no Weight)
                    original_df = original_df.iloc[:, :4]
                    original_df.columns = self.expected_distance_headers[:4]
                    # Add empty Line Orientation column and default Weight column
                    original_df['Line Orientation'] = ''
                    original_df['Weight'] = 1.0
            else:
                raise ValueError(f"Distances sheet must have at least 4 columns, got {len(original_df.columns)}")
            
            # Filter out rows with "ignore" in the Status column (case-insensitive)
            if 'Status' in original_df.columns:
                # Convert Status column to string and handle missing values
                original_df['Status'] = original_df['Status'].fillna('').astype(str)
                # Filter out rows where Status contains "ignore" (case-insensitive)
                original_df = original_df[~original_df['Status'].str.lower().str.contains('ignore', na=False)]
            
            # Process Line Orientation column
            if 'Line Orientation' in original_df.columns:
                # Convert to string, remove spaces, convert to uppercase
                original_df['Line Orientation'] = original_df['Line Orientation'].fillna('').astype(str).str.replace(' ', '').str.upper()
                # Keep only 'H' and 'V' values, ignore others
                original_df['Line Orientation'] = original_df['Line Orientation'].apply(lambda x: x if x in ['H', 'V'] else '')
            
        except Exception as e:
            raise ValueError(f"Error reading original distances sheet for line orientations: {str(e)}")
        
        line_orientations = {}
        for _, row in original_df.iterrows():
            point1 = row['First point ID']
            point2 = row['Second point ID']
            orientation = row.get('Line Orientation', '')
            
            # Ensure consistent ordering (smaller ID first)
            if point1 < point2:
                key = (point1, point2)
            else:
                key = (point2, point1)
            
            line_orientations[key] = orientation
        
        return line_orientations
    
    def get_weights(self) -> Dict[Tuple[str, str], float]:
        """
        Get weight information for distance measurements.
        
        Returns:
            Dictionary mapping (point_id1, point_id2) to weight value.
            Default weight is 1.0 if not specified.
        """
        if self.distances_df is None:
            raise ValueError("Data not loaded yet. Call load_data() first.")
        
        weights = {}
        for _, row in self.distances_df.iterrows():
            point1 = row['First point ID']
            point2 = row['Second point ID']
            weight = row.get('Weight', 1.0)
            
            # Ensure consistent ordering (smaller ID first)
            if point1 < point2:
                key = (point1, point2)
            else:
                key = (point2, point1)
            
            weights[key] = weight
        
        return weights
    
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
            
            # Handle columns - need at least 4, optionally 5 or 6
            if len(df.columns) >= 4:
                if len(df.columns) >= 6:
                    # Use first 6 columns (including Line Orientation and Weight)
                    df = df.iloc[:, :6]
                    df.columns = self.expected_distance_headers
                elif len(df.columns) >= 5:
                    # Use first 5 columns (including Line Orientation, no Weight)
                    df = df.iloc[:, :5]
                    df.columns = self.expected_distance_headers[:5]
                    # Add default Weight column
                    df['Weight'] = 1.0
                else:
                    # Use first 4 columns (no Line Orientation, no Weight)
                    df = df.iloc[:, :4]
                    df.columns = self.expected_distance_headers[:4]
                    # Add empty Line Orientation column and default Weight column
                    df['Line Orientation'] = ''
                    df['Weight'] = 1.0
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
            required_columns = ['First point ID', 'Second point ID']
            if df[required_columns].isnull().any().any():
                raise ValueError("Missing values found in required columns of distances sheet")
            
            # Allow missing distance values for structural lines (line orientation constraints only)
            # Filter out rows with missing distances for distance optimization
            df_with_distances = df[df['Distance'].notna()]
            df_structural_only = df[df['Distance'].isna()]
            
            if len(df_structural_only) > 0:
                print(f"Found {len(df_structural_only)} structural lines without distance measurements")
                # Check that structural lines have line orientation specified
                structural_with_orientation = df_structural_only[df_structural_only['Line Orientation'].isin(['H', 'V'])]
                if len(structural_with_orientation) != len(df_structural_only):
                    print(f"Warning: {len(df_structural_only) - len(structural_with_orientation)} structural lines without H/V orientation")
            
            # Use only rows with valid distances for distance optimization
            df = df_with_distances
            
            # Check for negative distances
            if (df['Distance'] <= 0).any():
                raise ValueError("All distances must be positive")
            
            # Process Line Orientation column
            if 'Line Orientation' in df.columns:
                # Convert to string, remove spaces, convert to uppercase
                df['Line Orientation'] = df['Line Orientation'].fillna('').astype(str).str.replace(' ', '').str.upper()
                # Keep only 'H' and 'V' values, ignore others
                df['Line Orientation'] = df['Line Orientation'].apply(lambda x: x if x in ['H', 'V'] else '')
            
            # Process Weight column
            if 'Weight' in df.columns:
                # Fill missing values with default weight of 1.0
                df['Weight'] = df['Weight'].fillna(1.0)
                # Validate that weights are numeric and positive
                if not df['Weight'].dtype in ['float64', 'int64']:
                    raise ValueError("Weight column must contain numeric values")
                if (df['Weight'] <= 0).any():
                    raise ValueError("All weights must be positive")
            
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
            
            # Check if we have the expected number of columns
            if len(df.columns) >= 4:
                # Use the first 4 columns
                df = df.iloc[:, :4]
                # Rename columns to expected headers
                df.columns = self.expected_coordinate_headers
            else:
                raise ValueError(f"Coordinates sheet must have at least 4 columns, got {len(df.columns)}")
            
            # Validate data types for coordinate columns
            if not df['x_guess'].dtype in ['float64', 'int64']:
                raise ValueError("x_guess column must contain numeric values")
            if not df['y_guess'].dtype in ['float64', 'int64']:
                raise ValueError("y_guess column must contain numeric values")
            
            # Check for missing values in required columns
            required_columns = ['Point ID', 'x_guess', 'y_guess']
            if df[required_columns].isnull().any().any():
                raise ValueError("Missing values found in required columns of coordinates sheet")
            
            # Handle missing values in Fixed Point column
            df['Fixed Point'] = df['Fixed Point'].fillna('')
            
            return df
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Error reading coordinates sheet: {str(e)}")
    
    def _validate_data_consistency(self):
        """
        Validate consistency between distance and coordinate data.
        
        Raises:
            ValueError: If data is inconsistent
        """
        # Get all point IDs from both sheets
        distance_points = set()
        for _, row in self.distances_df.iterrows():
            distance_points.add(row['First point ID'])
            distance_points.add(row['Second point ID'])
        
        coordinate_points = set(self.coordinates_df['Point ID'])
        
        # Check for points in distances but not in coordinates
        missing_in_coords = distance_points - coordinate_points
        if missing_in_coords:
            raise ValueError(f"Points in distance measurements missing from coordinates: {missing_in_coords}")
        
        # Check for points in coordinates but not in distances (this is OK, just warn)
        unused_points = coordinate_points - distance_points
        if unused_points:
            print(f"Warning: Points in coordinates but not in distance measurements: {unused_points}")
    
    def _convert_distances_to_dict(self) -> Dict[Tuple[str, str], float]:
        """
        Convert distances DataFrame to dictionary format.
        
        Returns:
            Dictionary mapping (point_id1, point_id2) to distance
        """
        distances = {}
        for _, row in self.distances_df.iterrows():
            point1 = row['First point ID']
            point2 = row['Second point ID']
            distance = row['Distance']
            
            # Ensure consistent ordering (smaller ID first)
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
            Dictionary mapping point IDs to (x, y) coordinates
        """
        coordinates = {}
        for _, row in self.coordinates_df.iterrows():
            point_id = row['Point ID']
            x = row['x_guess']
            y = row['y_guess']
            coordinates[point_id] = (x, y)
        
        return coordinates
    
    def get_all_point_ids(self) -> List[str]:
        """
        Get all point IDs from the data.
        
        Returns:
            List of all point IDs
        """
        if self.coordinates_df is None:
            raise ValueError("Data not loaded yet. Call load_data() first.")
        
        return sorted(self.coordinates_df['Point ID'].tolist())
    
    def get_connectivity_info(self) -> Tuple[List[Set[str]], bool]:
        """
        Analyze connectivity between points based on distance measurements.
        
        Returns:
            Tuple of (connected_components, is_connected)
            - connected_components: List of sets, each containing point IDs in a connected component
            - is_connected: True if all points are in a single connected component
        """
        if self.distances_df is None:
            raise ValueError("Data not loaded yet. Call load_data() first.")
        
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
        
        # Find connected components using DFS
        visited = set()
        components = []
        
        for point in adjacency.keys():
            if point not in visited:
                component = set()
                self._dfs(point, adjacency, visited, component)
                components.append(component)
        
        # Check if all points are connected
        is_connected = len(components) == 1
        
        return components, is_connected
    
    def _dfs(self, node: str, adjacency: Dict[str, Set[str]], 
             visited: Set[str], component: Set[str]):
        """
        Depth-first search to find connected component.
        
        Args:
            node: Current node to visit
            adjacency: Adjacency list representation of graph
            visited: Set of visited nodes
            component: Set to add nodes in current component
        """
        visited.add(node)
        component.add(node)
        
        if node in adjacency:
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    self._dfs(neighbor, adjacency, visited, component)
    
    def analyze_data_quality(self) -> Dict:
        """
        Perform comprehensive data quality analysis.
        
        Returns:
            Dictionary containing various data quality metrics and issues
        """
        if self.distances_df is None or self.coordinates_df is None:
            raise ValueError("Data not loaded yet. Call load_data() first.")
        
        analysis = {}
        
        # Point consistency analysis
        analysis['point_consistency'] = self._check_point_consistency()
        
        # Distance statistics
        analysis['distance_statistics'] = self._analyze_distance_statistics()
        
        # Coordinate analysis
        analysis['coordinate_analysis'] = self._analyze_coordinates()
        
        # Connectivity analysis
        analysis['connectivity'] = self._analyze_connectivity()
        
        # Find specific issues
        analysis['duplicate_measurements'] = self._find_duplicate_measurements()
        analysis['extreme_distances'] = self._find_extreme_distances()
        analysis['isolated_points'] = self._find_isolated_points()
        analysis['coordinate_outliers'] = self._find_coordinate_outliers()
        
        return analysis
    
    def _check_point_consistency(self) -> Dict:
        """
        Check consistency between points in distance and coordinate data.
        
        Returns:
            Dictionary with consistency information
        """
        distance_points = set()
        for _, row in self.distances_df.iterrows():
            distance_points.add(row['First point ID'])
            distance_points.add(row['Second point ID'])
        
        coordinate_points = set(self.coordinates_df['Point ID'])
        
        missing_in_coords = distance_points - coordinate_points
        missing_in_distances = coordinate_points - distance_points
        
        return {
            'consistent': len(missing_in_coords) == 0,
            'missing_in_coordinates': list(missing_in_coords),
            'missing_in_distances': list(missing_in_distances),
            'total_distance_points': len(distance_points),
            'total_coordinate_points': len(coordinate_points)
        }
    
    def _analyze_distance_statistics(self) -> Dict:
        """
        Analyze statistics of distance measurements.
        
        Returns:
            Dictionary with distance statistics
        """
        distances = self.distances_df['Distance']
        
        return {
            'count': len(distances),
            'mean': distances.mean(),
            'std': distances.std(),
            'min': distances.min(),
            'max': distances.max(),
            'median': distances.median()
        }
    
    def _analyze_coordinates(self) -> Dict:
        """
        Analyze coordinate data.
        
        Returns:
            Dictionary with coordinate analysis
        """
        x_coords = self.coordinates_df['x_guess']
        y_coords = self.coordinates_df['y_guess']
        
        return {
            'x_stats': {
                'mean': x_coords.mean(),
                'std': x_coords.std(),
                'min': x_coords.min(),
                'max': x_coords.max()
            },
            'y_stats': {
                'mean': y_coords.mean(),
                'std': y_coords.std(),
                'min': y_coords.min(),
                'max': y_coords.max()
            },
            'fixed_points': {
                'origin': len(self.coordinates_df[self.coordinates_df['Fixed Point'].str.lower().isin(['origin', 'o'])])
            }
        }
    
    def _analyze_connectivity(self) -> Dict:
        """
        Analyze connectivity between points.
        
        Returns:
            Dictionary with connectivity information
        """
        components, is_connected = self.get_connectivity_info()
        
        return {
            'is_connected': is_connected,
            'num_components': len(components),
            'component_sizes': [len(comp) for comp in components],
            'largest_component_size': max(len(comp) for comp in components) if components else 0
        }
    
    def _find_duplicate_measurements(self) -> List[Dict]:
        """
        Find duplicate distance measurements.
        
        Returns:
            List of duplicate measurement dictionaries
        """
        duplicates = []
        seen_pairs = set()
        
        for _, row in self.distances_df.iterrows():
            point1 = row['First point ID']
            point2 = row['Second point ID']
            distance = row['Distance']
            
            # Create consistent pair key
            if point1 < point2:
                pair_key = (point1, point2)
            else:
                pair_key = (point2, point1)
            
            if pair_key in seen_pairs:
                duplicates.append({
                    'point1': point1,
                    'point2': point2,
                    'distance': distance
                })
            else:
                seen_pairs.add(pair_key)
        
        return duplicates
    
    def _find_extreme_distances(self) -> List[Dict]:
        """
        Find distance measurements that are statistical outliers.
        
        Returns:
            List of extreme distance dictionaries
        """
        distances = self.distances_df['Distance']
        mean = distances.mean()
        std = distances.std()
        
        extreme_distances = []
        for _, row in self.distances_df.iterrows():
            distance = row['Distance']
            z_score = abs((distance - mean) / std) if std > 0 else 0
            
            if z_score > 2.0:  # More than 2 standard deviations
                extreme_distances.append({
                    'point1': row['First point ID'],
                    'point2': row['Second point ID'],
                    'distance': distance,
                    'z_score': z_score,
                    'type': 'high' if distance > mean else 'low'
                })
        
        return extreme_distances
    
    def _find_isolated_points(self) -> List[str]:
        """
        Find points that have only one distance measurement.
        
        Returns:
            List of isolated point IDs
        """
        point_counts = {}
        
        for _, row in self.distances_df.iterrows():
            point1 = row['First point ID']
            point2 = row['Second point ID']
            
            point_counts[point1] = point_counts.get(point1, 0) + 1
            point_counts[point2] = point_counts.get(point2, 0) + 1
        
        return [point for point, count in point_counts.items() if count == 1]
    
    def _find_coordinate_outliers(self) -> List[Dict]:
        """
        Find coordinate values that are statistical outliers.
        
        Returns:
            List of coordinate outlier dictionaries
        """
        x_coords = self.coordinates_df['x_guess']
        y_coords = self.coordinates_df['y_guess']
        
        x_mean, x_std = x_coords.mean(), x_coords.std()
        y_mean, y_std = y_coords.mean(), y_coords.std()
        
        outliers = []
        for _, row in self.coordinates_df.iterrows():
            point_id = row['Point ID']
            x, y = row['x_guess'], row['y_guess']
            
            x_z_score = abs((x - x_mean) / x_std) if x_std > 0 else 0
            y_z_score = abs((y - y_mean) / y_std) if y_std > 0 else 0
            
            if x_z_score > 2.0 or y_z_score > 2.0:
                outliers.append({
                    'point_id': point_id,
                    'x': x,
                    'y': y,
                    'x_z_score': x_z_score,
                    'y_z_score': y_z_score
                })
        
        return outliers
    
    def print_data_quality_report(self):
        """
        Print a comprehensive data quality report.
        """
        if self.distances_df is None or self.coordinates_df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        print("\n" + "="*60)
        print("DATA QUALITY REPORT")
        print("="*60)
        
        # Basic statistics
        print(f"Distance measurements: {len(self.distances_df)}")
        print(f"Points: {len(self.coordinates_df)}")
        
        # Point consistency
        consistency = self._check_point_consistency()
        print(f"\nPoint consistency: {'✓' if consistency['consistent'] else '✗'}")
        if not consistency['consistent']:
            print(f"  Missing in coordinates: {consistency['missing_in_coordinates']}")
        
        # Distance statistics
        dist_stats = self._analyze_distance_statistics()
        print(f"\nDistance statistics:")
        print(f"  Mean: {dist_stats['mean']:.3f}")
        print(f"  Std: {dist_stats['std']:.3f}")
        print(f"  Range: {dist_stats['min']:.3f} - {dist_stats['max']:.3f}")
        
        # Connectivity
        connectivity = self._analyze_connectivity()
        print(f"\nConnectivity: {'✓' if connectivity['is_connected'] else '✗'}")
        if not connectivity['is_connected']:
            print(f"  Components: {connectivity['num_components']}")
            print(f"  Component sizes: {connectivity['component_sizes']}")
        
        # Issues
        analysis = self.analyze_data_quality()
        
        if analysis['duplicate_measurements']:
            print(f"\nDuplicate measurements: {len(analysis['duplicate_measurements'])}")
        
        if analysis['extreme_distances']:
            print(f"\nExtreme distances: {len(analysis['extreme_distances'])}")
        
        if analysis['isolated_points']:
            print(f"\nIsolated points: {len(analysis['isolated_points'])}")
        
        if analysis['coordinate_outliers']:
            print(f"\nCoordinate outliers: {len(analysis['coordinate_outliers'])}")
        
        print("\n" + "="*60) 