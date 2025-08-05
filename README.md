# Floor Plan Coordinate Estimation

A Python project for estimating x,y coordinates of points in a floor plan using distance measurements between point pairs.

## Overview

This project solves the problem of determining the spatial coordinates of multiple points when given distance measurements between some (but not all) point pairs. This is commonly encountered in:

- Floor plan digitization
- Indoor positioning systems
- Surveying and mapping
- Network topology reconstruction

## Problem Statement

Given:
- A set of points (P₁, P₂, ..., Pₙ)
- Distance measurements dᵢⱼ between some point pairs (Pᵢ, Pⱼ)
- Initial guess coordinates for all points

Find:
- Optimal x,y coordinates for all points that minimize the error between measured and calculated distances

## Mathematical Approach

The problem is formulated as a nonlinear least squares optimization:

```
minimize Σ(dᵢⱼ_measured - dᵢⱼ_calculated)²
```

where:
- dᵢⱼ_calculated = √((xᵢ - xⱼ)² + (yᵢ - yⱼ)²)
- (xᵢ, yᵢ) are the coordinates of point Pᵢ

## Constraints and Assumptions

To reduce translational and rotational degrees of freedom:
- One point is fixed at (0, 0)
- Another point is fixed at (0, y) where y > 0

## Project Structure

```
floor_plan/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── coordinate_estimator.py
│   ├── distance_calculator.py
│   ├── data_loader.py
│   ├── visualization.py
│   ├── output_analyzer.py
│   └── output_manager.py
├── data/
│   ├── sample_measurements.xlsx
│   ├── flexible_example.xlsx
│   └── 50_e_1_st_measurements.xlsx
├── tests/
│   ├── __init__.py
│   ├── test_coordinate_estimator.py
│   └── test_distance_calculator.py
└── examples/
    ├── simple_example.py
    ├── flexible_example.py
    └── enhanced_example.py
```

## Key Components

### Coordinate Estimator
- Implements nonlinear least squares optimization
- Uses scipy.optimize for minimization
- Handles missing distance measurements
- Optimizes both x and y coordinates simultaneously

### Distance Calculator
- Computes Euclidean distances between points
- Calculates error metrics
- Validates distance measurements

### Data Loader
- Reads Excel files with distance measurements and initial coordinates
- Validates data format and consistency
- Handles alphanumeric point IDs

### Visualization
- Plots point locations and connections
- Shows distance measurements
- Displays optimization convergence

### Output Analyzer
- Generates enhanced distance analysis with optimized distances and distance errors
- Creates enhanced coordinate analysis with error metrics
- Provides sorted and organized output tables
- Calculates per-point error statistics
- Identifies distance measurements with highest errors



## Input Data Format

The project expects input data in Excel (.xlsx) format with two sheets:

### Sheet 1: Distance Measurements
- **Column A**: First point ID (alphanumeric)
- **Column B**: Second point ID (alphanumeric) 
- **Column C**: Distance between the two points
- **Column D**: Status (optional - rows with "ignore" in this column will be excluded)
- **Headers**: "First point ID", "Second point ID", "Distance", "Status"

### Sheet 2: Initial Coordinates
- **Column A**: Point ID (alphanumeric)
- **Column B**: Initial x-coordinate
- **Column C**: Initial y-coordinate
- **Column D**: Fixed Point designation
- **Headers**: "Point ID", "x_guess", "y_guess", "Fixed Point"

**Fixed Point column values:**
- `"origin"` or `"o"` (case-insensitive) - Point fixed at (0,0)
- `"y_axis"` or `"y"` (case-insensitive) - Point fixed at (0,y) where y > 0
- **Anything else** (including missing data, empty strings, other values) - Point free to optimize

### Example Excel Structure:

```
Sheet 1: Distances
A1: "First point ID"    B1: "Second point ID"    C1: "Distance"    D1: "Status"
A2: "A"                 B2: "B"                  C2: 5.0          D2: (empty)
A3: "B"                 B3: "C"                  C3: 3.0          D3: "ignore"
A4: "A"                 B4: "C"                  C4: 4.0          D4: (empty)

Sheet 2: Initial Coordinates
A1: "Point ID"    B1: "x_guess"    C1: "y_guess"    D1: "Fixed Point"
A2: "C"           B2: 3.0          C2: 4.0          D2: (empty/missing)
A3: "A"           B3: 0.0          C3: 0.0          D3: "origin"
A4: "B"           B4: 0.0          C4: 5.0          D4: "y_axis"
```

**Note:** The "Fixed Point" column is very flexible:
- Only `"origin"` and `"y_axis"` (case-insensitive) designate fixed points
- Missing data, empty strings, or any other value designates free points
- This makes it easy to use - just leave the column empty for free points

**Note:** The "Status" column in the distances sheet is optional:
- Rows with "ignore" (case-insensitive) in the Status column will be excluded from processing
- Empty cells or any other value in the Status column will be included
- This allows you to easily exclude problematic measurements without deleting them

## Usage

### Basic Example

```python
from src.coordinate_estimator import CoordinateEstimator
from src.distance_calculator import DistanceCalculator
from src.data_loader import ExcelDataLoader
from src.visualization import FloorPlanVisualizer

# Load data from Excel file
loader = ExcelDataLoader("data/measurements.xlsx")
distances, initial_coords = loader.load_data()

# Create estimator and optimize
estimator = CoordinateEstimator(distances, initial_coords)
result = estimator.optimize()

# Create visualizations (saved to timestamped output directory)
visualizer = FloorPlanVisualizer()
fig1 = visualizer.plot_coordinates(initial_coords, "Initial Coordinates")
visualizer.save_plot(fig1, "initial_coordinates.png")

# Access results
final_coordinates = result['coordinates']
error_metrics = result['error_metrics']
print(f"Output directory: {visualizer.get_output_directory()}")
```

### Enhanced Output Example

```python
from src.output_analyzer import OutputAnalyzer

# Generate enhanced output analysis
analyzer = OutputAnalyzer()
enhanced_distances, enhanced_coordinates = analyzer.generate_enhanced_output(
    original_distances_df, original_coordinates_df, 
    result['coordinates'], distances
)

# Save enhanced output to Excel
analyzer.save_enhanced_output(enhanced_distances, enhanced_coordinates, "enhanced_analysis.xlsx")

# Print summary
analyzer.print_enhanced_summary(enhanced_distances, enhanced_coordinates)
```



### Enhanced Output Features

The enhanced output provides:

**Enhanced Distance Analysis:**
- Original distance measurements
- Optimized distances calculated from final coordinates
- **Distance errors** (absolute difference between measured and optimized)
- Sorted point pairs alphabetically
- Rows sorted by primary and secondary keys

**Enhanced Coordinate Analysis:**
- Original coordinates and fixed point designations
- Optimized x,y coordinates
- Number of distance measurements per point
- RMS error for all measurements involving each point
- Points sorted alphabetically

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd floor_plan
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- `numpy`: Numerical computations
- `scipy`: Optimization algorithms
- `matplotlib`: Visualization
- `pandas`: Data handling and Excel I/O
- `openpyxl`: Excel file reading
- `pytest`: Testing framework

## Testing

Run the test suite:
```bash
pytest tests/
```

## Output Management

The system automatically organizes output files in timestamped directories:

```
output/
├── run_20250802_102744/
│   ├── initial_coordinates.png
│   ├── optimized_coordinates.png
│   ├── coordinate_comparison.png
│   ├── error_analysis.png
│   └── enhanced_analysis.xlsx
└── run_20250802_103015/
    └── ...
```

### Enhanced Output Files

The enhanced analysis generates:
- **Enhanced Distances sheet**: Original + optimized distances + distance errors
- **Enhanced Coordinates sheet**: Original + optimized coordinates + error metrics
- **Visualizations**: Initial, optimized, comparison, and error analysis plots

### Output Management Utilities

```python
from src.output_manager import OutputManager

# List existing runs
manager = OutputManager()
runs = manager.list_runs()
print(f"Found {len(runs)} previous runs")

# Get latest run
latest = manager.get_latest_run()
print(f"Latest run: {latest}")

# Clean up old runs (keep last 7 days)
removed = manager.cleanup_old_runs(keep_days=7)
print(f"Removed {removed} old run directories")
```

## Future Enhancements

- Support for 3D coordinates
- Robust optimization algorithms
- Uncertainty quantification
- Real-time coordinate updates
- Integration with CAD software
- Support for angle measurements
- Multi-floor building support
- Advanced error analysis and outlier detection
- Interactive visualization tools
- Batch processing for multiple datasets

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

[Add your license information here]

## Authors

[Add author information here] 