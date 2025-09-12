# GeoMask

A very simple lib for creating geometric masks from spatial data using regular grids.

## Features

- Create regular grids of points inside spacial geometries (polygons or multipolygons)
- Control grid spacing with a resolution parameter
- Optional point limits to avoid huge grids
- Export to pandas DataFrames

## Installation

```bash
pip install geomask
```

## Quick Start

```python
from geomask import GeoMask
from shapely import Polygon

# Create a polygon
polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

# Generate a grid mask
mask = GeoMask(geom=polygon, resolution=1.0)

print(f"Generated {len(mask)} points")
print(f"Area: {mask.area}")
print(f"Bounds: {mask.bounds}")
```

## Core Functionality

### Creating Masks

```python
from geomask import GeoMask
from shapely import Polygon, MultiPolygon

# Basic usage
polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
mask = GeoMask(geom=polygon, resolution=2.0)

# With point limit
mask = GeoMask(geom=polygon, resolution=0.5, limit=100)

# With grid offset
mask = GeoMask(geom=polygon, resolution=1.0, offset=(0.5, 0.5))

# Works with MultiPolygons too
poly1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
poly2 = Polygon([(10, 10), (15, 10), (15, 15), (10, 15)])
multipoly = MultiPolygon([poly1, poly2])
mask = GeoMask(geom=multipoly, resolution=1.0)
```

### Extracting Coordinates

```python
# As numpy array
coords = mask.to_coordinates()
print(f"Shape: {coords.shape}")  # (n_points, 2)

# As pandas DataFrame (requires pandas)
df = mask.to_dataframe()
print(df.head())

# Custom column names
df = mask.to_dataframe(x_col='longitude', y_col='latitude')
df = mask.to_dataframe(x_col='easting', y_col='northing')
```

### Filtering and Analysis

```python
# Filter by another geometry
filter_polygon = Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])
filtered_mask = mask.filter_by_geometry(filter_polygon)

# Properties and methods
print(f"Point count: {len(mask)}")
print(f"Has points: {bool(mask)}")
print(f"Area: {mask.area}")
print(f"Bounds: {mask.bounds}")
```

### Integration with Pandas

```python
import pandas as pd
import numpy as np

# Convert to DataFrame for analysis
df = mask.to_dataframe(x_col='x', y_col='y')

# Add computed columns
df['distance_from_origin'] = np.sqrt(df.x**2 + df.y**2)
df['quadrant'] = df.apply(
    lambda row: ('N' if row.y >= 0 else 'S') + ('E' if row.x >= 0 else 'W'), 
    axis=1
)

# Analysis
print(df.describe())
print(df.quadrant.value_counts())

# Filter points
close_points = df[df.distance_from_origin < 5.0]
```

## Advanced Usage

### Working with Complex Geometries

```python
# Polygon with hole
outer = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
inner = [(3, 3), (7, 3), (7, 7), (3, 7), (3, 3)]
polygon_with_hole = Polygon(outer, [inner])

mask = GeoMask(geom=polygon_with_hole, resolution=1.0)
```

### Performance Optimization

```python
# Use limits for large areas
large_polygon = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])

# This will automatically adjust resolution to meet the limit
mask = GeoMask(geom=large_polygon, resolution=1.0, limit=10000)
print(f"Actual resolution used: {mask.resolution}")
```

## Requirements

- Python 3.11+
- shapely >= 2.1.1
- numpy
- pandas (optional, for DataFrame functionality)

## Development

### Testing

Run the comprehensive test suite:

```bash
# Basic tests
pytest test_geomask.py

# With coverage
pytest test_geomask.py --cov=geomask --cov-report=term-missing

# Verbose output
pytest test_geomask.py -v
```
