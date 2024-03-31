# geo-mask

Is a python-based repository that provides a set of tools for point-clipping and shape-based masking of geospatial datasets.
The core focus is on using `numpy`, `xarray`, and `shapely` to enable efficient spatial data operations.
These tools are useful for geospatial data analysis, allowing clip and mask datasets based on geometric shapes.

## Features
- Clipping datasets based on a single polygon.
- Clipping datasets with multiple shapes.

## Usage
```python
from src.masker import Masker

# Clip the DataArray with the polygon
masked_dataset = Masker.mask(polygon, mesh_resolution=0.25)

clipped_dataset = dataset.where(masked_dataset.mask, drop=True).plot()
```
![](docs/result.png)

## License
Geo-Mask is licensed under the MIT License. See the `LICENSE` file for more information.
