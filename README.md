# geo-mask

Is a python-based repository that provides a set of tools for point-clipping and shape-based masking of geospatial datasets.
The core focus is on using `numpy`, `xarray`, and `shapely` to enable efficient spatial data operations.
These tools are useful for geospatial data analysis, allowing clip and mask datasets based on geometric shapes.

## usage
```python
from src.masker import Masker

masked_dataset = Masker.mask(polygon, 0.25)
```
![](docs/result.png)
