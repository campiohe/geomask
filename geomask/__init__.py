from dataclasses import dataclass, field
from typing import Self, Union, TYPE_CHECKING
import numpy as np
from shapely import Polygon, MultiPolygon, intersection, Geometry, MultiPoint

if TYPE_CHECKING:
    import pandas as pd

SpatialGeometry = Union[Polygon, MultiPolygon]


@dataclass(slots=True)
class GeoMask:
    """A geometric mask for spatial data processing.

    This class creates a regular grid of points within a given geometry
    and provides methods for spatial masking operations.

    Attributes:
        geom: The input spatial geometry (Polygon or MultiPolygon)
        resolution: Grid resolution for point generation
        offset: Optional offset for grid positioning
        limit: Optional limit on number of points
        mask: The resulting geometric mask after intersection (computed automatically)
    """

    geom: SpatialGeometry
    resolution: float
    offset: tuple[float, float] | None = None
    limit: int | None = None
    mask: Geometry = field(init=False)

    def __post_init__(self) -> None:
        if self.resolution <= 0:
            raise ValueError("Resolution must be positive")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("Limit must be positive")
        if not isinstance(self.geom, (Polygon, MultiPolygon)):
            raise TypeError("Geometry must be a Polygon or MultiPolygon")

        self.mask = self._generate_mask()

    def _generate_mask(self) -> Geometry:
        """Generate the grid mask based on geometry and parameters.

        Returns:
            The geometric mask after intersection with the grid
        """
        actual_resolution = self.resolution
        if self.limit:
            estimated_points = self.geom.area / (self.resolution**2)
            if estimated_points > self.limit:
                actual_resolution = np.sqrt(self.geom.area / self.limit)
                object.__setattr__(self, "resolution", actual_resolution)

        xmin, ymin, xmax, ymax = self.geom.bounds

        grid_bounds = (
            np.floor(xmin / actual_resolution) * actual_resolution,
            np.floor(ymin / actual_resolution) * actual_resolution,
            np.ceil(xmax / actual_resolution) * actual_resolution,
            np.ceil(ymax / actual_resolution) * actual_resolution,
        )

        xcoords, ycoords = np.meshgrid(
            np.arange(grid_bounds[0], grid_bounds[2], actual_resolution),
            np.arange(grid_bounds[1], grid_bounds[3], actual_resolution),
        )

        if self.offset:
            xoffset, yoffset = self.offset
            xcoords += xoffset
            ycoords += yoffset

        mcoords = np.column_stack((xcoords.flatten(), ycoords.flatten()))
        points = MultiPoint(mcoords)

        return intersection(self.geom, points)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.geom.bounds

    @property
    def area(self) -> float:
        return self.geom.area

    @property
    def point_count(self) -> int:
        """Get the number of points in the mask."""
        if self.mask.is_empty:
            return 0
        elif hasattr(self.mask, "geoms"):
            return len([p for p in self.mask.geoms if not p.is_empty])
        return 1 if not self.mask.is_empty else 0

    def to_coordinates(self) -> np.ndarray:
        """Extract coordinates from the mask as a numpy array.

        Returns:
            Array of shape (n, 2) containing x, y coordinates
        """
        if self.mask.is_empty:
            return np.array([]).reshape(0, 2)
        elif hasattr(self.mask, "geoms"):
            coords = []
            for point in self.mask.geoms:
                if not point.is_empty:
                    coords.append([point.x, point.y])
            return np.array(coords) if coords else np.array([]).reshape(0, 2)
        elif hasattr(self.mask, "x") and hasattr(self.mask, "y"):
            return np.array([[self.mask.x, self.mask.y]])
        return np.array([]).reshape(0, 2)

    def to_dataframe(self, x_col: str = "x", y_col: str = "y") -> "pd.DataFrame":
        """Extract coordinates from the mask as a pandas DataFrame.

        Args:
            x_col: Name for the x-coordinate column (default: 'x')
            y_col: Name for the y-coordinate column (default: 'y')

        Returns:
            DataFrame with specified column names containing coordinates

        Raises:
            ImportError: If pandas is not available
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install it with: pip install pandas"
            )

        coords = self.to_coordinates()
        if coords.size == 0:
            return pd.DataFrame(columns=[x_col, y_col])

        return pd.DataFrame(coords, columns=[x_col, y_col])

    def filter_by_geometry(self, filter_geom: SpatialGeometry, **kwargs) -> Self:
        """Create a new GeoMask filtered by another geometry.

        Args:
            filter_geom: Geometry to filter by
            **kwargs: Additional arguments passed to shapely.intersection

        Returns:
            A new filtered GeoMask instance
        """
        filtered_mask = intersection(self.mask, filter_geom, **kwargs)

        new_instance = self.__class__(
            geom=self.geom,
            resolution=self.resolution,
            offset=self.offset,
            limit=self.limit,
        )
        object.__setattr__(new_instance, "mask", filtered_mask)
        return new_instance

    def __len__(self) -> int:
        return self.point_count

    def __bool__(self) -> bool:
        return not self.mask.is_empty

    def __repr__(self) -> str:
        return (
            f"GeoMask(area={self.area:.2f}, resolution={self.resolution:.3f}, "
            f"points={self.point_count}, offset={self.offset}, limit={self.limit})"
        )
