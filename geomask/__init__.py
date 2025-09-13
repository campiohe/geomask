from dataclasses import dataclass, field
from typing import Self, Union, TYPE_CHECKING
import numpy as np
from shapely import Polygon, MultiPolygon, intersection, Geometry, MultiPoint

try:
    from importlib.metadata import version

    __version__ = version("geomask")
except ImportError:
    try:
        from importlib_metadata import version

        __version__ = version("geomask")
    except ImportError:
        __version__ = "unknown"

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr

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

    def to_xarray(self, x_variable: str = "x", y_variable: str = "y") -> "xr.Dataset":
        """Extract variables from the mask as an xarray Dataset.

        Returns:
            xarray Dataset containing coordinates as variables
        Raises:
            ImportError: If xarray is not available
        """
        try:
            import xarray as xr
        except ImportError:
            raise ImportError(
                "xarray is required for to_xarray(). Install it with: pip install xarray"
            )
        ds = self.to_dataframe(x_variable, y_variable).to_xarray()
        ds.attrs.update(
            {
                "geomask_version": __version__,
                "area": self.area,
                "resolution": self.resolution,
                "point_count": self.point_count,
                "bounds": self.bounds,
                "offset": self.offset,
                "limit": self.limit,
            }
        )
        return ds

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

    def plot(
        self,
        figsize: tuple[float, float] = (10, 8),
        geometry_color: str = "lightblue",
        geometry_edgecolor: str = "black",
        geometry_alpha: float = 0.7,
        points_color: str = "red",
        points_size: float = 20.0,
        points_alpha: float = 0.8,
        show_grid: bool = False,
        grid_color: str = "gray",
        grid_alpha: float = 0.3,
        title: str | None = None,
        ax=None,
        **kwargs,
    ):
        """Plot the geometric mask showing both the geometry and grid points.

        Args:
            figsize: Figure size as (width, height)
            geometry_color: Fill color for the geometry
            geometry_edgecolor: Edge color for the geometry
            geometry_alpha: Transparency for the geometry fill
            points_color: Color for the grid points
            points_size: Size of the grid points
            points_alpha: Transparency for the grid points
            show_grid: Whether to show a grid in the background
            grid_color: Color of the background grid
            grid_alpha: Transparency of the background grid
            title: Plot title (auto-generated if None)
            ax: Optional matplotlib Axes to plot on (creates new if None)
            **kwargs: Additional keyword arguments passed to ax.scatter()

        Returns:
            Matplotlib Figure and Axes objects

        Raises:
            ImportError: If matplotlib is not available
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. Install it with: pip install matplotlib"
            )
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        self._plot_geometry(
            ax, color=geometry_color, edgecolor=geometry_edgecolor, alpha=geometry_alpha
        )

        coords = self.to_coordinates()
        if len(coords) > 0:
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=points_color,
                s=points_size,
                alpha=points_alpha,
                zorder=5,
                **kwargs,
            )

        ax.set_aspect("equal")

        xmin, ymin, xmax, ymax = self.bounds
        padding = max(xmax - xmin, ymax - ymin) * 0.05
        ax.set_xlim(xmin - padding, xmax + padding)
        ax.set_ylim(ymin - padding, ymax + padding)

        if show_grid:
            ax.grid(
                True, color=grid_color, alpha=grid_alpha, linestyle="-", linewidth=0.5
            )

        if title is None:
            title = (
                f"GeoMask: {self.point_count} points (resolution={self.resolution:.3f})"
            )

        ax.set_title(title)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        plt.tight_layout()

        return fig, ax

    def _plot_geometry(self, ax, color="lightblue", edgecolor="black", alpha=0.7):
        from matplotlib.collections import PatchCollection

        patches = []
        if isinstance(self.geom, Polygon):
            patches.append(self._polygon_to_patch(self.geom))
        elif isinstance(self.geom, MultiPolygon):
            for geom in self.geom.geoms:
                patches.append(self._polygon_to_patch(geom))

        if patches:
            collection = PatchCollection(
                patches, facecolor=color, edgecolor=edgecolor, alpha=alpha, zorder=1
            )
            ax.add_collection(collection)

    def _polygon_to_patch(self, polygon):
        from matplotlib.patches import Polygon as MplPolygon

        exterior_coords = list(polygon.exterior.coords)

        patch = MplPolygon(exterior_coords, closed=True)

        if polygon.interiors:
            # For polygons with holes, we need to create a more complex patch
            # This is a simplified approach - for more complex hole handling.
            # consider using matplotlib.path.Path
            pass

        return patch

    def __len__(self) -> int:
        return self.point_count

    def __bool__(self) -> bool:
        return not self.mask.is_empty

    def __repr__(self) -> str:
        return (
            f"GeoMask(area={self.area:.2f}, resolution={self.resolution:.3f}, "
            f"points={self.point_count}, offset={self.offset}, limit={self.limit})"
        )
