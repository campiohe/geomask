import numpy as np
import xarray as xr
from shapely import Polygon, MultiPolygon, Point, MultiPoint, intersection


class MaskError(Exception):
    pass


def create_virtual_meshgrid(bounds: tuple, mesh_resolution: float) -> list[np.ndarray]:
    min_x, min_y, max_x, max_y = bounds

    nearest_min_x = np.floor(min_x / mesh_resolution) * mesh_resolution
    nearest_min_y = np.floor(min_y / mesh_resolution) * mesh_resolution
    nearest_max_x = np.ceil(max_x / mesh_resolution) * mesh_resolution
    nearest_max_y = np.ceil(max_y / mesh_resolution) * mesh_resolution

    return np.meshgrid(
        np.arange(nearest_min_x, nearest_max_x, mesh_resolution),
        np.arange(nearest_min_y, nearest_max_y, mesh_resolution),
    )


def crop_mesh_by_polygon(
    mesh: list[np.ndarray], poly: Polygon | MultiPolygon
) -> np.ndarray:
    xx, yy = mesh
    xx_yy = np.column_stack((xx.flatten(), yy.flatten()))

    mesh_points_poly = intersection(poly, MultiPoint(xx_yy))

    if isinstance(mesh_points_poly, Point):
        if mesh_points_poly.is_empty:
            points_in_poly = []
        else:
            points_in_poly = [(mesh_points_poly.x, mesh_points_poly.y)]
    else:
        points_in_poly = list(map(lambda p: (p.x, p.y), mesh_points_poly.geoms))

    return np.array(points_in_poly)


def create_mask_dataset(
    poly: Polygon | MultiPolygon,
    mesh_resolution: float = 0.1,
    x_coord_name: str = "x",
    y_coord_name: str = "y",
) -> xr.Dataset:
    """
    Create a dataset of points within a specified polygon or multipolygon.

    This class is designed to generate a virtual mesh around a given polygon (or multipolygon)
    with a specified resolution and then calculate which points from the mesh fall inside the polygon.

    The resulting dataset can be used for various purposes, such as clipping globe datasets to
    a specific geographic area.
    """
    meshgrid = create_virtual_meshgrid(poly.bounds, mesh_resolution)

    meshpoints = crop_mesh_by_polygon(meshgrid, poly)

    if not bool(meshpoints.size):
        raise MaskError

    unique_x, x_indices = np.unique(meshpoints[:, 0], return_inverse=True)
    unique_y, y_indices = np.unique(meshpoints[:, 1], return_inverse=True)

    mask = np.zeros(shape=(unique_y.size, unique_x.size), dtype=np.bool_)
    mask[y_indices, x_indices] = True

    return xr.Dataset(
        data_vars={"mask": ((y_coord_name, x_coord_name), mask)},
        coords={y_coord_name: unique_y, x_coord_name: unique_x},
    )
