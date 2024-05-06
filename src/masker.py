import logging
import numpy as np
import xarray as xr
from shapely import Polygon, MultiPolygon, Point, MultiPoint, intersection

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MaskError(Exception):
    pass


class Masker:
    """
    Create a dataset of points within a specified polygon or multipolygon.

    This class is designed to generate a virtual mesh around a given polygon (or multipolygon)
    with a specified resolution and then calculate which points from the mesh fall inside the polygon.

    The resulting dataset can be used for various purposes, such as clipping globe datasets to
    a specific geographic area.
    """

    @staticmethod
    def mask(poly: Polygon | MultiPolygon, mesh_resolution: float = 0.1) -> xr.Dataset:
        min_x, min_y, max_x, max_y = poly.bounds

        nearest_min_x = np.floor(min_x / mesh_resolution) * mesh_resolution
        nearest_min_y = np.floor(min_y / mesh_resolution) * mesh_resolution
        nearest_max_x = np.ceil(max_x / mesh_resolution) * mesh_resolution
        nearest_max_y = np.ceil(max_y / mesh_resolution) * mesh_resolution

        xx, yy = np.meshgrid(
            np.arange(nearest_min_x, nearest_max_x, mesh_resolution),
            np.arange(nearest_min_y, nearest_max_y, mesh_resolution),
        )

        xx_yy = np.column_stack((xx.flatten(), yy.flatten()))

        mesh_points_poly = intersection(poly, MultiPoint(xx_yy))

        if isinstance(mesh_points_poly, Point):
            if mesh_points_poly.is_empty:
                raise MaskError(f"mesh {mesh_resolution} has no points inside! try a smaller resolution")
            points_in_poly = [(mesh_points_poly.x, mesh_points_poly.y)]
        else:
            points_in_poly = list(map(lambda p: (p.x, p.y), mesh_points_poly.geoms))

        points_in_poly_array = np.array(points_in_poly)

        longitude, latitude = points_in_poly_array[:, 0], points_in_poly_array[:, 1]

        unique_longitude, lon_indices = np.unique(longitude, return_inverse=True)
        unique_latitude, lat_indices = np.unique(latitude, return_inverse=True)

        mask = np.zeros(
            shape=(unique_latitude.size, unique_longitude.size), dtype=np.bool_
        )
        mask[lat_indices, lon_indices] = 1

        return xr.Dataset(
            data_vars={"mask": (("latitude", "longitude"), mask)},
            coords={"latitude": unique_latitude, "longitude": unique_longitude},
        )
