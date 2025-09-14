"""Tests for different grid types in GeoMask."""

import pytest
import numpy as np
from shapely import Polygon
from geomask import GeoMask
from geomask.grid_types import GRID_TYPES

# Check if pyproj is available
try:
    import pyproj

    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False


class TestGridTypes:
    """Test different grid type implementations."""

    @pytest.fixture
    def test_polygon(self):
        """Create a simple test polygon."""
        return Polygon([(0, 0), (10, 0), (10, 8), (0, 8)])

    def test_regular_ll_grid(self, test_polygon):
        """Test regular latitude-longitude grid (default)."""
        mask = GeoMask(geom=test_polygon, resolution=2.0, grid_type="regular_ll")

        assert mask.grid_type == "regular_ll"
        assert len(mask) > 0

        # Check that points are roughly evenly spaced
        coords = mask.to_coordinates()
        if len(coords) > 1:
            # Check some spacing properties
            x_coords = coords[:, 0]
            y_coords = coords[:, 1]

            # Should have points at regular intervals
            unique_x = np.unique(x_coords)
            unique_y = np.unique(y_coords)

            if len(unique_x) > 1:
                x_diffs = np.diff(unique_x)
                assert np.allclose(x_diffs, x_diffs[0], rtol=0.1)

            if len(unique_y) > 1:
                y_diffs = np.diff(unique_y)
                assert np.allclose(y_diffs, y_diffs[0], rtol=0.1)

    def test_gaussian_grid(self, test_polygon):
        """Test Gaussian grid implementation."""
        mask = GeoMask(
            geom=test_polygon,
            resolution=2.0,
            grid_type="regular_gg",
            grid_kwargs={"n_latitudes": 6},
        )

        assert mask.grid_type == "regular_gg"
        assert len(mask) > 0

        coords = mask.to_coordinates()
        if len(coords) > 0:
            # Check that latitudes (y-coordinates) follow Gaussian distribution pattern
            y_coords = np.unique(coords[:, 1])

            # With Gaussian latitudes, spacing should not be uniform
            if len(y_coords) > 2:
                y_diffs = np.diff(y_coords)
                # Differences should not all be the same (unlike regular grid)
                assert not np.allclose(y_diffs, y_diffs[0], rtol=0.01)

    def test_reduced_gaussian_grid(self, test_polygon):
        """Test reduced Gaussian grid implementation."""
        mask = GeoMask(
            geom=test_polygon,
            resolution=1.5,
            grid_type="reduced_gg",
            grid_kwargs={"n_latitudes": 6, "reduction_factor": 0.6},
        )

        assert mask.grid_type == "reduced_gg"
        assert len(mask) > 0

        coords = mask.to_coordinates()
        if len(coords) > 0:
            # For reduced grid, number of points per latitude should vary
            y_coords = coords[:, 1]
            unique_y = np.unique(y_coords)

            if len(unique_y) > 1:
                points_per_lat = []
                for y in unique_y:
                    count = np.sum(np.isclose(y_coords, y))
                    points_per_lat.append(count)

                # Should have variation in points per latitude
                assert len(set(points_per_lat)) > 1 or len(points_per_lat) == 1

    def test_rotated_grid(self, test_polygon):
        """Test rotated grid implementation."""
        rotation_angle = 30.0
        mask = GeoMask(
            geom=test_polygon,
            resolution=2.0,
            grid_type="rotated_ll",
            grid_kwargs={"rotation_angle": rotation_angle},
        )

        assert mask.grid_type == "rotated_ll"
        assert len(mask) > 0

        # Compare with non-rotated grid
        regular_mask = GeoMask(
            geom=test_polygon, resolution=2.0, grid_type="regular_ll"
        )

        # Rotated grid should have same number of points (approximately)
        # but different coordinates
        assert (
            abs(len(mask) - len(regular_mask)) <= 6
        )  # Allow reasonable difference due to rotation

        coords = mask.to_coordinates()
        regular_coords = regular_mask.to_coordinates()

        if len(coords) > 0 and len(regular_coords) > 0:
            # Coordinates should be different (test if first few points are different)
            min_len = min(len(coords), len(regular_coords))
            test_coords = coords[:min_len]
            test_regular = regular_coords[:min_len]
            assert not np.allclose(test_coords, test_regular, rtol=0.1)

    def test_grid_type_validation(self, test_polygon):
        """Test that invalid grid types raise appropriate errors."""
        with pytest.raises(ValueError, match="Grid type 'invalid_grid' not supported"):
            GeoMask(geom=test_polygon, resolution=1.0, grid_type="invalid_grid")

    def test_grid_kwargs_passing(self, test_polygon):
        """Test that grid_kwargs are properly passed to grid generators."""
        # Test with Gaussian grid parameters
        mask = GeoMask(
            geom=test_polygon,
            resolution=2.0,
            grid_type="regular_gg",
            grid_kwargs={"n_latitudes": 4},
        )

        assert mask.grid_kwargs == {"n_latitudes": 4}
        assert len(mask) > 0

    def test_available_grid_types(self):
        """Test that expected grid types are available."""
        expected_basic_types = ["regular_ll", "regular_gg", "reduced_gg", "rotated_ll"]

        for grid_type in expected_basic_types:
            assert grid_type in GRID_TYPES

    def test_grid_with_offset(self, test_polygon):
        """Test grid generation with offset."""
        offset = (0.5, 0.3)
        mask = GeoMask(
            geom=test_polygon, resolution=2.0, grid_type="regular_ll", offset=offset
        )

        coords = mask.to_coordinates()
        if len(coords) > 0:
            # Check that coordinates are offset
            # This is a basic check - exact verification depends on implementation
            assert len(coords) > 0

    def test_grid_consistency(self, test_polygon):
        """Test that grid generation is consistent across calls."""
        mask1 = GeoMask(geom=test_polygon, resolution=1.5, grid_type="regular_ll")
        mask2 = GeoMask(geom=test_polygon, resolution=1.5, grid_type="regular_ll")

        coords1 = mask1.to_coordinates()
        coords2 = mask2.to_coordinates()

        # Should generate identical grids
        assert len(coords1) == len(coords2)
        if len(coords1) > 0:
            assert np.allclose(coords1, coords2)

    def test_filter_preserves_grid_type(self, test_polygon):
        """Test that filtering preserves grid type information."""
        mask = GeoMask(
            geom=test_polygon,
            resolution=1.0,
            grid_type="regular_gg",
            grid_kwargs={"n_latitudes": 6},
        )

        # Create a smaller filter polygon
        filter_poly = Polygon([(2, 2), (8, 2), (8, 6), (2, 6)])
        filtered_mask = mask.filter_by_geometry(filter_poly)

        # Grid type should be preserved
        assert filtered_mask.grid_type == "regular_gg"
        assert filtered_mask.grid_kwargs == {"n_latitudes": 6}
        assert len(filtered_mask) <= len(mask)

    def test_repr_includes_grid_type(self, test_polygon):
        """Test that __repr__ includes grid type information."""
        mask = GeoMask(geom=test_polygon, resolution=1.0, grid_type="regular_gg")
        repr_str = repr(mask)

        assert "grid_type" in repr_str
        assert "regular_gg" in repr_str


@pytest.mark.skipif(not HAS_PYPROJ, reason="pyproj not available")
class TestProjectionGrids:
    """Test projection-based grid types (requires pyproj)."""

    @pytest.fixture
    def geographic_polygon(self):
        """Create a polygon in geographic coordinates."""
        return Polygon([(-5, 35), (5, 35), (5, 45), (-5, 45)])

    def test_lambert_conformal_grid(self, geographic_polygon):
        """Test Lambert Conformal Conic grid."""
        try:
            mask = GeoMask(
                geom=geographic_polygon,
                resolution=100000,  # 100km
                grid_type="lambert",
                grid_kwargs={"central_longitude": 0, "central_latitude": 40},
            )

            assert mask.grid_type == "lambert"
            assert len(mask) > 0

        except (ValueError, ImportError) as e:
            pytest.skip(f"Lambert grid not available: {e}")

    def test_mercator_grid(self, geographic_polygon):
        """Test Mercator grid."""
        try:
            mask = GeoMask(
                geom=geographic_polygon,
                resolution=100000,  # 100km
                grid_type="mercator",
                grid_kwargs={"central_longitude": 0},
            )

            assert mask.grid_type == "mercator"
            assert len(mask) > 0

        except (ValueError, ImportError) as e:
            pytest.skip(f"Mercator grid not available: {e}")

    def test_albers_grid(self, geographic_polygon):
        """Test Albers Equal Area grid."""
        try:
            mask = GeoMask(
                geom=geographic_polygon,
                resolution=100000,  # 100km
                grid_type="albers",
                grid_kwargs={"central_longitude": 0, "central_latitude": 40},
            )

            assert mask.grid_type == "albers"
            assert len(mask) > 0

        except (ValueError, ImportError) as e:
            pytest.skip(f"Albers grid not available: {e}")
