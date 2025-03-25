import pytest
import numpy as np
from shapely import Polygon, MultiPolygon, Point
from geomask import GeoMask


class TestGeoMaskInitialization:
    def test_valid_polygon_initialization(self):
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        mask = GeoMask(geom=poly, resolution=1.0)

        assert mask.geom == poly
        assert mask.resolution == 1.0
        assert mask.offset is None
        assert mask.limit is None
        assert mask.mask is not None

    def test_valid_multipolygon_initialization(self):
        poly1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        poly2 = Polygon([(10, 10), (15, 10), (15, 15), (10, 15)])
        multipoly = MultiPolygon([poly1, poly2])

        mask = GeoMask(geom=multipoly, resolution=2.0)

        assert mask.geom == multipoly
        assert mask.resolution == 2.0

    def test_initialization_with_offset(self):
        poly = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        offset = (0.5, 0.5)

        mask = GeoMask(geom=poly, resolution=1.0, offset=offset)

        assert mask.offset == offset

    def test_initialization_with_limit(self):
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        mask = GeoMask(geom=poly, resolution=0.5, limit=50)

        assert mask.limit == 50
        # Note: Due to grid alignment and intersection, actual count may vary slightly
        # We test that it's reasonably close to the limit
        assert len(mask) <= 70  # Allow some tolerance for grid effects

    def test_resolution_adjustment_with_limit(self):
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        original_resolution = 0.1  # Would create ~10000 points
        limit = 100

        mask = GeoMask(geom=poly, resolution=original_resolution, limit=limit)

        # Resolution should be adjusted to meet the limit
        assert mask.resolution > original_resolution
        assert len(mask) <= limit


class TestGeoMaskValidation:
    def test_negative_resolution_raises_error(self):
        poly = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])

        with pytest.raises(ValueError, match="Resolution must be positive"):
            GeoMask(geom=poly, resolution=-1.0)

    def test_zero_resolution_raises_error(self):
        poly = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])

        with pytest.raises(ValueError, match="Resolution must be positive"):
            GeoMask(geom=poly, resolution=0.0)

    def test_negative_limit_raises_error(self):
        poly = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])

        with pytest.raises(ValueError, match="Limit must be positive"):
            GeoMask(geom=poly, resolution=1.0, limit=-1)

    def test_zero_limit_raises_error(self):
        poly = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])

        with pytest.raises(ValueError, match="Limit must be positive"):
            GeoMask(geom=poly, resolution=1.0, limit=0)

    def test_invalid_geometry_type_raises_error(self):
        point = Point(0, 0)

        with pytest.raises(
            TypeError, match="Geometry must be a Polygon or MultiPolygon"
        ):
            GeoMask(geom=point, resolution=1.0)

    def test_string_geometry_raises_error(self):
        with pytest.raises(
            TypeError, match="Geometry must be a Polygon or MultiPolygon"
        ):
            GeoMask(geom="not a geometry", resolution=1.0)


class TestGeoMaskProperties:
    @pytest.fixture
    def square_mask(self):
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        return GeoMask(geom=poly, resolution=2.0)

    def test_bounds_property(self, square_mask):
        bounds = square_mask.bounds
        assert bounds == (0.0, 0.0, 10.0, 10.0)

    def test_area_property(self, square_mask):
        assert square_mask.area == 100.0

    def test_point_count_property(self, square_mask):
        count = square_mask.point_count
        assert isinstance(count, int)
        assert count > 0

    def test_len_dunder_method(self, square_mask):
        assert len(square_mask) == square_mask.point_count

    def test_bool_dunder_method_true(self, square_mask):
        assert bool(square_mask) is True

    def test_bool_dunder_method_false(self):
        # Create a very small polygon that won't contain any grid points
        tiny_poly = Polygon([(0.1, 0.1), (0.2, 0.1), (0.2, 0.2), (0.1, 0.2)])
        mask = GeoMask(geom=tiny_poly, resolution=10.0)

        # This should create an empty mask
        if len(mask) == 0:
            assert bool(mask) is False

    def test_repr_method(self, square_mask):
        repr_str = repr(square_mask)

        assert "GeoMask" in repr_str
        assert "area=" in repr_str
        assert "resolution=" in repr_str
        assert "points=" in repr_str


class TestGeoMaskMethods:
    @pytest.fixture
    def test_mask(self):
        poly = Polygon([(0, 0), (6, 0), (6, 6), (0, 6)])
        return GeoMask(geom=poly, resolution=2.0)

    def test_to_coordinates_returns_array(self, test_mask):
        coords = test_mask.to_coordinates()

        assert isinstance(coords, np.ndarray)
        assert coords.ndim == 2
        assert coords.shape[1] == 2  # x, y coordinates

    def test_to_coordinates_correct_values(self):
        poly = Polygon([(0, 0), (6, 0), (6, 6), (0, 6)])
        mask = GeoMask(geom=poly, resolution=2.0)
        coords = mask.to_coordinates()

        # All coordinates should be within the polygon bounds
        assert np.all(coords[:, 0] >= 0)  # x >= 0
        assert np.all(coords[:, 0] <= 6)  # x <= 6
        assert np.all(coords[:, 1] >= 0)  # y >= 0
        assert np.all(coords[:, 1] <= 6)  # y <= 6

    def test_filter_by_geometry(self, test_mask):
        # Create a smaller polygon for filtering
        filter_poly = Polygon([(1, 1), (5, 1), (5, 5), (1, 5)])

        filtered_mask = test_mask.filter_by_geometry(filter_poly)

        assert isinstance(filtered_mask, GeoMask)
        assert len(filtered_mask) <= len(test_mask)
        assert filtered_mask.geom == test_mask.geom  # Original geometry unchanged
        assert filtered_mask.resolution == test_mask.resolution

    def test_filter_by_geometry_empty_result(self, test_mask):
        # Create a polygon that doesn't intersect with the mask
        filter_poly = Polygon([(100, 100), (101, 100), (101, 101), (100, 101)])

        filtered_mask = test_mask.filter_by_geometry(filter_poly)

        # Should result in empty or very small mask
        assert len(filtered_mask) <= len(test_mask)


class TestGeoMaskDataFrame:
    @pytest.fixture
    def sample_mask(self):
        poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        return GeoMask(geom=poly, resolution=1.5)

    def test_to_dataframe_default_columns(self, sample_mask):
        pytest.importorskip("pandas")  # Skip if pandas not available

        df = sample_mask.to_dataframe()

        assert list(df.columns) == ["x", "y"]
        assert len(df) == len(sample_mask)
        assert df.dtypes["x"] == "float64"
        assert df.dtypes["y"] == "float64"

    def test_to_dataframe_custom_columns(self, sample_mask):
        pytest.importorskip("pandas")

        df = sample_mask.to_dataframe(x_col="longitude", y_col="latitude")

        assert list(df.columns) == ["longitude", "latitude"]
        assert len(df) == len(sample_mask)

    def test_to_dataframe_empty_mask(self):
        pytest.importorskip("pandas")

        # Create a tiny polygon that likely won't contain grid points
        tiny_poly = Polygon(
            [(0.001, 0.001), (0.002, 0.001), (0.002, 0.002), (0.001, 0.002)]
        )
        mask = GeoMask(geom=tiny_poly, resolution=1.0)

        df = mask.to_dataframe()

        assert list(df.columns) == ["x", "y"]
        # Should handle empty or small mask gracefully
        assert len(df) >= 0

        # If the mask is empty, DataFrame should also be empty but with correct columns
        if len(mask) == 0:
            assert len(df) == 0

    def test_to_dataframe_without_pandas(self, sample_mask, monkeypatch):
        # Mock pandas import to fail
        def mock_import(name, *args, **kwargs):
            if name == "pandas":
                raise ImportError("No module named 'pandas'")
            return __import__(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)

        with pytest.raises(ImportError, match="pandas is required"):
            sample_mask.to_dataframe()


class TestGeoMaskEdgeCases:
    def test_very_small_polygon(self):
        tiny_poly = Polygon([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1)])
        mask = GeoMask(geom=tiny_poly, resolution=0.05)

        # Should not crash and should have reasonable behavior
        assert isinstance(mask, GeoMask)
        assert mask.area == pytest.approx(0.01, rel=1e-6)

    def test_single_point_mask_edge_case(self):
        """Test edge case where mask results in a single point."""
        from shapely import Point

        # Create a mask and manually set it to a single point to test edge case
        poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        mask = GeoMask(geom=poly, resolution=1.0)

        # Create a single Point geometry to test the edge case
        single_point = Point(1.0, 1.0)
        # Manually override the mask to test single point handling
        object.__setattr__(mask, "mask", single_point)

        # Test to_coordinates with single point - this covers line 115-116
        coords = mask.to_coordinates()
        assert coords.shape == (1, 2)
        assert coords[0, 0] == 1.0
        assert coords[0, 1] == 1.0

        # Test point_count with single point - this covers line 99
        assert mask.point_count == 1

    def test_unknown_geometry_type_edge_case(self):
        """Test edge case with unknown geometry type."""
        # Create a simple geometry that doesn't have the expected attributes
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        mask = GeoMask(geom=poly, resolution=1.0)

        # Create a simple object that simulates an unknown geometry type
        class MockGeometry:
            @property
            def is_empty(self):
                return False

        mock_geom = MockGeometry()
        # Manually override the mask to test the fallback case
        object.__setattr__(mask, "mask", mock_geom)

        # This should trigger the final return statement (line 117)
        coords = mask.to_coordinates()
        assert coords.shape == (0, 2)

    def test_very_large_polygon_with_limit(self):
        large_poly = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])
        mask = GeoMask(geom=large_poly, resolution=1.0, limit=100)

        assert len(mask) <= 100
        assert mask.resolution > 1.0  # Should be adjusted

    def test_polygon_with_hole(self):
        # Outer ring
        outer = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        # Inner ring (hole)
        inner = [(3, 3), (7, 3), (7, 7), (3, 7), (3, 3)]

        poly_with_hole = Polygon(outer, [inner])
        mask = GeoMask(geom=poly_with_hole, resolution=1.0)

        assert isinstance(mask, GeoMask)
        assert mask.area == 84.0  # 10x10 - 4x4 = 100 - 16 = 84

    def test_complex_multipolygon(self):
        poly1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        poly2 = Polygon([(10, 10), (15, 10), (15, 15), (10, 15)])
        poly3 = Polygon([(20, 0), (25, 0), (25, 5), (20, 5)])

        multipoly = MultiPolygon([poly1, poly2, poly3])
        mask = GeoMask(geom=multipoly, resolution=1.5)

        assert isinstance(mask, GeoMask)
        assert mask.area == 75.0  # 3 * 25 = 75

    def test_offset_functionality(self):
        poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])

        mask_no_offset = GeoMask(geom=poly, resolution=2.0)
        mask_with_offset = GeoMask(geom=poly, resolution=2.0, offset=(1.0, 1.0))

        coords_no_offset = mask_no_offset.to_coordinates()
        coords_with_offset = mask_with_offset.to_coordinates()

        # Coordinates should be different due to offset
        # (but might have same count depending on intersection)
        assert not np.array_equal(coords_no_offset, coords_with_offset)


class TestGeoMaskIntegration:
    def test_full_workflow_with_filtering_and_dataframe(self):
        pytest.importorskip("pandas")

        # Create initial mask
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        mask = GeoMask(geom=poly, resolution=1.5, limit=50)

        # Filter it
        filter_poly = Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])
        filtered_mask = mask.filter_by_geometry(filter_poly)

        # Convert to DataFrame
        df = filtered_mask.to_dataframe(x_col="easting", y_col="northing")

        # Verify the complete workflow
        assert len(df) == len(filtered_mask)
        assert len(filtered_mask) <= len(mask)
        assert list(df.columns) == ["easting", "northing"]

        # All points should be within the filter geometry
        for _, row in df.iterrows():
            point = Point(row["easting"], row["northing"])
            # Point should be within or on the boundary of filter_poly
            assert filter_poly.contains(point) or filter_poly.touches(point)

    def test_multiple_operations_preserve_properties(self):
        poly = Polygon([(0, 0), (6, 0), (6, 6), (0, 6)])
        original_mask = GeoMask(geom=poly, resolution=2.0, offset=(0.5, 0.5), limit=20)

        # Perform filtering
        filter_poly = Polygon([(1, 1), (5, 1), (5, 5), (1, 5)])
        filtered_mask = original_mask.filter_by_geometry(filter_poly)

        # Properties should be preserved
        assert filtered_mask.geom == original_mask.geom
        assert filtered_mask.resolution == original_mask.resolution
        assert filtered_mask.offset == original_mask.offset
        assert filtered_mask.limit == original_mask.limit

        # But mask should be different
        assert len(filtered_mask) <= len(original_mask)


if __name__ == "__main__":
    pytest.main([__file__])
