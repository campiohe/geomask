import pytest
from shapely import Polygon, MultiPolygon
from geomask import GeoMask
from unittest.mock import patch, MagicMock


class TestGeoMaskPlotting:
    @pytest.fixture
    def simple_mask(self):
        poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        return GeoMask(geom=poly, resolution=1.0)

    @pytest.fixture
    def multipolygon_mask(self):
        poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = Polygon([(4, 4), (6, 4), (6, 6), (4, 6)])
        multipoly = MultiPolygon([poly1, poly2])
        return GeoMask(geom=multipoly, resolution=0.8)

    @pytest.fixture
    def polygon_with_hole(self):
        outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
        inner = [(3, 3), (7, 3), (7, 7), (3, 7)]
        poly_with_hole = Polygon(outer, [inner])
        return GeoMask(geom=poly_with_hole, resolution=1.5)

    def test_plot_import_error(self, simple_mask):
        with patch("builtins.__import__") as mock_import:

            def side_effect(name, *args, **kwargs):
                if name == "matplotlib.pyplot":
                    raise ImportError("No module named 'matplotlib'")
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = side_effect

            with pytest.raises(
                ImportError, match="matplotlib is required for plotting"
            ):
                simple_mask.plot()

    def test_plot_basic_functionality(self, simple_mask):
        try:
            import matplotlib.pyplot as plt

            with (
                patch.object(plt, "subplots") as mock_subplots,
                patch.object(plt, "tight_layout") as mock_tight_layout,
            ):
                mock_fig = MagicMock()
                mock_ax = MagicMock()
                mock_subplots.return_value = (mock_fig, mock_ax)

                mock_ax.get_figure.return_value = mock_fig

                result = simple_mask.plot()

                mock_subplots.assert_called_once_with(figsize=(10, 8))

                mock_tight_layout.assert_called_once()

                assert result == (mock_fig, mock_ax)
                assert mock_ax.scatter.called

                mock_ax.set_aspect.assert_called_with("equal")

        except ImportError:
            pytest.skip("matplotlib not available")

    def test_plot_custom_figsize(self, simple_mask):
        try:
            import matplotlib.pyplot as plt

            with (
                patch.object(plt, "subplots") as mock_subplots,
                patch.object(plt, "tight_layout"),
            ):
                mock_fig = MagicMock()
                mock_ax = MagicMock()
                mock_subplots.return_value = (mock_fig, mock_ax)

                simple_mask.plot(figsize=(12, 6))

                mock_subplots.assert_called_once_with(figsize=(12, 6))

        except ImportError:
            pytest.skip("matplotlib not available")

    def test_plot_with_existing_axes(self, simple_mask):
        try:
            import matplotlib.pyplot as plt

            with (
                patch.object(plt, "subplots") as mock_subplots,
                patch.object(plt, "tight_layout"),
            ):
                mock_fig = MagicMock()
                mock_ax = MagicMock()
                mock_ax.get_figure.return_value = mock_fig

                result = simple_mask.plot(ax=mock_ax)

                mock_subplots.assert_not_called()

                mock_ax.get_figure.assert_called_once()

                assert result == (mock_fig, mock_ax)

        except ImportError:
            pytest.skip("matplotlib not available")

    def test_plot_custom_styling(self, simple_mask):
        try:
            import matplotlib.pyplot as plt

            with (
                patch.object(plt, "subplots") as mock_subplots,
                patch.object(plt, "tight_layout"),
            ):
                mock_fig = MagicMock()
                mock_ax = MagicMock()
                mock_subplots.return_value = (mock_fig, mock_ax)

                simple_mask.plot(
                    geometry_color="red",
                    geometry_edgecolor="blue",
                    geometry_alpha=0.5,
                    points_color="green",
                    points_size=50,
                    points_alpha=0.7,
                    title="Custom Title",
                )

                mock_ax.scatter.assert_called()
                call_args = mock_ax.scatter.call_args
                assert call_args[1]["c"] == "green"
                assert call_args[1]["s"] == 50
                assert call_args[1]["alpha"] == 0.7

                mock_ax.set_title.assert_called_with("Custom Title")

        except ImportError:
            pytest.skip("matplotlib not available")

    def test_plot_with_grid(self, simple_mask):
        try:
            import matplotlib.pyplot as plt

            with (
                patch.object(plt, "subplots") as mock_subplots,
                patch.object(plt, "tight_layout"),
            ):
                mock_fig = MagicMock()
                mock_ax = MagicMock()
                mock_subplots.return_value = (mock_fig, mock_ax)

                simple_mask.plot(show_grid=True, grid_color="gray", grid_alpha=0.5)

                mock_ax.grid.assert_called_with(
                    True, color="gray", alpha=0.5, linestyle="-", linewidth=0.5
                )

        except ImportError:
            pytest.skip("matplotlib not available")

    def test_plot_empty_mask(self):
        try:
            import matplotlib.pyplot as plt

            tiny_poly = Polygon([(0, 0), (0.01, 0), (0.01, 0.01), (0, 0.01)])
            mask = GeoMask(geom=tiny_poly, resolution=10.0)

            with (
                patch.object(plt, "subplots") as mock_subplots,
                patch.object(plt, "tight_layout"),
            ):
                mock_fig = MagicMock()
                mock_ax = MagicMock()
                mock_subplots.return_value = (mock_fig, mock_ax)

                result = mask.plot()

                assert result == (mock_fig, mock_ax)

        except ImportError:
            pytest.skip("matplotlib not available")

    def test_plot_multipolygon(self, multipolygon_mask):
        try:
            import matplotlib.pyplot as plt

            with (
                patch.object(plt, "subplots") as mock_subplots,
                patch.object(plt, "tight_layout"),
            ):
                mock_fig = MagicMock()
                mock_ax = MagicMock()
                mock_subplots.return_value = (mock_fig, mock_ax)

                result = multipolygon_mask.plot()

                assert result == (mock_fig, mock_ax)

                assert mock_ax.add_collection.called

        except ImportError:
            pytest.skip("matplotlib not available")

    def test_plot_polygon_with_hole(self, polygon_with_hole):
        try:
            import matplotlib.pyplot as plt

            with (
                patch.object(plt, "subplots") as mock_subplots,
                patch.object(plt, "tight_layout"),
            ):
                mock_fig = MagicMock()
                mock_ax = MagicMock()
                mock_subplots.return_value = (mock_fig, mock_ax)

                result = polygon_with_hole.plot()

                assert result == (mock_fig, mock_ax)

        except ImportError:
            pytest.skip("matplotlib not available")

    def test_plot_auto_title(self, simple_mask):
        try:
            import matplotlib.pyplot as plt

            with (
                patch.object(plt, "subplots") as mock_subplots,
                patch.object(plt, "tight_layout"),
            ):
                mock_fig = MagicMock()
                mock_ax = MagicMock()
                mock_subplots.return_value = (mock_fig, mock_ax)

                simple_mask.plot()

                mock_ax.set_title.assert_called()
                title_call = mock_ax.set_title.call_args[0][0]

                assert "GeoMask" in title_call
                assert "points" in title_call
                assert "resolution=" in title_call

        except ImportError:
            pytest.skip("matplotlib not available")

    def test_plot_axis_labels(self, simple_mask):
        try:
            import matplotlib.pyplot as plt

            with (
                patch.object(plt, "subplots") as mock_subplots,
                patch.object(plt, "tight_layout"),
            ):
                mock_fig = MagicMock()
                mock_ax = MagicMock()
                mock_subplots.return_value = (mock_fig, mock_ax)

                simple_mask.plot()

                mock_ax.set_xlabel.assert_called_with("X")
                mock_ax.set_ylabel.assert_called_with("Y")

        except ImportError:
            pytest.skip("matplotlib not available")

    def test_plot_limits_with_padding(self, simple_mask):
        try:
            import matplotlib.pyplot as plt

            with (
                patch.object(plt, "subplots") as mock_subplots,
                patch.object(plt, "tight_layout"),
            ):
                mock_fig = MagicMock()
                mock_ax = MagicMock()
                mock_subplots.return_value = (mock_fig, mock_ax)

                simple_mask.plot()

                mock_ax.set_xlim.assert_called()
                mock_ax.set_ylim.assert_called()

                # Verify padding was applied (limits should be larger than bounds)
                xlim_call = mock_ax.set_xlim.call_args[0]
                ylim_call = mock_ax.set_ylim.call_args[0]

                bounds = simple_mask.bounds
                assert xlim_call[0] < bounds[0]  # xmin with padding
                assert xlim_call[1] > bounds[2]  # xmax with padding
                assert ylim_call[0] < bounds[1]  # ymin with padding
                assert ylim_call[1] > bounds[3]  # ymax with padding

        except ImportError:
            pytest.skip("matplotlib not available")

    def test_plot_kwargs_passed_to_scatter(self, simple_mask):
        try:
            import matplotlib.pyplot as plt

            with (
                patch.object(plt, "subplots") as mock_subplots,
                patch.object(plt, "tight_layout"),
            ):
                mock_fig = MagicMock()
                mock_ax = MagicMock()
                mock_subplots.return_value = (mock_fig, mock_ax)

                simple_mask.plot(marker="x", linewidth=2)

                call_args = mock_ax.scatter.call_args
                assert "marker" in call_args[1]
                assert call_args[1]["marker"] == "x"
                assert "linewidth" in call_args[1]
                assert call_args[1]["linewidth"] == 2

        except ImportError:
            pytest.skip("matplotlib not available")


class TestGeoMaskPlottingHelpers:
    @pytest.fixture
    def simple_mask(self):
        poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        return GeoMask(geom=poly, resolution=1.0)

    def test_plot_geometry_polygon(self, simple_mask):
        try:
            mock_ax = MagicMock()

            with patch(
                "matplotlib.collections.PatchCollection"
            ) as mock_patch_collection:
                mock_collection = MagicMock()
                mock_patch_collection.return_value = mock_collection

                simple_mask._plot_geometry(
                    mock_ax, color="red", edgecolor="blue", alpha=0.7
                )

                mock_patch_collection.assert_called_once()

                mock_ax.add_collection.assert_called_once_with(mock_collection)

        except ImportError:
            pytest.skip("matplotlib not available")

    def test_plot_geometry_multipolygon(self):
        try:
            import matplotlib.pyplot as plt

            poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
            poly2 = Polygon([(4, 4), (6, 4), (6, 6), (4, 6)])
            multipoly = MultiPolygon([poly1, poly2])
            mask = GeoMask(geom=multipoly, resolution=1.0)

            mock_ax = MagicMock()

            with patch(
                "matplotlib.collections.PatchCollection"
            ) as mock_patch_collection:
                mock_collection = MagicMock()
                mock_patch_collection.return_value = mock_collection

                mask._plot_geometry(mock_ax)

                mock_patch_collection.assert_called_once()
                mock_ax.add_collection.assert_called_once()

        except ImportError:
            pytest.skip("matplotlib not available")

    def test_polygon_to_patch(self, simple_mask):
        try:
            with patch("matplotlib.patches.Polygon") as mock_mpl_polygon:
                mock_patch = MagicMock()
                mock_mpl_polygon.return_value = mock_patch

                result = simple_mask._polygon_to_patch(simple_mask.geom)

                mock_mpl_polygon.assert_called_once()
                call_args = mock_mpl_polygon.call_args

                coords = call_args[0][0]
                assert len(coords) > 0
                assert call_args[1]["closed"] == True

                assert result == mock_patch

        except ImportError:
            pytest.skip("matplotlib not available")

    def test_polygon_to_patch_with_holes(self):
        try:
            outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
            inner = [(3, 3), (7, 3), (7, 7), (3, 7)]
            poly_with_hole = Polygon(outer, [inner])
            mask = GeoMask(geom=poly_with_hole, resolution=1.0)

            with patch("matplotlib.patches.Polygon") as mock_mpl_polygon:
                mock_patch = MagicMock()
                mock_mpl_polygon.return_value = mock_patch

                result = mask._polygon_to_patch(poly_with_hole)

                mock_mpl_polygon.assert_called_once()
                assert result == mock_patch

        except ImportError:
            pytest.skip("matplotlib not available")


if __name__ == "__main__":
    pytest.main([__file__])
