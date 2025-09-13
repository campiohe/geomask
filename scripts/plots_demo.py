import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from shapely import MultiPolygon, Polygon
from geomask import GeoMask

OUTPUT_DIR = Path("plots_demo")
OUTPUT_DIR.mkdir(exist_ok=True)


def save_figure(fig, filename):
    filepath = OUTPUT_DIR / f"{filename}.png"
    fig.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {filepath}")


def demo_basic_plot() -> None:
    print("Creating basic plot...")
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    mask = GeoMask(poly, resolution=1.5)
    fig, ax = mask.plot(title="Basic GeoMask Plot")
    save_figure(fig, "basic_plot")
    plt.close(fig)


def demo_custom_styling() -> None:
    poly = Polygon([(0, 0), (8, 2), (10, 8), (4, 10), (-2, 6)])
    mask = GeoMask(poly, resolution=0.8)
    fig, ax = mask.plot(
        geometry_color="lightgreen",
        geometry_edgecolor="darkgreen",
        geometry_alpha=0.6,
        points_color="purple",
        points_size=30,
        points_alpha=0.9,
        show_grid=True,
        title="Custom Styled GeoMask",
    )
    save_figure(fig, "custom_plot.png")
    plt.close(fig)


def demo_multipolygon() -> None:
    poly1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    poly2 = Polygon([(7, 7), (12, 7), (12, 12), (7, 12)])
    poly3 = Polygon([(2, 8), (4, 8), (3, 10)])
    multipoly = MultiPolygon([poly1, poly2, poly3])
    mask = GeoMask(geom=multipoly, resolution=0.7)
    fig, ax = mask.plot(
        geometry_color="orange",
        points_color="blue",
        points_size=25,
        title="MultiPolygon GeoMask",
    )
    save_figure(fig, "multipolygon_plot")
    plt.close(fig)


def demo_subplot_comparison() -> None:
    poly = Polygon([(0, 0), (6, 1), (8, 4), (5, 7), (1, 6)])
    resolutions = [2.0, 1.0, 0.5]
    sks = [GeoMask(poly, resolution=res) for res in resolutions]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (mask, res) in enumerate(zip(sks, resolutions)):
        mask.plot(
            ax=axes[i], title=f"Resolution: {res} ({len(mask)} points)", points_size=15
        )
    plt.tight_layout()
    save_figure(fig, "comparison_plot")
    plt.close(fig)


def create_star_polygon(center_x, center_y, outer_radius, inner_radius, num_points=5):
    angles = np.linspace(0, 2 * np.pi, num_points * 2, endpoint=False)
    coords = []

    for i, angle in enumerate(angles):
        radius = outer_radius if i % 2 == 0 else inner_radius
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        coords.append((x, y))

    return Polygon(coords)


def create_spiral_polygon(center_x, center_y, max_radius, turns=2, num_points=50):
    angles = np.linspace(0, turns * 2 * np.pi, num_points)
    coords = []

    for angle in angles:
        radius = max_radius * (angle / (turns * 2 * np.pi))
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        coords.append((x, y))

    coords.append((center_x, center_y))
    return Polygon(coords)


def create_heart_polygon(center_x, center_y, size=1):
    t = np.linspace(0, 2 * np.pi, 100)
    x = size * (16 * np.sin(t) ** 3)
    y = size * (13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))

    coords = [(center_x + xi, center_y + yi) for xi, yi in zip(x, y)]
    return Polygon(coords)


def create_irregular_coastline():
    base_points = [
        (0, 0),
        (10, 2),
        (15, -1),
        (25, 3),
        (35, 1),
        (45, 4),
        (50, 0),
        (45, -5),
        (35, -3),
        (25, -6),
        (15, -4),
        (5, -2),
    ]

    np.random.seed(42)
    noisy_points = []
    for x, y in base_points:
        noise_x = np.random.normal(0, 1)
        noise_y = np.random.normal(0, 0.5)
        noisy_points.append((x + noise_x, y + noise_y))

    return Polygon(noisy_points)


def demo_star_polygon():
    star = create_star_polygon(
        center_x=0, center_y=0, outer_radius=8, inner_radius=3, num_points=6
    )
    mask = GeoMask(star, resolution=1.0)

    fig, ax = mask.plot(
        figsize=(10, 10),
        geometry_color="gold",
        geometry_edgecolor="darkorange",
        geometry_alpha=0.7,
        points_color="darkred",
        points_size=25,
        title="Star Polygon GeoMask",
    )
    save_figure(fig, "01_star_polygon")
    plt.close(fig)


def demo_spiral_polygon():
    spiral = create_spiral_polygon(center_x=0, center_y=0, max_radius=10, turns=3)
    mask = GeoMask(spiral, resolution=0.8)

    fig, ax = mask.plot(
        figsize=(10, 10),
        geometry_color="lightsteelblue",
        geometry_edgecolor="navy",
        geometry_alpha=0.6,
        points_color="crimson",
        points_size=20,
        show_grid=True,
        title="Spiral Polygon GeoMask",
    )
    save_figure(fig, "02_spiral_polygon")
    plt.close(fig)


def demo_heart_polygon():
    heart = create_heart_polygon(center_x=0, center_y=0, size=0.5)
    mask = GeoMask(heart, resolution=1.5)

    fig, ax = mask.plot(
        figsize=(10, 10),
        geometry_color="pink",
        geometry_edgecolor="darkred",
        geometry_alpha=0.8,
        points_color="purple",
        points_size=30,
        title="Heart Polygon GeoMask",
    )
    save_figure(fig, "03_heart_polygon")
    plt.close(fig)


def demo_irregular_coastline():
    coastline = create_irregular_coastline()
    mask = GeoMask(coastline, resolution=1.2)

    fig, ax = mask.plot(
        figsize=(12, 8),
        geometry_color="lightblue",
        geometry_edgecolor="darkblue",
        geometry_alpha=0.6,
        points_color="brown",
        points_size=15,
        title="Irregular Coastline GeoMask",
    )
    save_figure(fig, "04_irregular_coastline")
    plt.close(fig)


def demo_complex_multipolygon():
    star1 = create_star_polygon(
        center_x=-15, center_y=10, outer_radius=4, inner_radius=2, num_points=5
    )
    star2 = create_star_polygon(
        center_x=15, center_y=10, outer_radius=3, inner_radius=1.5, num_points=8
    )
    heart = create_heart_polygon(center_x=0, center_y=-5, size=0.3)

    circle_points = [
        (5 * np.cos(angle), -15 + 5 * np.sin(angle))
        for angle in np.linspace(0, 2 * np.pi, 20)
    ]
    circle = Polygon(circle_points)

    triangle = Polygon([(-10, -15), (-5, -10), (-15, -10)])

    complex_multi = MultiPolygon([star1, star2, heart, circle, triangle])
    mask = GeoMask(complex_multi, resolution=0.8)

    fig, ax = mask.plot(
        figsize=(14, 12),
        geometry_color="lightgreen",
        geometry_edgecolor="darkgreen",
        geometry_alpha=0.7,
        points_color="orange",
        points_size=18,
        title="Complex MultiPolygon GeoMask",
    )
    save_figure(fig, "05_complex_multipolygon")
    plt.close(fig)


def demo_archipelago():
    islands = []

    main_island = Polygon(
        [
            (0, 0),
            (15, 3),
            (20, 8),
            (18, 15),
            (12, 18),
            (5, 16),
            (2, 12),
            (-3, 8),
            (-2, 4),
        ]
    )
    islands.append(main_island)

    for center in [(25, 5), (22, 15), (-8, 12), (8, -5), (18, -3)]:
        radius = np.random.uniform(2, 4)
        num_points = np.random.randint(8, 16)
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

        coords = []
        for angle in angles:
            r = radius * np.random.uniform(0.7, 1.3)
            x = center[0] + r * np.cos(angle)
            y = center[1] + r * np.sin(angle)
            coords.append((x, y))
        islands.append(Polygon(coords))

    archipelago = MultiPolygon(islands)
    mask = GeoMask(archipelago, resolution=1.0)

    fig, ax = mask.plot(
        figsize=(14, 10),
        geometry_color="sandybrown",
        geometry_edgecolor="saddlebrown",
        geometry_alpha=0.8,
        points_color="forestgreen",
        points_size=22,
        title="Archipelago MultiPolygon GeoMask",
    )
    save_figure(fig, "06_archipelago")
    plt.close(fig)


def demo_polygon_with_multiple_holes():
    outer = [(-10, -10), (10, -10), (10, 10), (-10, 10)]

    hole1 = [(-6, -6), (-2, -6), (-2, -2), (-6, -2)]

    hole2_center = (4, 4)
    hole2_radius = 2
    hole2 = [
        (
            hole2_center[0] + hole2_radius * np.cos(angle),
            hole2_center[1] + hole2_radius * np.sin(angle),
        )
        for angle in np.linspace(0, 2 * np.pi, 12)
    ]

    hole3_center = (-4, 4)
    hole3 = []
    for i in range(10):
        angle = i * np.pi / 5
        radius = 1.5 if i % 2 == 0 else 0.8
        x = hole3_center[0] + radius * np.cos(angle)
        y = hole3_center[1] + radius * np.sin(angle)
        hole3.append((x, y))

    hole4 = [(2, -4), (6, -4), (4, -1)]

    poly_with_holes = Polygon(outer, [hole1, hole2, hole3, hole4])
    mask = GeoMask(poly_with_holes, resolution=0.5)

    fig, ax = mask.plot(
        figsize=(12, 12),
        geometry_color="lightcyan",
        geometry_edgecolor="teal",
        geometry_alpha=0.7,
        points_color="maroon",
        points_size=15,
        show_grid=True,
        grid_alpha=0.2,
        title="Polygon with Multiple Complex Holes",
    )
    save_figure(fig, "07_multiple_holes")
    plt.close(fig)


def demo_fractal_like_polygon():
    def create_koch_like_side(start, end, depth=2):
        if depth == 0:
            return [start, end]

        dx = (end[0] - start[0]) / 3
        dy = (end[1] - start[1]) / 3

        p1 = start
        p2 = (start[0] + dx, start[1] + dy)
        p4 = (start[0] + 2 * dx, start[1] + 2 * dy)
        p5 = end

        mid_x = (p2[0] + p4[0]) / 2
        mid_y = (p2[1] + p4[1]) / 2
        height = 0.5 * np.sqrt(dx**2 + dy**2)

        perp_x = -dy / np.sqrt(dx**2 + dy**2) * height
        perp_y = dx / np.sqrt(dx**2 + dy**2) * height

        p3 = (mid_x + perp_x, mid_y + perp_y)

        result = []
        for segment in [(p1, p2), (p2, p3), (p3, p4), (p4, p5)]:
            result.extend(create_koch_like_side(segment[0], segment[1], depth - 1)[:-1])
        result.append(p5)
        return result

    base_points = [(0, 0), (10, 0), (10, 10), (0, 10)]

    fractal_points = []
    for i in range(len(base_points)):
        start = base_points[i]
        end = base_points[(i + 1) % len(base_points)]
        side_points = create_koch_like_side(start, end, depth=1)
        fractal_points.extend(side_points[:-1])

    fractal_poly = Polygon(fractal_points)
    mask = GeoMask(fractal_poly, resolution=0.3)

    fig, ax = mask.plot(
        figsize=(12, 12),
        geometry_color="lavender",
        geometry_edgecolor="indigo",
        geometry_alpha=0.8,
        points_color="darkviolet",
        points_size=12,
        title="Fractal-like Polygon GeoMask",
    )
    save_figure(fig, "08_fractal_like")
    plt.close(fig)


def main():
    print("Running all custom polygon plotting demos...\n")
    print(f"Figures will be saved to: {OUTPUT_DIR.absolute()}\n")

    demo_basic_plot()
    demo_custom_styling()
    demo_multipolygon()
    demo_subplot_comparison()
    demo_star_polygon()
    demo_spiral_polygon()
    demo_heart_polygon()
    demo_irregular_coastline()
    demo_complex_multipolygon()
    demo_archipelago()
    demo_polygon_with_multiple_holes()
    demo_fractal_like_polygon()

    print(
        f"\nAll custom polygon demos completed! Check the '{OUTPUT_DIR}' directory for saved figures."
    )


if __name__ == "__main__":
    main()
