import matplotlib.pyplot as plt
from shapely import Polygon
from geomask import GeoMask


def create_test_polygon():
    return Polygon([(0, 0), (10, 0), (10, 8), (0, 8)])


def demonstrate_grid_types():
    poly = create_test_polygon()
    resolution = 1.0

    grid_configs = [
        {"grid_type": "regular_ll", "title": "Regular Lat-Lon Grid"},
        {
            "grid_type": "regular_gg",
            "title": "Regular Gaussian Grid",
            "grid_kwargs": {"n_latitudes": 8},
        },
        {
            "grid_type": "reduced_gg",
            "title": "Reduced Gaussian Grid",
            "grid_kwargs": {"n_latitudes": 8, "reduction_factor": 0.6},
        },
        {
            "grid_type": "rotated_ll",
            "title": "Rotated Grid (30°)",
            "grid_kwargs": {"rotation_angle": 30},
        },
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, config in enumerate(grid_configs):
        try:
            mask = GeoMask(
                geom=poly,
                resolution=resolution,
                grid_type=config["grid_type"],
                grid_kwargs=config.get("grid_kwargs", {}),
            )

            mask.plot(ax=axes[i], title=config["title"])

            print(f"{config['title']}: {len(mask)} points")

        except Exception as e:
            print(f"Error with {config['title']}: {e}")
            axes[i].text(
                0.5,
                0.5,
                f"Error: {str(e)}",
                transform=axes[i].transAxes,
                ha="center",
                va="center",
            )
            axes[i].set_title(config["title"])

    plt.tight_layout()
    plt.savefig("grid_types_demo.png", dpi=150, bbox_inches="tight")
    plt.show()


def demonstrate_projection_grids():
    try:
        import pyproj

        poly = Polygon([(-10, 30), (10, 30), (10, 50), (-10, 50)])
        resolution = 100000  # 100km in meters for projected grids

        projection_configs = [
            {
                "grid_type": "lambert",
                "title": "Lambert Conformal Grid",
                "grid_kwargs": {"central_longitude": 0, "central_latitude": 40},
            },
            {
                "grid_type": "mercator",
                "title": "Mercator Grid",
                "grid_kwargs": {"central_longitude": 0},
            },
            {
                "grid_type": "albers",
                "title": "Albers Equal Area Grid",
                "grid_kwargs": {"central_longitude": 0, "central_latitude": 40},
            },
        ]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, config in enumerate(projection_configs):
            try:
                mask = GeoMask(
                    geom=poly,
                    resolution=resolution,
                    grid_type=config["grid_type"],
                    grid_kwargs=config["grid_kwargs"],
                )

                mask.plot(ax=axes[i], title=config["title"])
                print(f"{config['title']}: {len(mask)} points")

            except Exception as e:
                print(f"Error with {config['title']}: {e}")
                axes[i].text(
                    0.5,
                    0.5,
                    f"Error: {str(e)}",
                    transform=axes[i].transAxes,
                    ha="center",
                    va="center",
                )
                axes[i].set_title(config["title"])

        plt.tight_layout()
        plt.savefig("projection_grids_demo.png", dpi=150, bbox_inches="tight")
        plt.show()

    except ImportError:
        print("pyproj not available. Skipping projection grid demonstration.")


if __name__ == "__main__":
    print("Demonstrating different grid types...")
    print("=" * 50)

    demonstrate_grid_types()

    print("\n" + "=" * 50)
    print("Testing projection grids...")

    demonstrate_projection_grids()

    print("\nDemo complete!")
