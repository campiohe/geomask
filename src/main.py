import numpy as np
from pathlib import Path
import xarray as xr

if __name__ == "__main__":
    ds = xr.load_dataset(
        Path("./data/data.grib2").as_posix(),
        engine="cfgrib",
        # filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 2, "shortName": "2t"},
    )

    lat_lon_array = np.array(np.meshgrid(ds.latitude, ds.longitude)).T.reshape(-1, 2)

    with open("./data/mesh.bln", "w") as f:
        f.writelines([f"{lat},{lon}\n" for lat, lon in lat_lon_array])
