from pathlib import Path

import geopandas as gpd
import xarray as xr

from src.masker import Masker

import matplotlib.pyplot as plt

if __name__ == "__main__":
    ds = xr.load_dataset(Path("./data/data.grib2").as_posix(), engine="cfgrib")
    df = gpd.read_file("./data/shape/BR_UF_2022.shp")

    mask_ds = Masker.mask(df.loc[df.SIGLA_UF == "SP"]["geometry"].values[0], 0.25)

    fig, axs = plt.subplots(1, 3, figsize=(30, 10))

    ds.sel(number=1).t2m.plot(ax=axs[0])

    mask_ds.mask.plot(ax=axs[1])

    ds.where(mask_ds.mask, drop=True).sel(number=1).t2m.plot(ax=axs[2])

    plt.savefig("result.png")

    pass
