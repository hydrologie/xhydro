"""Functions for processing OI data into formats used by PyGMET."""

import numpy as np
import xarray as xr


def convert_2d_nc_to_1d_stations(path_nc_oi_precip: str, path_nc_grid_temperature: str, outpath: str) -> bool:
    """
    Transform the 2D ERA5Land data into a 1D station-like format.

    Parameters
    ----------
    path_nc_oi_precip : str
        Path to the nc file containing the output of the optimal interpolation results.
    path_nc_grid_temperature : str
        Path to the nc file containing the gridded temperature matching the times and locations of the OI precip
        product.
    outpath : str
        Path to the output file containing the nan-less station-like vector netcdf of ERA5Land + OI grids (used as
        stations in PyGMET).

    Returns
    -------
    bool :
        True if the code ran correctly, error message if not.
    """
    # NetCDF file with ERA5Land precipitation processed with OI (only precip)
    ds = xr.open_dataset(path_nc_oi_precip)

    # OI code will return a percentile, so this is to take care of that particular case.
    if "percentile" in ds:
        ds = ds.sel(percentile=50)

    # Transpose to work with data in correct dimensions.
    ds = ds.transpose("longitude", "latitude", "time")

    # NetCDF file with ERA5Land tmin and tmax.
    dstemp = xr.open_dataset(path_nc_grid_temperature)

    # Get the same times.
    dstemp = dstemp.sel(time=slice(ds["time"].values[0], ds["time"].values[-1]))

    # We want to create a mask of NaN values. Precipitation processed with OI will not have NaNs so we need to use the
    # temperature to find them.
    maskvar = np.empty((len(ds.longitude), len(ds.latitude)))
    maskvar[:] = 1
    temp_c = dstemp["tasmax"].transpose("longitude", "latitude", "time")

    # find the nan values by taking a random day. I don't like taking the 1st so I use 10 here.
    nanmask = temp_c[:, :, 10].isnull()
    maskvar[nanmask] = 0

    # Make the nan mask 3D for the application to the initial dataset of lat x lon x time.
    mask_3d_broadcast = np.broadcast_to(nanmask, [ds["tp"].shape[2], nanmask.shape[0], nanmask.shape[1]])
    mask_3d_broadcast = mask_3d_broadcast.transpose(1, 2, 0)

    # Apply the mask to precipitation. Recall that temperature already has the NaNs.
    ds["tp"].values = ds["tp"].where(~mask_3d_broadcast, np.nan)

    # Stack in station 1D by unrolling the 3D matrices into a 2D (station x time) matrix. Do this for the precip and
    # also the temperature.
    da_stn = ds["tp"].stack(stn=("latitude", "longitude"))
    da_stn_tmax = dstemp["tasmax"].stack(stn=("latitude", "longitude"))
    da_stn_tmin = dstemp["tasmin"].stack(stn=("latitude", "longitude"))

    # Get the latitude and longitude values associated with each station.
    lat_vals = da_stn.indexes["stn"].get_level_values("latitude")
    lon_vals = da_stn.indexes["stn"].get_level_values("longitude")

    # Remove stations that are all NaNs.
    nan_columns = da_stn.isnull().all(dim="time").values
    valid_mask = ~nan_columns
    lat_vals = lat_vals[valid_mask]
    lon_vals = lon_vals[valid_mask]

    da_stn = da_stn.dropna(dim="stn", how="any")
    da_stn_tmax = da_stn_tmax.dropna(dim="stn", how="any")
    da_stn_tmin = da_stn_tmin.dropna(dim="stn", how="any")

    # Rebuild the index of stations starting at 0.
    da_stn = da_stn.assign_coords(stn=np.arange(da_stn.sizes["stn"]))

    # Make the dataset, write to disk.
    ds = xr.Dataset(
        {
            "precip": (("stn", "time"), da_stn.values.transpose()),
            "tmax": (("stn", "time"), da_stn_tmax.values.transpose()),
            "tmin": (("stn", "time"), da_stn_tmin.values.transpose()),
            "latitude": (("stn"), lat_vals),
            "longitude": (("stn"), lon_vals),
        },
        coords={"time": ds.time.values, "stn": np.arange(0, da_stn.sizes["stn"])},
    )

    ds["stnid"] = ds["stn"]
    ds.to_netcdf(outpath)

    return True


def make_target_pygmet_grid(
    ds_in_path: str,
    outpath: str,
):
    """
    Build the netcdf Dataset required by PyGMET to use as the output grid.

    Parameters
    ----------
    ds_in_path : str
        Path to the template dataset to use as the basis for 2D grid.
    outpath : str
        Path to the 2D mask file.

    Returns
    -------
    bool :
        True if the function exits with no error.
    """
    # Open the dataset
    ds_in = xr.open_dataset(ds_in_path)
    lat_grid = ds_in["latitude"].values
    lon_grid = ds_in["longitude"].values

    # Define some fixed parameters
    nparam = 1

    # Number of latitudes and longitudes (i.e. size of grid)
    ny = lat_grid.shape[0]
    nx = lon_grid.shape[0]

    # indices of the coordinates
    param = np.arange(nparam)  # ex. [0]

    # Create the variables
    # step for longitude and latitude
    dx = np.full((nparam,), np.round(ds_in.longitude[1].values - ds_in.longitude[0].values, 5))
    dy = np.full((nparam,), np.round(ds_in.latitude[1].values - ds_in.latitude[0].values, 5))

    # Origins for latitude and longitude (from where the steps begin)
    startx = np.array(np.round(ds_in.longitude.values[0], 5)).reshape(1)  # longitude origin
    starty = np.array(np.round(ds_in.latitude.values[0], 5)).reshape(1)  # latitude origin

    # Rebuild 2D array of lat/lon
    lon2d, lat2d = np.meshgrid(lon_grid, lat_grid)

    # Dimensions (lat, lon)
    elev = np.zeros((ny, nx)) + 1.0  # altitude (m), pygmet can use it as an extra covariate, but we don't.
    mask = np.ones((ny, nx), dtype=np.int32)  # mask of 0/1, but we will force it later.

    # Build the xarray Dataset
    ds = xr.Dataset(
        data_vars={
            "dx": (("param",), dx),
            "dy": (("param",), dy),
            "startx": (("param",), startx),
            "starty": (("param",), starty),
            "elev": (("y", "x"), elev),
            "latitude": (("y", "x"), lat2d),
            "longitude": (("y", "x"), lon2d),
            "mask": (("y", "x"), mask),
        },
        # Add coordinates
        coords={
            "param": param,
            "y": lat_grid,
            "x": lon_grid,
        },
    )

    # Write to disk
    ds.to_netcdf(outpath)
