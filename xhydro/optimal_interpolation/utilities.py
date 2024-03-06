"""Utilities required for managing data in the interpolation toolbox."""

import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

__all__ = ["plot_results",
           "general_ecf",
           "write_netcdf_flow_percentiles",
           ]


def plot_results(kge, kge_l1o, nse, nse_l1o):
    """Generate a plot of the results of model evaluation using various metrics.

    Parameters
    ----------
    kge : array-like
        Kling-Gupta Efficiency for the entire dataset.
    kge_l1o : array-like
        Kling-Gupta Efficiency for leave-one-out cross-validation.
    nse : array-like
        Nash-Sutcliffe Efficiency for the entire dataset.
    nse_l1o : array-like
        Nash-Sutcliffe Efficiency for leave-one-out cross-validation.

    Returns
    -------
    None :
        No return.
    """
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.scatter(kge, kge_l1o)
    ax1.set_xlabel("KGE")
    ax1.set_ylabel("KGE Leave-one-out OI")
    ax1.axline((0, 0), (1, 1), linewidth=2)
    ax1.set_xlim(0.3, 1)
    ax1.set_ylim(0.3, 1)

    ax2.scatter(nse, nse_l1o)
    ax2.set_xlabel("NSE")
    ax2.set_ylabel("NSE Leave-one-out OI")
    ax2.axline((0, 0), (1, 1), linewidth=2)
    ax2.set_xlim(0.3, 1)
    ax2.set_ylim(0.3, 1)

    plt.show()


def general_ecf(h, par, form):
    """Define the form of the Error Covariance Function (ECF) equations.

    Parameters
    ----------
    h : float or array
        The distance or distances at which to evaluate the ECF.
    par : list
        Parameters for the ECF equation.
    form : int
        The form of the ECF equation to use (1, 2, or other).

    Returns
    -------
    float or array:
        The calculated ECF values based on the specified form.
    """
    if form == 1:
        return par[0] * (1 + h / par[1]) * np.exp(-h / par[1])
    elif form == 2:
        return par[0] * np.exp(-0.5 * np.power(h / par[1], 2))
    else:
        return par[0] * np.exp(-h / par[1])


def write_netcdf_flow_percentiles(
    write_file, station_id, lon, lat, drain_area, time, percentile, discharge
):
    """Write discharge data to a NetCDF file.

    Parameters
    ----------
    write_file : str
        Name of the NetCDF file to be created.
    station_id : array-like
        List of station IDs.
    lon : array-like
        List of longitudes corresponding to each station.
    lat : array-like
        List of latitudes corresponding to each station.
    drain_area : array-like
        List of drainage areas corresponding to each station.
    time : array-like
        List of datetime objects representing time.
    percentile : list or None
        List of percentiles or None if not applicable.
    discharge : numpy.ndarray
        3D array of discharge data, dimensions (percentile, station, time).

    Returns
    -------
    None :
        No return.

    Notes
    -----
    - The function creates a NetCDF file using the provided data and saves it with the specified filename.
    - If the file already exists, it is deleted before creating a new one.
    - The function includes appropriate metadata and attributes for each variable.
    """
    if os.path.exists(write_file):
        os.remove(write_file)

    # Create dataset
    ds = xr.Dataset()
    discharge = np.array(discharge)
    axis_time = np.where(np.array(discharge.shape) == len(time))
    axis_stations = np.where(np.array(discharge.shape) == len(drain_area))

    # Prepare discharge data
    if percentile:
        axis_percentile = np.where(np.array(discharge.shape) == len(percentile))
        ds["streamflow"] = (
            ["percentile", "station_id", "time"],
            np.transpose(
                discharge, (axis_percentile[0][0], axis_stations[0][0], axis_time[0][0])
            ),
        )
        ds["percentile"] = ("percentile", percentile)
    else:
        ds["streamflow"] = (
            ["station_id", "time"],
            np.transpose(discharge, (axis_stations[0][0], axis_time[0][0])),
        )

    # Other variables
    ds["time"] = ("time", time)
    ds["lat"] = ("station_id", lat)
    ds["lon"] = ("station_id", lon)
    ds["drainage_area"] = ("station_id", drain_area)
    ds["station_id"] = ("station_id", station_id)

    # Time bounds
    ta = np.array(time)
    time_bnds = np.array([ta - 1, time]).T
    ds["time_bnds"] = (("time", "nbnds"), time_bnds)

    # Set attributes
    ds["time"].attrs = {
        "long_name": "time",
        "standard_name": "time",
        "axis": "T",
        "bounds": "time_bnds",
    }
    ds["streamflow"].attrs = {
        "long_name": "discharge",
        "standard_name": "discharge",
        "units": "m3/s",
        "cell_methods": "time: mean",
        "coverage_content_type": "modelResult",
        "coordinates": "time station_id",
    }
    ds["lat"].attrs = {
        "long_name": "latitude_of_river_stretch_outlet",
        "standard_name": "latitude",
        "units": "degrees_north",
        "axis": "Y",
    }
    ds["lon"].attrs = {
        "long_name": "longitude_of_river_stretch_outlet",
        "standard_name": "longitude",
        "units": "degrees_east",
        "axis": "X",
    }

    ds["drainage_area"].attrs = {
        "long_name": "drainage_area_at_river_stretch_outlet",
        "standard_name": "drainage_area",
        "units": "km2",
        "coverage_content_type": "auxiliaryInformation",
        "coordinates": "lat lon station_id",
    }
    ds["station_id"].attrs = {"long_name": "Station ID", "cf_role": "timeseries_id"}

    # Write to file
    ds.to_netcdf(write_file)
