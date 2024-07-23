"""Compare results between simulations and observations."""

import logging

import numpy as np
import xarray as xr

import xhydro.optimal_interpolation.utilities as util
from xhydro.modelling.obj_funcs import get_objective_function

logger = logging.getLogger(__name__)

__all__ = ["compare"]


def compare(
    qobs: xr.Dataset,
    qsim: xr.Dataset,
    flow_l1o: xr.Dataset,
    station_correspondence: xr.Dataset,
    observation_stations: list,
    percentile_to_plot: int = 50,
    show_comparison: bool = True,
):
    """Start the computation of the comparison method.

    Parameters
    ----------
    qobs : xr.Dataset
        Streamflow and catchment properties dataset for observed data.
    qsim : xr.Dataset
        Streamflow and catchment properties dataset for simulated data.
    flow_l1o : xr.Dataset
        Streamflow and catchment properties dataset for simulated leave-one-out cross-validation results.
    station_correspondence : xr.Dataset
        Matching between the tag in the simulated files and the observed station number for the obs dataset.
    observation_stations : list
        Observed hydrometric dataset stations to be used in the cross-validation step.
    percentile_to_plot : int
        Percentile value to plot (default is 50).
    show_comparison : bool
        Whether to display the comparison plots (default is True).
    """
    time_range = len(qobs["time"])

    # Read percentiles list (which percentile thresholds were used)
    percentile = flow_l1o["percentile"]

    # Find position of the desired percentile
    idx_pct = np.where(percentile == percentile_to_plot)[0]
    if idx_pct is None:
        raise ValueError(
            "The desired percentile is not computed in the results file \
             provided. Please make sure your percentile value is expressed \
             in percent (i.e. 50th percentile = 50)"
        )

    station_count = len(observation_stations)
    selected_flow_sim = np.empty((time_range, station_count)) * np.nan
    selected_flow_obs = np.empty((time_range, station_count)) * np.nan
    selected_flow_l1o = np.empty((time_range, station_count)) * np.nan

    for i in range(0, station_count):
        msg = f"Reading data from station {i + 1} of {station_count}"
        logger.info(msg)
        # For each validation station:
        cv_station_id = observation_stations[i]

        # Get the station number from the obs database which has the same codification for station ids.
        index_correspondence = np.where(
            station_correspondence["station_id"] == cv_station_id
        )[0][0]
        station_code = station_correspondence["reach_id"][index_correspondence]

        # Search for data in the Qsim file
        index_in_sim = np.where(qsim["station_id"].values == station_code.data)[0]
        sup_sim = qsim["drainage_area"].values[index_in_sim]
        selected_flow_sim[:, i] = (
            qsim["streamflow"].isel(station=index_in_sim) / sup_sim
        )

        # Get data in Qobs file
        index_in_obs = np.where(qobs["station_id"] == cv_station_id)[0]
        sup_obs = qobs["drainage_area"].values[index_in_obs]
        selected_flow_obs[:, i] = (
            qobs["streamflow"].isel(station=index_in_obs) / sup_obs
        )

        # Get data in Leave one out file
        index_in_l1o = np.where(flow_l1o["station_id"] == cv_station_id)[0]
        sup_l1o = qobs["drainage_area"].values[index_in_l1o]
        selected_flow_l1o[:, i] = (
            flow_l1o["streamflow"]
            .isel(station=index_in_l1o, percentile=idx_pct)
            .squeeze()
            / sup_l1o
        )

    # Prepare the arrays for kge results
    kge = np.empty(station_count) * np.nan
    nse = np.empty(station_count) * np.nan
    kge_l1o = np.empty(station_count) * np.nan
    nse_l1o = np.empty(station_count) * np.nan

    for n in range(0, station_count):
        kge[n] = get_objective_function(
            selected_flow_obs[:, n], selected_flow_sim[:, n], "kge"
        )
        nse[n] = get_objective_function(
            selected_flow_obs[:, n], selected_flow_sim[:, n], "nse"
        )
        kge_l1o[n] = get_objective_function(
            selected_flow_obs[:, n], selected_flow_l1o[:, n], "kge"
        )
        nse_l1o[n] = get_objective_function(
            selected_flow_obs[:, n], selected_flow_l1o[:, n], "nse"
        )

    if show_comparison:
        util.plot_results(kge, kge_l1o, nse, nse_l1o)
