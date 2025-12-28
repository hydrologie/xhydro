"""Evaluate cross-validation of optimal interpolation results."""

import cartopy
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def plot_cross_validation_rmse_results(filename_stations):
    """
    Generate a plot of the RMSE improvements by performing the optimal interpolation.

    Parameters
    ----------
    filename_stations : str
        The path and name of the file that contains the stations (observations) data. Must be a netcdf file.
    """
    # Get the list of validation stations from the run we just did.
    validation_stations_precip = np.loadtxt("./validation_stations.txt").astype(np.int32)
    all_stations = xr.open_dataset(filename_stations)

    # Get the latitude and longitude of all possible stations
    all_latitudes = all_stations.latitude.values
    all_longitudes = all_stations.longitude.values

    # Get the latitude and longitude of the stations used in validation
    latitudes = all_latitudes[validation_stations_precip]
    longitudes = all_longitudes[validation_stations_precip]

    # Load the RMSE errors calculated for the raw ERA5Land and the OI processed at the validation station locations.
    validation_rmse_values_era5 = np.loadtxt("./Test_validation_rmse_values_ERA5.txt")
    validation_rmse_values_oi = np.loadtxt("./Test_validation_rmse_values_OI.txt")

    # Make the figure showing the changes in RMSE
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, subplot_kw={"projection": ccrs.PlateCarree()})

    # Plot the raw results
    plt.subplot(311)
    sc1 = plt.scatter(x=longitudes, y=latitudes, c=validation_rmse_values_era5, s=5)
    plt.title("Original")
    plt.colorbar(sc1)
    sc1.set_cmap("plasma")
    ax1.set_aspect("equal", adjustable=None)
    ax1.add_feature(cartopy.feature.BORDERS, linestyle="-", alpha=1)
    ax1.add_feature(cartopy.feature.OCEAN, zorder=100, edgecolor="k")
    ax1.coastlines()
    ax1.gridlines(draw_labels=True)

    # Plot the OI results
    plt.subplot(312)
    sc2 = plt.scatter(x=longitudes, y=latitudes, c=validation_rmse_values_era5, s=3)
    plt.title("Corrected")
    plt.colorbar(sc2)
    sc2.set_cmap("plasma")
    ax2.set_aspect("equal", adjustable=None)
    ax2.add_feature(cartopy.feature.BORDERS, linestyle="-", alpha=1)
    ax2.add_feature(cartopy.feature.OCEAN, zorder=100, edgecolor="k")
    ax2.coastlines()
    ax2.gridlines(draw_labels=True)

    # Calculate and plot the differences between both (delta RMSE).
    diffs = validation_rmse_values_era5 - validation_rmse_values_oi
    col_norm = TwoSlopeNorm(vmin=np.nanmin(diffs), vcenter=0, vmax=np.nanmax(diffs))
    plt.subplot(313)
    sc3 = plt.scatter(x=longitudes, y=latitudes, c=diffs, s=12, norm=col_norm, cmap="RdBu_r", edgecolors="k", linewidths=0.5)
    plt.title("Difference (Original RMSE - OI RMSE): Higher is better ")
    plt.colorbar(sc3)
    ax3.set_aspect("equal", adjustable=None)
    ax3.add_feature(cartopy.feature.BORDERS, linestyle="-", alpha=1)
    ax3.add_feature(cartopy.feature.OCEAN, zorder=100, edgecolor="k")
    ax3.coastlines()
    ax3.gridlines(draw_labels=True)

    # Show the figure
    plt.show()


def rmse_ignore_nan(y_true, y_pred):
    """
    Compute the Root Mean Squared Error (RMSE) between two arrays while ignoring any positions containing NaN values.

    Parameters
    ----------
    y_true : array_like
        Ground truth (correct) target values.
    y_pred : array_like
        Estimated target values.

    Returns
    -------
    float
        The RMSE calculated only over positions where both y_true and y_pred are finite (non-NaN).
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Create a mask of valid (non-NaN) indices
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)

    # Filter out NaN values from both arrays
    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]

    # If no valid values remain, return NaN
    if y_true_valid.size == 0:
        return np.nan

    # Compute Mean Squared Error on valid values
    mse = np.mean((y_true_valid - y_pred_valid) ** 2)

    # Return the square root of MSE (RMSE)
    return np.sqrt(mse)


def evaluate_cross_validation(
    observed_precip_valid: np.array,
    estimated_precip_reshaped: np.array,
    precip_medians: np.array,
    lat_obs_valid: list[float],
    lon_obs_valid: list[float],
    lat_est: list[float],
    lon_est: list[float],
):
    """
    Calculate and plot the evaluation of gains in performance using optimal interpolation compared to the raw data.

    Parameters
    ----------
    observed_precip_valid : np.array
        Observed precipitation for the validation set, used to compute evaluation metrics for cross-validation.
    estimated_precip_reshaped : np.array
        Estimated precipitation in a reshaped matrix such that it is compatible with the interpolator requirements.
    precip_medians : np.array
        Estimated precipitation at all locations of the background field.
    lat_obs_valid : array-like
        Latitude of the observation stations for the validation dataset.
    lon_obs_valid : array-like
        Longitude of the observation stations for the validation dataset.
    lat_est : array-like
        Latitude of the estimated gridpoints (complete grid).
    lon_est : array-like
        Longitude of the estimated gridpoints (complete grid).
    """
    # Prepare the variables to store the precipitation and RMSE values and fill with NaN.
    precip_table_stations = np.empty((observed_precip_valid.shape[1], observed_precip_valid.shape[0]))
    precip_table_era5 = np.empty((observed_precip_valid.shape[1], observed_precip_valid.shape[0]))
    precip_table_oi = np.empty((observed_precip_valid.shape[1], observed_precip_valid.shape[0]))
    precip_table_stations[:] = np.nan
    precip_table_era5[:] = np.nan
    precip_table_oi[:] = np.nan

    # Preallocate the RMSE tables
    rmse_values_era5 = np.empty(len(lon_obs_valid))
    rmse_values_oi = np.empty(len(lon_obs_valid))

    # For each station, extract the nearest ERA5Land station, which should overlap exactly if data lat/lon are rounded.
    # Get the precipitation values
    for station in range(0, len(lon_obs_valid)):
        pos = np.argwhere((lon_est == lon_obs_valid[station]) & (lat_est == lat_obs_valid[station]))
        pos = pos[0][0]
        precip_table_era5[:, station] = estimated_precip_reshaped[pos, :]
        precip_table_stations[:, station] = observed_precip_valid[station, :]
        precip_table_oi[:, station] = precip_medians[pos, :]

    # Again, for each station, now compute the RMSE values for the raw ERA5Land set and then the OI set.
    for station in range(0, len(lon_obs_valid)):
        rmse_values_era5[station] = rmse_ignore_nan(precip_table_stations[:, station], precip_table_era5[:, station])
        rmse_values_oi[station] = rmse_ignore_nan(precip_table_stations[:, station], precip_table_oi[:, station])

    # Prepare plots
    plt.plot(rmse_values_era5, "g")
    plt.plot(rmse_values_oi, "r")
    plt.plot(rmse_values_era5 - rmse_values_oi, "b")
    plt.axhline(y=0.0, color="k", linestyle="-")
    plt.legend(("Without OI", "With OI", "Difference (higher means OI is better)"))
    plt.title("Precipitation OI results in validation")
    plt.ylabel("RMSE [mm]")
    plt.show()

    # Compute means of RMSE values
    rmse_original = np.nanmean(rmse_values_era5)
    rmse_oi = np.nanmean(rmse_values_oi)

    # Save variables to textfiles for future needs or inspection.
    np.savetxt("Test_validation_stations_precip.txt", precip_table_stations)
    np.savetxt("Test_validation_ERA5_precip.txt", precip_table_era5)
    np.savetxt("Test_validation_OI_precip.txt", precip_table_oi)
    np.savetxt("Test_validation_rmse_values_ERA5.txt", rmse_values_era5)
    np.savetxt("Test_validation_rmse_values_OI.txt", rmse_values_oi)
    np.savetxt("Test_validation_rmse_difference_OI_ERA5.txt", rmse_values_oi - rmse_values_era5)
    np.savetxt("Test_validation_rmse_mean_values_Original_vs_OI.txt", np.array((rmse_original, rmse_oi)))

    print("Original RMSE is " + str(rmse_original) + " and the OI RMSE is " + str(rmse_oi) + ".")
