"""Subsampling tools to select stations for PyGMET variability."""

import numpy as np
import xarray as xr


def isel_every_about_n(
    path_to_nc: str,
    dim: str,
    outpath: str,
    rng: float | None,
    threshold: float = 0.1,
    n: int = 10,
    jitter: int = 1,
):
    """
    Select along `dim` with step sizes drawn from {n-jitter, …, n+jitter}. Works for Dataset or DataArray.

    Parameters
    ----------
    path_to_nc : str
        Path to the file containing the dataset of the full data series (lat/long) that we want to subsample.
    dim : str
        Dimension to operate the subsampling on. Example: stn for stations in a vector of stations.
    outpath : str
        Path to the output file to write to disk.
    rng : float
        Initial seed for the random number generator. Using 'None' will make it random.
    threshold : float
        Threshold for the amount of precipitation considered trace, as to not influence the distribution vs machine
        precision of the reanalysis models.
    n : int
        Average interval length to jump from one to the next sampling point.
    jitter : int
        Noise added to the interval to randomize the search pattern and not systematically sample ever n points.

    Returns
    -------
    bool :
        Returns True if the function exited properly.
    """
    # Clean precip (threshold and format)
    ds = clean_precip(path_to_nc, threshold=threshold)

    # Get the random seed.
    rng = np.random.default_rng(rng)

    # Maximum available value to sample.
    k = ds.sizes[dim]

    # Draw steps until we pass the end; then build cumulative indices.
    steps = []
    start = 0

    while True:
        step = n + rng.integers(-jitter, jitter + 1)
        if step <= 0:
            continue  # guard against pathological jitter
        steps.append(step)
        if start + sum(steps) >= k:
            break
    idx = start + np.cumsum([0] + steps[:-1])  # starting index + cumulative steps
    idx = idx[idx < k]

    # Build and save the dataset with the dimension subsampled according to the built index.
    subset = ds.isel({dim: idx})
    subset.to_netcdf(outpath)

    return True


def clean_precip(
    ds_path: str,
    threshold: float = 0.1,
) -> xr.Dataset:
    """
    Convert to float32 and remove any negative precipitation value.

    Parameters
    ----------
    ds_path : str
        Path to the netcdf file containing the precip to load and clean.
    threshold : float
        Threshold for the amount of precipitation considered trace, as to not influence the distribution vs machine
        precision of the reanalysis models.

    Returns
    -------
    xr.Dataset :
        The cleaned precipitation dataset.
    """
    # Load the precipitation dataset.
    ds = xr.open_dataset(ds_path)

    # Convert to float32.
    ds["precip"] = ds["precip"].astype(np.float32)

    # Apply threshold for minimum trace value.
    ds["precip"].values[ds["precip"].values < threshold] = 0.0

    # Return the cleaned dataset
    return ds
