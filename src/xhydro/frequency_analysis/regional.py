"""
Provides functions for performing regional frequency analysis on hydrological data.

Key Functions:
- cluster_indices: Get indices of elements with a specific cluster number.
- get_groups_indices: Retrieve indices of groups from a clustering result.
- get_group_from_fit: Obtain indices of groups from a fitted model.
- fit_pca: Perform Principal Component Analysis (PCA) on the input dataset.
- moment_l_vector: Calculate L-moments for multiple datasets.
- moment_l: Compute various L-moments and L-moment ratios for a given dataset.

The module utilizes NumPy, pandas, scikit-learn, and xarray for efficient data manipulation and analysis.

Example Usage:
    import xarray as xr
    from regional_frequency_analysis import fit_pca, moment_l_vector

    # Load your dataset
    ds = xr.open_dataset('your_data.nc')

    # Perform PCA
    data_pca, pca_obj = fit_pca(ds, n_components=3)

    # Calculate L-moments
    l_moments = moment_l_vector(ds.values)

This module is designed for hydrologists and data scientists working with regional frequency analysis in water resources.
"""

import datetime
import math
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from xhydro import __version__
from xhydro.utils import update_history


def cluster_indices(clust_num: int, labels_array: np.ndarray) -> np.ndarray:
    """
    Get the indices of elements with a specific cluster number using NumPy.

    Parameters
    ----------
    clust_num : int
        Cluster number to find indices for.
    labels_array : numpy.ndarray
        Array containing cluster labels.

    Returns
    -------
    numpy.ndarray
        Indices of elements with the specified cluster number.
    """
    return np.where(labels_array == clust_num)[0]


def get_groups_indices(cluster: list, sample: xr.Dataset) -> list:
    """
    Get indices of groups from a clustering result, excluding the group labeled -1.

    Parameters
    ----------
    cluster : list
        Clustering result with labels attribute.
    sample : xr.Dataset
        Data sample to fit the model.

    Returns
    -------
    list
        List of indices for each non-excluded group.
    """
    return [
        sample.index.to_numpy()[cluster_indices(i, cluster.labels_)]
        for i in range(np.max(cluster.labels_) + 1)
    ]


def get_group_from_fit(model: object, param: dict, sample: xr.Dataset) -> list:
    """
    Get indices of groups from a fit using the specified model and parameters.

    Parameters
    ----------
    model : obj
        Model class or instance with a fit method.
    param : dict
        Parameters for the model.
    sample : xr.Dataset
        Data sample to fit the model.

    Returns
    -------
    list :
        List of indices for each non-excluded group.
    """
    sample = (
        sample.to_dataframe(name="value")
        .reset_index()
        .pivot(index="Station", columns="components")
    )
    return get_groups_indices(model(**param).fit(sample), sample)


def fit_pca(ds: xr.Dataset, **kwargs: dict) -> tuple:
    r"""
    Perform Principal Component Analysis (PCA) on the input dataset.

    This function scales the input data, applies PCA transformation, and returns
    the transformed data along with the PCA object.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to perform PCA on.
    \*\*kwargs : dict
        Additional keyword arguments to pass to the PCA constructor.

    Returns
    -------
    tuple: A tuple containing:
        - data_pca (xarray.DataArray): PCA-transformed data with 'Station' and 'components' as coordinates.
        - obj_pca (sklearn.decomposition.PCA): Fitted PCA object.

    Notes
    -----
    - The input data is scaled before PCA is applied.
    - The number of components in the output depends on the n_components parameter passed to PCA.
    """
    ds = _scale_data(ds)
    df = ds.to_dataframe()
    pca = PCA(**kwargs)
    obj_pca = pca.fit(df)
    data_pca = pca.transform(df)
    data_pca = xr.DataArray(
        data_pca,
        coords={
            "Station": ds.coords["Station"].values,
            "components": list(range(pca.n_components_)),
        },
    )

    data_pca.attrs["long_name"] = "Fitted Scaled Data"
    data_pca.attrs["description"] = (
        "Fitted scaled data with StandardScaler and PCA from sklearn.preprocessing and sklearn.decomposition"
    )
    data_pca.attrs["fitted_variables"] = [v for v in ds.var()]

    return data_pca, obj_pca


def _scale_data(ds: xr.Dataset) -> xr.Dataset:
    scalar = StandardScaler()
    df = ds.to_dataframe()

    scaled_data = pd.DataFrame(scalar.fit_transform(df))  # scaling the data
    scaled_data.columns = (
        df.columns
    )  # Sets columns name and index from original dataframe to scaled dataframe
    scaled_data.index = df.index
    return xr.Dataset(scaled_data)


def _moment_l_vector(x_vec: np.array) -> list:
    return [_moment_l(x[~np.isnan(x)]) for x in x_vec]


# L-moments calculation
def _moment_l(x: np.array) -> list:
    """
    Calculate L-moments for a given dataset.

    This function computes various L-moments and L-moment ratios for a given array of data.
    It can return the results either as a list or as an OrderedDict.

    Parameters
    ----------
    x : list or array-like
        Input data for which to calculate L-moments.

    Returns
    -------
    list :
        Returns List in the order: [lambda1, lambda2, lambda3, tau, tau3, tau4]

    L-moments calculated:
    - lambda1: First L-moment (location)
    - lambda2: Second L-moment (scale)
    - lambda3: Third L-moment
    - tau: L-CV (Coefficient of L-Variation)
    - tau3: L-CS (L-skewness)
    - tau4: L-CK (L-kurtosis)

    Note:
    - NaN values are removed from the input data before calculations.
    - The input data is sorted in descending order for the calculations.
    """
    x = x[~np.isnan(x)]

    # Sorting the data in descending order
    x_sort = np.sort(x)
    x_sort = x_sort[::-1]

    n = np.size(x_sort)
    j = np.arange(1, n + 1)

    # ponderation of moments
    b0 = np.mean(x_sort)
    b1 = np.dot((n - j) / (n * (n - 1)), x_sort)
    b2 = np.dot((n - j) * (n - j - 1) / (n * (n - 1) * (n - 2)), x_sort)
    b3 = np.dot(
        (n - j) * (n - j - 1) * (n - j - 2) / (n * (n - 1) * (n - 2) * (n - 3)), x_sort
    )

    # L-Moment
    lambda1 = b0
    lambda2 = 2 * b1 - b0
    lambda3 = 6 * b2 - 6 * b1 + b0
    lambda4 = 20 * b3 - 30 * b2 + 12 * b1 - b0

    tau = lambda2 / lambda1  # L_CV # tau2 dans Anctil, tau dans Hosking et al.
    tau3 = lambda3 / lambda2  # L_CS
    tau4 = lambda4 / lambda2  # L_CK

    return [lambda1, lambda2, lambda3, tau, tau3, tau4]


def calc_h_z(
    ds_groups: xr.Dataset,
    ds_moments_groups: xr.Dataset,
    kap: object,
    seed: int | None = None,
) -> xr.Dataset:
    """
    Calculate heterogeneity measure H and Z-score for regional frequency analysis.

    Parameters
    ----------
    ds_groups : xarray.Dataset
        Dataset containing grouped data.
    ds_moments_groups : xarray.Dataset
        Dataset containing L-moments for grouped data.
    kap : scipy.stats.kappa3
        Kappa3 distribution object.
    seed : int, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    xarray.Dataset
        Dataset containing calculated H values and Z-scores for each group.

    Notes
    -----
    This function applies the heterogeneity measure and Z-score calculations
    to grouped data for regional frequency analysis. It uses L-moments and
    the Kappa3 distribution in the process.
    This function does not support lazy evaluation.
    Equations are based on Hosking, J. R. M., & Wallis, J. R. (1997). Regional frequency analysis (p. 240).
    """
    tau = ds_moments_groups.sel(lmom="tau").load()
    tau3 = ds_moments_groups.sel(lmom="tau3").load()
    tau4 = ds_moments_groups.sel(lmom="tau4").load()
    longeur = ds_groups.copy().count(dim="time").load()

    station_dim = ds_groups.cf.cf_roles["timeseries_id"][0]

    ds_h, ds_b4, ds_sigma4, ds_tau4_r = xr.apply_ufunc(
        _heterogeneite_et_score_z,
        kap,
        longeur,
        tau,
        tau3,
        tau4,
        seed,
        input_core_dims=[
            [],
            [station_dim],
            [station_dim],
            [station_dim],
            [station_dim],
            [],
        ],
        output_core_dims=[[], [], [], []],
        vectorize=True,
    )
    ds_tau4 = _calculate_gev_tau4(ds_groups.load(), ds_moments_groups.load())

    z_score = (ds_tau4 - ds_tau4_r + ds_b4) / ds_sigma4

    z_score = _append_ds_vars_names(z_score, "_Z")

    ds_h = _append_ds_vars_names(ds_h, "_H")
    ds = _combine_h_z(xr.merge([z_score, ds_h]))
    ds["crit"].attrs[
        "description"
    ] = f"H and Z score based on Hosking, J. R. M., & Wallis, J. R. (1997). Regional frequency analysis (p. 240). - xhydro version: {__version__}"
    ds["crit"].attrs["long_name"] = "Score"
    for v in ds.var():
        ds[v].attrs[
            "description"
        ] = f"H and Z score based on Hosking, J. R. M., & Wallis, J. R. (1997). Regional frequency analysis (p. 240). - xhydro version: {__version__}"
    return ds


def _calculate_gev_tau4(
    ds_groups: xr.Dataset, ds_moments_groups: xr.Dataset
) -> xr.Dataset:
    # H&W
    lambda_r_1, lambda_r_2, lambda_r_3 = _calc_lambda_r(ds_groups, ds_moments_groups)

    kappa = _calc_kappa(lambda_r_2, lambda_r_3)

    # Hosking et Wallis, eq. A53
    tau4 = (5 * (1 - 4**-kappa) - 10 * (1 - 3**-kappa) + 6 * (1 - 2**-kappa)) / (
        1 - 2**-kappa
    )
    return tau4


def _heterogeneite_et_score_z(
    kap: object, n: np.array, t: np.array, t3: np.array, t4: np.array, seed=None
) -> tuple:

    # We remove nan or 0 length
    # If not enough values to calulculate some moments, other moments are removed as well
    bool_maks = (n != 0) & (~np.isnan(t)) & (~np.isnan(t3)) & (~np.isnan(t4))
    n = n[bool_maks]
    t = t[bool_maks]
    t3 = t3[bool_maks]
    t4 = t4[bool_maks]

    # Hosking et Wallis, eq. 4.3
    tau_r = np.dot(n, t) / np.sum(n)
    tau3_r = np.dot(n, t3) / np.sum(n)
    tau4_r = np.dot(n, t4) / np.sum(n)

    # L-CVs ponderated standard deviation for the sample (eq. 4.4)
    v = np.sqrt(np.dot(n, (t - tau_r) ** 2) / np.sum(n))

    # Kappa distribution
    # Fit a kappa distribution to the region average L-moment ratios:
    try:
        kappa_param = kap.lmom_fit(lmom_ratios=[1, tau_r, tau3_r, tau4_r])
    except ValueError as error:
        warnings.warn(
            f"Kappa distribution fit blablabla (quelle serait la cause d'un ValueError?), returning all NaNs. Error: {error}."
        )
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )  # Returning nans for H, b4, sigma4, tau4_r
    except Exception as error:
        warnings.warn(
            f"Kappa distribution fit failed to converge, returning all NaNs. Error: {error}."
        )
        if "Failed to converge" in repr(error):
            return (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )  # Returning nans for H, b4, sigma4, tau4_r
        else:
            raise error
    n_sim = 500  # Number of "virtual regions" simulated

    def _calc_tsim(kappa_param: dict, length: float, n_sim: int) -> np.array:

        # For each station, we get n_sim vectors de same lenght than the observations
        rvs = kap.rvs(
            kappa_param["k"],
            kappa_param["h"],
            kappa_param["loc"],
            kappa_param["scale"],
            size=(n_sim, int(length)),
            random_state=seed,
        )

        return _momentl_optim(rvs)

    t_sim_tau4m = [_calc_tsim(kappa_param, length, n_sim) for length in n]

    t_sim = np.array(
        [tt[3] for tt in t_sim_tau4m]
    )  # Tau corresponds to the 3rd moment.
    tau4m = np.array([tt[5] for tt in t_sim_tau4m])  # Tau4 corresponds to the 5th term.

    b4 = np.mean(tau4m - tau4_r)

    sigma4 = np.sqrt(
        (1 / (n_sim - 1)) * (np.sum((tau4m - tau4_r) ** 2) - (n_sim * b4 * b4))
    )

    # Calculating V
    tau_r_sim = np.dot(n, t_sim) / np.sum(n)

    v_sim = np.sqrt(np.dot(n, (t_sim - tau_r_sim) ** 2) / np.sum(n))

    mu_v = np.mean(v_sim)
    sigma_v = np.std(v_sim)

    h = (v - mu_v) / sigma_v

    return h, b4, sigma4, tau4_r


# Calculating L-moments
def _momentl_optim(x: np.array) -> list:
    if x.ndim == 1:
        x = x[~np.isnan(x)]
        # reverse sorting
        x_sort = np.sort(x)
        x_sort = x_sort[::-1]
        n = np.size(x_sort)
        j = np.arange(1, n + 1)

        b0 = np.mean(x_sort)
        b1 = np.dot((n - j) / (n * (n - 1)), x_sort)
        b2 = np.dot((n - j) * (n - j - 1) / (n * (n - 1) * (n - 2)), x_sort)
        b3 = np.dot(
            (n - j) * (n - j - 1) * (n - j - 2) / (n * (n - 1) * (n - 2) * (n - 3)),
            x_sort,
        )
    elif x.ndim == 2:
        x.sort()
        x_sort = np.fliplr(x)

        nn = np.shape(x_sort)
        j = np.repeat([np.arange(1, nn[1] + 1)], nn[0], axis=0)
        n = nn[1]
        b0 = np.mean(x_sort, axis=1)
        b1 = np.dot((n - j) / (n * (n - 1)), x_sort.T)[0]
        b2 = np.dot((n - j) * (n - j - 1) / (n * (n - 1) * (n - 2)), x_sort.T)[0]
        b3 = np.dot(
            (n - j) * (n - j - 1) * (n - j - 2) / (n * (n - 1) * (n - 2) * (n - 3)),
            x_sort.T,
        )[0]
    else:
        raise NotImplementedError("Only 1d and 2d have been implemented")

    # Moment L
    lambda1 = b0
    lambda2 = 2 * b1 - b0
    lambda3 = 6 * b2 - 6 * b1 + b0
    lambda4 = 20 * b3 - 30 * b2 + 12 * b1 - b0

    tau = lambda2 / lambda1  # L_CV # tau2 in Anctil, tau dans Hosking et al.
    tau3 = lambda3 / lambda2  # L_CS
    tau4 = lambda4 / lambda2  # L_CK

    return [lambda1, lambda2, lambda3, tau, tau3, tau4]


def _append_ds_vars_names(ds: xr.Dataset, suffix: str) -> xr.Dataset:
    for name in ds.data_vars:
        ds = ds.rename({name: name + suffix})
    return ds


def mask_h_z(
    ds: xr.Dataset, thresh_h: float | None = 1, thresh_z: float | None = 1.64
) -> xr.DataArray:
    """
    Create a boolean mask based on heterogeneity measure H and Z-score thresholds.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing H and Z values for each group.
    thresh_h : float, optional
        Threshold for the heterogeneity measure H. Default is 1.
    thresh_z : float, optional
        Threshold for the absolute Z-score. Default is 1.64.

    Returns
    -------
    xarray.DataArray
        Boolean mask where True indicates groups that meet both threshold criteria.
    """
    ds_out = (ds.sel(crit="H") < thresh_h) & (abs(ds.sel(crit="Z")) < thresh_z)
    for v in ds_out.var():
        ds_out[v].attrs["H_threshold"] = thresh_h
        ds_out[v].attrs["Z_threshold"] = thresh_z
        ds_out[v].attrs["long_name"] = "Mask"
        ds_out[v].attrs["description"] = "Mask for regions based on H & Z thresholds"
        ds_out[v].attrs["history"] = update_history(
            f"Mask for regions based on H ({thresh_h}) & Z ({thresh_z}) thresholds",
            ds_out[v],
        )
    return ds_out


def _combine_h_z(ds: xr.Dataset) -> xr.Dataset:
    new_ds = xr.Dataset()
    for v in ds:
        if "_Z" in v:
            new_ds[v.removesuffix("_Z")] = xr.concat(
                [ds[v.replace("_Z", "_H")], ds[v]], dim="crit"
            ).assign_coords(crit=["H", "Z"])
    return new_ds


def calculate_rp_from_afr(
    ds_groups: xr.Dataset,
    ds_moments_groups: xr.Dataset,
    rp: np.array,
    l1: xr.DataArray | None = None,
) -> xr.DataArray:
    """
    Calculate return periods from Annual Flow Regime (AFR) analysis.

    Parameters
    ----------
    ds_groups : xarray.Dataset
        Dataset containing grouped flow data.
    ds_moments_groups : xarray.Dataset
        Dataset containing L-moments for grouped data.
    rp : array-like
        Return periods to calculate.
    l1 : xarray.DataArray, optional
        First L-moment (location) values. L-moment can be specified for ungauged catchments.
        If None, values are taken from ds_moments_groups.

    Returns
    -------
    xarray.DataArray
        Calculated return periods for each group and specified return period.

    Notes
    -----
    This function calculates return periods using the Annual Flow Regime method.
    If l1 is not provided, it uses the first L-moment from ds_moments_groups.
    The function internally calls calculate_ic_from_AFR to compute the flood index.
    Equations are based on Hosking, J. R. M., & Wallis, J. R. (1997). Regional frequency analysis (p. 240).
    """
    if l1 is None:
        station_dim = ds_moments_groups.cf.cf_roles["timeseries_id"][0]
        l1 = ds_moments_groups.sel(lmom="l1").dropna(dim=station_dim, how="all")
    ds = _calculate_ic_from_afr(ds_groups, ds_moments_groups, rp) * l1
    for v in ds.var():
        ds[v].attrs["long_name"] = "Return period"
        ds[v].attrs[
            "description"
        ] = "Calculated return periods for each group and specified return period."
        ds[v].attrs["history"] = update_history("Computed return periods", ds[v])
        ds[v].attrs["units"] = ds_groups[v].attrs["units"]
    return ds


def _calculate_ic_from_afr(
    ds_groups: xr.Dataset, ds_moments_groups: xr.Dataset, rp: list
) -> xr.Dataset:

    lambda_r_1, lambda_r_2, lambda_r_3 = _calc_lambda_r(ds_groups, ds_moments_groups)

    # alpha = location
    # xi    = scale
    # kappa = shape

    kappa = _calc_kappa(lambda_r_2, lambda_r_3)

    term = xr.apply_ufunc(_calc_gamma, (1 + kappa), vectorize=True)

    # Hosking et Wallis, eq. A56. et Anctil et al. 1998, eq. 7 et 8.
    alpha = (lambda_r_2 * kappa) / ((1 - (2**-kappa)) * term)
    xi = lambda_r_1 + (alpha * (term - 1)) / kappa

    # Calculating wanted return periods
    t = xr.DataArray(data=rp, dims="return_period").assign_coords(return_period=rp)

    # Hosking et Wallis, eq. A44 et Anctil et al. 1998, eq. 5.
    q_rt = xi + alpha * (1 - (-np.log((t - 1) / t)) ** kappa) / kappa

    return q_rt


def _calc_gamma(val):
    return math.gamma(val)


def remove_small_regions(ds: xr.Dataset, thresh: int = 5) -> xr.Dataset:
    """
    Remove regions from the dataset that have fewer than the threshold number of stations.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset containing regions and stations.
    thresh : int, optional
        The minimum number of stations required for a region to be kept. Default is 5.

    Returns
    -------
    xarray.Dataset
        The dataset with small regions removed.
    """
    station_dim = ds.cf.cf_roles["timeseries_id"][0]
    for gr in ds.group_id:
        if (
            len(ds.sel(group_id=gr).dropna(dim=station_dim, how="all")[station_dim])
            < thresh
        ):
            ds = ds.drop_sel(group_id=gr)
    return ds


def _calc_kappa(lambda_r_2, lambda_r_3):

    # Hosking et Wallis, éq. A55
    c = (2 / (3 + (lambda_r_3 / lambda_r_2))) - (np.log(2) / np.log(3))

    # Hosking et Wallis, éq. A55 (generally acceptable approximation)
    kappa = 7.8590 * c + 2.9554 * (c**2)
    return kappa


def _calc_lambda_r(ds_groups: xr.Dataset, ds_moments_groups: xr.Dataset) -> xr.Dataset:
    station_dim = ds_groups.cf.cf_roles["timeseries_id"][0]

    nr = ds_moments_groups.count(dim=station_dim).isel(lmom=0)
    nk = ds_groups.dropna(dim=station_dim, how="all").count(dim="time")
    wk = (nk * nr) / (nk + nr)

    lambda_k0 = ds_moments_groups.sel(lmom="l1").dropna(dim=station_dim, how="all")
    lambda_k1 = ds_moments_groups.sel(lmom="l2").dropna(dim=station_dim, how="all")
    lambda_k2 = ds_moments_groups.sel(lmom="l3").dropna(dim=station_dim, how="all")

    l2 = (lambda_k1 / lambda_k0) * wk
    l3 = (lambda_k2 / lambda_k0) * wk

    lambda_r_1 = 1  # Anctil et al. 1998 (p.362)
    lambda_r_2 = l2.sum(dim=station_dim) / wk.sum(dim=station_dim)
    lambda_r_3 = l3.sum(dim=station_dim) / wk.sum(dim=station_dim)
    return lambda_r_1, lambda_r_2, lambda_r_3


def calc_moments(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculate L-moments for multiple stations.

    Parameters
    ----------
    ds : xarray.Dataset
        A vector of stations, where each element is an array-like object
        containing the data for which to calculate L-moments.

    Returns
    -------
    xarray.Dataset
        L-moment dataset with a new lmom dimension.

    Notes
    -----
        NaN values in each stations are removed before calculating L-moments.
        The function uses the `moment_l` function to calculate L-moments for each individual stations.
        Equations are based on Hosking, J. R. M., & Wallis, J. R. (1997). Regional frequency analysis (p. 240).
    """
    ds = xr.apply_ufunc(
        _moment_l_vector,
        ds,
        input_core_dims=[["time"]],
        output_core_dims=[["lmom"]],
        keep_attrs=True,
    ).assign_coords(lmom=["l1", "l2", "l3", "tau", "tau3", "tau4"])
    # TODO: add attributes
    for v in ds.var():
        ds[v].attrs["long_name"] = "L-moments"
        ds[v].attrs[
            "description"
        ] = "L-moments based on Hosking, J. R. M., & Wallis, J. R. (1997). Regional frequency analysis (p. 240)"
        ds[v].attrs["history"] = update_history("Computed L-moments", ds[v])
        ds[v].attrs.pop("cell_methods", None)
        ds[v].attrs.pop("units", None)
    return ds


def group_ds(ds: xr.Dataset, groups: list) -> xr.Dataset:
    """
    Group a dataset by a list of groups.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset to be grouped.
    groups : list
        A list of groups to be used for grouping the dataset.

    Returns
    -------
    xarray.Dataset
        A new dataset with the grouped data.
    """
    ds_groups = xr.concat(
        [
            ds.sel(id=groups[i]).assign_coords(group_id=i).expand_dims("group_id")
            for i in range(len(groups))
        ],
        dim="group_id",
    )
    ds_groups["group_id"].attrs["cf_role"] = "group_id"
    for v in ds_groups.var():
        ds_groups[v].attrs["history"] = update_history("Grouped with", ds_groups[v])
    return ds_groups
