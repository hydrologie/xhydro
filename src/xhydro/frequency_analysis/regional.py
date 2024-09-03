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

import math

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def cluster_indices(clust_num, labels_array):
    """
    Get the indices of elements with a specific cluster number using NumPy.

    Parameters
    ----------
    clust_num : (int)
        Cluster number to find indices for.
    labels_array : (numpy.ndarray)
        Array containing cluster labels.

    Returns
    -------
    numpy.ndarray
        Indices of elements with the specified cluster number.
    """
    return np.where(labels_array == clust_num)[0]


def get_groups_indices(cluster, sample):
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


def get_group_from_fit(model, param, sample):
    """
    Get indices of groups from a fit using the specified model and parameters.

    Parameters
    ----------
    model : obj
        Model class or instance with a fit method.
    param : disct
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


def fit_pca(ds, **kwargs):
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
    return data_pca, obj_pca


def _scale_data(ds):
    scalar = StandardScaler()
    df = ds.to_dataframe()

    scaled_data = pd.DataFrame(scalar.fit_transform(df))  # scaling the data
    scaled_data.columns = (
        df.columns
    )  # Attribue les noms de colonnes et l'index du dataframe original au dataframe mis à l'échelle
    scaled_data.index = df.index
    return xr.Dataset(scaled_data)


def moment_l_vector(x_vec):
    """
    Calculate L-moments for multiple datasets.

    This function computes L-moments for each dataset in the input vector,
    ignoring NaN values.

    Parameters
    ----------
    x_vec : array-like
        A vector of datasets, where each element is an array-like object
        containing the data for which to calculate L-moments.

    Returns
    -------
    list
        A list of L-moment arrays, one for each input dataset.
        Each L-moment array contains the L-moments calculated
        for the corresponding dataset, excluding NaN values.

    Notes
    -----
    - NaN values in each dataset are removed before calculating L-moments.
    - The function uses the `moment_l` function to calculate L-moments
      for each individual dataset.

    Examples
    --------
    >>> import numpy as np
    >>> data1 = np.array([1, 2, 3, np.nan, 5])
    >>> data2 = np.array([2, 4, 6, 8, 10])
    >>> moment_l_vector([data1, data2])
    [array([...]), array([...])]  # L-moments for data1 and data2
    """
    return [_moment_l(x[~np.isnan(x)]) for x in x_vec]


# Calcul des moments L
def _moment_l(x=[]):
    """
    Calculate L-moments for a given dataset.

    This function computes various L-moments and L-moment ratios for a given array of data.
    It can return the results either as a list or as an OrderedDict.

    Parameters
    ----------
    x : list or array-like
        Input data for which to calculate L-moments.
    ordered_dict : bool
        If True, return results as an OrderedDict; if False, return as a list.

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
    from collections import OrderedDict

    import numpy as np

    x = x[~np.isnan(x)]

    # Trier en ordre decroissant
    x_sort = np.sort(x)
    x_sort = x_sort[::-1]

    n = np.size(x_sort)
    j = np.arange(1, n + 1)

    # Moments ponderes
    b0 = np.mean(x_sort)
    b1 = np.dot((n - j) / (n * (n - 1)), x_sort)
    b2 = np.dot((n - j) * (n - j - 1) / (n * (n - 1) * (n - 2)), x_sort)
    b3 = np.dot(
        (n - j) * (n - j - 1) * (n - j - 2) / (n * (n - 1) * (n - 2) * (n - 3)), x_sort
    )

    # Moment L
    lambda1 = b0
    lambda2 = 2 * b1 - b0
    lambda3 = 6 * b2 - 6 * b1 + b0
    lambda4 = 20 * b3 - 30 * b2 + 12 * b1 - b0

    tau = lambda2 / lambda1  # L_CV # tau2 dans Anctil, tau dans Hosking et al.
    tau3 = lambda3 / lambda2  # L_CS
    tau4 = lambda4 / lambda2  # L_CK

    return [lambda1, lambda2, lambda3, tau, tau3, tau4]


# Faire un fct qui appele
def calc_h_z(ds_groups, ds_moments_groups, kap):
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

    Returns
    -------
    xarray.Dataset
        Dataset containing calculated H values and Z-scores for each group.

    Notes
    -----
    This function applies the heterogeneity measure and Z-score calculations
    to grouped data for regional frequency analysis. It uses L-moments and
    the Kappa3 distribution in the process.
    """
    tau = ds_moments_groups.sel(lmom="tau").load()
    tau3 = ds_moments_groups.sel(lmom="tau3").load()
    tau4 = ds_moments_groups.sel(lmom="tau4").load()
    longeur = ds_groups.count(dim="time").load()

    ds_h, ds_b4, ds_sigma4, ds_tau4_r = xr.apply_ufunc(
        _heterogeneite_et_score_z,
        kap,
        longeur,
        tau,
        tau3,
        tau4,
        input_core_dims=[[], ["id"], ["id"], ["id"], ["id"]],
        output_core_dims=[[], [], [], []],
        vectorize=True,
    )
    ds_tau4 = _calculate_gev_tau4(ds_groups.load(), ds_moments_groups.load())

    z_score = (
        ds_tau4 - ds_tau4_r + ds_b4
    ) / ds_sigma4  # Should we calc abs here or leave it for later ?

    z_score = _append_ds_vars_names(z_score, "_Z")

    ds_h = _append_ds_vars_names(ds_h, "_H")
    return _combine_h_z(xr.merge([z_score, ds_h]))


def _calculate_gev_tau4(ds_groups, ds_moments_groups):  # calcul indice de crue
    # H&W
    lambda_r_1, lambda_r_2, lambda_r_3 = _calc_lambda_r(ds_groups, ds_moments_groups)

    kappa = _calc_kappa(lambda_r_2, lambda_r_3)

    # Hosking et Wallis, éq. A53
    tau4 = (5 * (1 - 4**-kappa) - 10 * (1 - 3**-kappa) + 6 * (1 - 2**-kappa)) / (
        1 - 2**-kappa
    )
    return tau4


def _heterogeneite_et_score_z(kap, n=[], t=[], t3=[], t4=[]):

    import numpy as np

    np.random.seed(seed=42)

    # On enlève les valeurs hors groupe ou aberrantes
    # Certaines longeurs sont trop courtes poru calculer les moments, on les enlèeve aussi
    bool_maks = (n != 0) & (~np.isnan(t)) & (~np.isnan(t3)) & (~np.isnan(t4))
    n = n[bool_maks]
    t = t[bool_maks]
    t3 = t3[bool_maks]
    t4 = t4[bool_maks]

    # Hosking et Wallis, eq. 4.3
    tau_r = np.dot(n, t) / np.sum(n)
    tau3_r = np.dot(n, t3) / np.sum(n)
    tau4_r = np.dot(n, t4) / np.sum(n)

    # Calcul de l'ecart-type pondere L-CVs pour l'echantillon (eq. 4.4)
    v = np.sqrt(np.dot(n, (t - tau_r) ** 2) / np.sum(n))

    # Kappa distribution
    # Fit a kappa distribution to the region average L-moment ratios:
    try:
        kappa_param = kap.lmom_fit(lmom_ratios=[1, tau_r, tau3_r, tau4_r])
    except ValueError:

        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )  # Returning nans for H, b4, sigma4, tau4_r
    except Exception as error:
        if "Failed to converge" in repr(error):
            return (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )  # Returning nans for H, b4, sigma4, tau4_r
        else:
            raise error
    n_sim = 500  # Nbre de "regions virtuelles" simulees

    def calc_tsim(kappa_param, longeur, n_sim):

        # Pour chaque station, on génère 500 vecteurs de même longeur que la série d'obs
        rvs = kap.rvs(
            kappa_param["k"],
            kappa_param["h"],
            kappa_param["loc"],
            kappa_param["scale"],
            size=(n_sim, int(longeur)),
        )

        return _momentl_optim(rvs)  # Tau correspond au 3e moment.

    t_sim_tau4m = [calc_tsim(kappa_param, longeur, n_sim) for longeur in n]

    t_sim = np.array([tt[3] for tt in t_sim_tau4m])  # Tau correspond au 3e moment.
    tau4m = np.array([tt[5] for tt in t_sim_tau4m])  # Tau4 correspond au 5e terme.

    b4 = np.mean(tau4m - tau4_r)

    sigma4 = np.sqrt(
        (1 / (n_sim - 1)) * (np.sum((tau4m - tau4_r) ** 2) - (n_sim * b4 * b4))
    )

    # Calcul du V
    tau_r_sim = np.dot(n, t_sim) / np.sum(n)

    v_sim = np.sqrt(np.dot(n, (t_sim - tau_r_sim) ** 2) / np.sum(n))

    mu_v = np.mean(v_sim)
    sigma_v = np.std(v_sim)

    h = (v - mu_v) / sigma_v

    return h, b4, sigma4, tau4_r


# Calcul des moments L
def _momentl_optim(x=[]):
    if x.ndim == 1:
        x = x[~np.isnan(x)]
        # Trier en ordre decroissant
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
        print("Only 1d and 2d have been implemented")

    # Moment L
    lambda1 = b0
    lambda2 = 2 * b1 - b0
    lambda3 = 6 * b2 - 6 * b1 + b0
    lambda4 = 20 * b3 - 30 * b2 + 12 * b1 - b0

    tau = lambda2 / lambda1  # L_CV # tau2 dans Anctil, tau dans Hosking et al.
    tau3 = lambda3 / lambda2  # L_CS
    tau4 = lambda4 / lambda2  # L_CK

    return [lambda1, lambda2, lambda3, tau, tau3, tau4]


def _append_ds_vars_names(ds, suffix):
    for name in ds.data_vars:
        ds = ds.rename({name: name + suffix})
    return ds


def mask_h_z(ds, thresh_h=1, thresh_z=1.64):
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
    return (ds.sel(crit="H") < thresh_h) & (abs(ds.sel(crit="Z")) < thresh_z)


def _combine_h_z(ds):
    new_ds = xr.Dataset()
    for v in ds:
        if "_Z" in v:
            new_ds[v.removesuffix("_Z")] = xr.concat(
                [ds[v.replace("_Z", "_H")], ds[v]], dim="crit"
            ).assign_coords(crit=["H", "Z"])
    return new_ds


def calculate_rp_from_afr(ds_groups, ds_moments_groups, rp, l1=None):
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
    The function internally calls calculate_ic_from_AFR to compute the index flood.
    """
    if l1 is None:
        l1 = ds_moments_groups.sel(lmom="l1").dropna(dim="id", how="all")
    return _calculate_ic_from_afr(ds_groups, ds_moments_groups, rp) * l1


def _calculate_ic_from_afr(ds_groups, ds_moments_groups, rp):  # calcul indice de crue

    lambda_r_1, lambda_r_2, lambda_r_3 = _calc_lambda_r(ds_groups, ds_moments_groups)

    # alpha = parametre de localisation (location)
    # xi    = parametre d'echelle (scale)
    # kappa = parametre de forme (shape)

    kappa = _calc_kappa(lambda_r_2, lambda_r_3)

    term = xr.apply_ufunc(_calc_gamma, (1 + kappa), vectorize=True)

    # Hosking et Wallis, éq. A56. et Anctil et al. 1998, eq. 7 et 8.
    alpha = (lambda_r_2 * kappa) / ((1 - (2**-kappa)) * term)
    xi = lambda_r_1 + (alpha * (term - 1)) / kappa

    # On calcul les périodes de retour souhaitées

    t = xr.DataArray(data=rp, dims="rp").assign_coords(rp=rp)

    # Hosking et Wallis, éq. A44 et Anctil et al. 1998, eq. 5.
    q_rt = xi + alpha * (1 - (-np.log((t - 1) / t)) ** kappa) / kappa

    # Pour obtenir le quantile à un site particulier, il suffit de multiplier Q_RT
    # par la moyenne des maximums annuels à ce site (lambda_1).
    # Hosking et Wallis, éq. 6.4 et Anctil et al. 1998, eq. 9.
    return q_rt


def _calc_gamma(val):
    return math.gamma(val)


def remove_small_regions(ds, thresh=5):
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
    for gr in ds.group_id:
        if len(ds.sel(group_id=gr).dropna(dim="id", how="all").id) < thresh:
            ds = ds.drop_sel(group_id=gr)
    return ds


def _calc_kappa(lambda_r_2, lambda_r_3):

    # NB Il y a une erreur de signe dans Anctil et al. 1998, eq. 6. Voir # Hosking et Wallis, éq. A55
    c = (2 / (3 + (lambda_r_3 / lambda_r_2))) - (np.log(2) / np.log(3))

    # Hosking et Wallis, éq. A55 (approximation generalement acceptable)
    kappa = 7.8590 * c + 2.9554 * (c**2)
    return kappa


def _calc_lambda_r(ds_groups, ds_moments_groups):
    # H&W
    nr = ds_moments_groups.count(dim="id").isel(lmom=0)
    nk = ds_groups.dropna(dim="id", how="all").count(dim="time")
    wk = (nk * nr) / (nk + nr)

    lambda_k0 = ds_moments_groups.sel(lmom="l1").dropna(dim="id", how="all")
    lambda_k1 = ds_moments_groups.sel(lmom="l2").dropna(dim="id", how="all")
    lambda_k2 = ds_moments_groups.sel(lmom="l3").dropna(dim="id", how="all")

    l2 = (lambda_k1 / lambda_k0) * wk
    l3 = (lambda_k2 / lambda_k0) * wk

    lambda_r_1 = 1  # Anctil et al. 1998 (p.362)
    lambda_r_2 = l2.sum(dim="id") / wk.sum(dim="id")
    lambda_r_3 = l3.sum(dim="id") / wk.sum(dim="id")
    return lambda_r_1, lambda_r_2, lambda_r_3
