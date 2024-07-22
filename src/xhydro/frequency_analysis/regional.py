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
def _moment_l(x=[], ordered_dict=False):
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
    OrderedDict or list : Depending on the ordered_dict parameter, returns either:
        - OrderedDict with keys: "loc" (lambda1), "scale" (lambda2), "tau" (L-CV), "tau3" (L-CS), "tau4" (L-CK)
        - List in the order: [lambda1, lambda2, lambda3, tau, tau3, tau4]

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
    # Not sure its still used, to check
    if ordered_dict:
        return OrderedDict(
            [
                ("loc", lambda1),
                ("scale", lambda2),
                ("tau", tau),
                ("tau3", tau3),
                ("tau4", tau4),
            ]
        )
    else:
        return [lambda1, lambda2, lambda3, tau, tau3, tau4]
