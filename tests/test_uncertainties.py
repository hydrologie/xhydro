import numpy as np
import pandas as pd
import xarray as xr
from xclim.testing.helpers import test_timeseries as timeseries

import xhydro.frequency_analysis as xhfa
from xhydro.frequency_analysis.uncertainities import (
    boostrap_dist,
    boostrap_obs,
    calc_moments_iter,
    calc_q_iter,
    fit_boot_dist,
    generate_combinations,
)


def test_boostrap_obs():
    time = pd.date_range("2020-01-01", periods=5)
    data = xr.DataArray(np.array([1, 2, 3, 4, 5]), coords={"time": time})
    n_samples = 1000
    result = boostrap_obs(data, n_samples)
    assert result.shape == (n_samples, len(data))
    assert np.all(np.isin(result, data))


def test_boostrap_dist():
    ds = timeseries(
        np.array([50, 65, 80, 95, 110, 125, 140, 155, 170, 185, 200]),
        variable="streamflow",
        start="2001-01-01",
        freq="YS",
        as_dataset=True,
    )
    params = xhfa.local.fit(ds, distributions=["gumbel_r", "pearson3"])
    n_samples = 1000
    result = boostrap_dist(ds, params, n_samples)
    assert len(result.samples) == n_samples
    assert "samples" in result.coords


def test_fit_boot_dist():
    ds = timeseries(
        np.array([50, 65, 80, 95, 110, 125, 140, 155, 170, 185, 200]),
        variable="streamflow",
        start="2001-01-01",
        freq="YS",
        as_dataset=True,
    )
    params = xhfa.local.fit(ds, distributions=["gumbel_r", "pearson3"])
    n_samples = 1000
    bo = boostrap_dist(ds, params, n_samples)

    result = fit_boot_dist(bo)
    assert isinstance(result, xr.Dataset)
    assert "loc" in result.dparams and "scale" in result.dparams
    assert len(result.samples) == n_samples


def test_calc_moments_iter():
    ds = timeseries(
        np.array([50, 65, 80, 95, 110, 125, 140, 155, 170, 185, 200]),
        variable="streamflow",
        start="2001-01-01",
        freq="YS",
        as_dataset=True,
    )
    n_samples = 1000
    bo = boostrap_obs(ds, n_samples).assign_coords(id="S1").expand_dims("id")
    result = calc_moments_iter(bo)
    assert isinstance(result, xr.Dataset)
    assert "l1" in result.lmom and "l2" in result.lmom
    assert len(result.samples) == n_samples


def test_calc_q_iter():
    ds = timeseries(
        np.array([50, 65, 80, 95, 110, 125, 140, 155, 170, 185, 200]),
        variable="streamflow",
        start="2001-01-01",
        freq="YS",
        as_dataset=True,
    )

    n_samples = 1000
    bo = boostrap_obs(ds, n_samples, seed=42).assign_coords(id="S1").expand_dims("id")
    ds_moments_iter = calc_moments_iter(bo)
    time = pd.date_range("2020-01-01", periods=54)
    data = np.array(
        [
            [
                np.array(
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        190.0,
                        82.3,
                        84.0,
                        81.0,
                        92.2,
                        81.9,
                        65.7,
                        86.4,
                        115.0,
                        64.3,
                        53.8,
                        65.0,
                        69.7,
                        95.67,
                        91.2,
                        96.42,
                        68.79,
                        93.95,
                        93.35,
                        125.1,
                        51.88,
                        75.86,
                        114.9,
                        143.7,
                        74.74,
                        121.2,
                        157.4,
                        87.05,
                        112.5,
                        182.7,
                        150.1,
                        137.0,
                        159.3,
                        89.38,
                        71.53,
                        99.27,
                        62.36,
                        68.75,
                        100.3,
                        139.9,
                        112.0,
                        102.5,
                        69.8,
                        68.45,
                        105.4,
                    ]
                ),
                np.array(
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        52.52,
                        47.39,
                        49.46,
                        33.54,
                        31.47,
                        38.14,
                        28.91,
                        29.95,
                        50.08,
                        54.95,
                        39.66,
                        38.66,
                        22.52,
                        41.65,
                        30.85,
                    ]
                ),
                np.array(
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        353.0,
                        246.0,
                        166.0,
                        213.0,
                        427.0,
                        100.0,
                        140.0,
                        234.0,
                        171.0,
                        307.2,
                        317.1,
                        266.5,
                        141.6,
                        np.nan,
                        316.0,
                        197.1,
                        87.83,
                        140.1,
                        221.6,
                        157.3,
                        136.1,
                        262.4,
                        247.3,
                        153.2,
                        216.2,
                        332.3,
                        260.2,
                        177.1,
                        244.2,
                        223.8,
                        132.2,
                        163.1,
                        121.5,
                        180.3,
                        181.5,
                        231.6,
                        208.2,
                        204.7,
                        111.8,
                        133.8,
                        186.1,
                    ]
                ),
            ]
        ]
    )
    ds_groups = xr.Dataset(
        {"streamflow": (("group_id", "id", "time"), data)},
        coords={"time": time, "id": ["S1", "B", "C"], "group_id": ["G1"]},
    )
    ds_groups["id"].attrs["cf_role"] = "timeseries_id"
    ds_groups["streamflow"].attrs["units"] = "m^3 s-1"
    ds_moments_iter = xr.concat(
        [ds_moments_iter, ds_moments_iter, ds_moments_iter], dim="id"
    )
    ds_moments_iter["id"] = ["S1", "B", "C"]
    ds_moments_iter["id"].attrs["cf_role"] = "timeseries_id"
    result = calc_q_iter(
        "S1",
        "streamflow",
        ds_groups,
        ds_moments_iter,
        [100, 1000],
        small_regions_threshold=1,
    )
    assert "obs_samples" in result.coords
    assert len(result.samples) == len(ds_moments_iter.samples) * len(ds_groups.group_id)
    np.testing.assert_almost_equal(
        260.68926104,
        result.streamflow.sel(id="S1", group_id="G1", return_period=1000).quantile(0.5),
    )


def test_generate_combinations():
    df = np.array(
        [
            [2.67035877, 1.75919216, 0.76206293],
            [1.06732866, 2.05965305, 0.38237151],
            [-2.03313717, 2.72214445, -1.10333638],
            [0.23552091, 2.30462729, -0.480144],
            [2.28301367, 1.79640619, 0.05511961],
        ]
    )
    station = np.array(["020302", "020404", "020502", "020602", "020802"])
    components = [0, 1, 2]
    da = xr.DataArray(
        data=df, coords=[station, components], dims=["Station", "components"]
    )
    n_omit = 2
    result = generate_combinations(da, n_omit)
    assert 16 == len(result)
    assert len(result[0]) == len(station)
    assert len(result[-1]) == len(station) - n_omit
