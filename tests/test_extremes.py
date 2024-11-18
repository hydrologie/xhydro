import os
import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

try:
    from xhydro.extreme_value_analysis.parameterestimation import fit, return_level
except ImportError:
    from xhydro.extreme_value_analysis import JULIA_WARNING

    warnings.warn(JULIA_WARNING)
    parameterestimation = None


# pytest -n0 --cov=src\xhydro --cov-report=html tests/test_extremes.py
@pytest.fixture()
def data_fre():
    """
    Data from Extremes.jl

    data = Extremes.dataset("fremantle")

    Sea-levels recorded at Fremantle in West Australia from 1897 to 1989, studied by Coles (2001) in Chapter 6.

    """
    data = {
        "Year": [
            1897,
            1898,
            1899,
            1900,
            1901,
            1903,
            1904,
            1905,
            1906,
            1908,
            1909,
            1912,
            1914,
            1915,
            1916,
            1917,
            1918,
            1919,
            1920,
            1921,
            1922,
            1923,
            1924,
            1925,
            1927,
            1928,
            1929,
            1930,
            1931,
            1932,
            1933,
            1934,
            1935,
            1936,
            1937,
            1938,
            1939,
            1940,
            1941,
            1943,
            1944,
            1945,
            1946,
            1947,
            1948,
            1949,
            1950,
            1951,
            1952,
            1953,
            1954,
            1955,
            1956,
            1957,
            1958,
            1959,
            1960,
            1961,
            1962,
            1963,
            1964,
            1965,
            1966,
            1967,
            1968,
            1969,
            1970,
            1971,
            1972,
            1973,
            1974,
            1975,
            1976,
            1977,
            1978,
            1979,
            1980,
            1981,
            1982,
            1983,
            1984,
            1985,
            1986,
            1987,
            1988,
            1989,
        ],
        "SeaLevel": [
            1.58,
            1.71,
            1.4,
            1.34,
            1.43,
            1.19,
            1.55,
            1.34,
            1.37,
            1.46,
            1.92,
            1.37,
            1.19,
            1.4,
            1.28,
            1.52,
            1.52,
            1.58,
            1.49,
            1.65,
            1.37,
            1.49,
            1.46,
            1.34,
            1.74,
            1.62,
            1.46,
            1.71,
            1.74,
            1.55,
            1.43,
            1.62,
            1.49,
            1.58,
            1.34,
            1.37,
            1.62,
            1.31,
            1.43,
            1.49,
            1.55,
            1.71,
            1.49,
            1.46,
            1.52,
            1.58,
            1.65,
            1.49,
            1.52,
            1.52,
            1.49,
            1.62,
            1.86,
            1.58,
            1.62,
            1.46,
            1.43,
            1.46,
            1.62,
            1.68,
            1.83,
            1.62,
            1.46,
            1.58,
            1.77,
            1.62,
            1.71,
            1.46,
            1.6,
            1.5,
            1.6,
            1.9,
            1.7,
            1.4,
            1.8,
            1.37,
            1.46,
            1.61,
            1.43,
            1.67,
            1.62,
            1.57,
            1.56,
            1.46,
            1.7,
            1.51,
        ],
        "SOI": [
            -0.67,
            0.57,
            0.16,
            -0.65,
            0.06,
            0.47,
            0.39,
            -1.78,
            0.2,
            0.28,
            0.28,
            -0.97,
            -0.92,
            0.16,
            0.62,
            2.12,
            0.05,
            -1.09,
            0.08,
            0.66,
            0.33,
            -0.36,
            0.33,
            -0.24,
            0.27,
            0.43,
            0.46,
            0.03,
            0.39,
            -0.68,
            0.09,
            -0.01,
            0.14,
            0.03,
            0.09,
            0.86,
            0.02,
            -1.52,
            -1.44,
            0.35,
            -0.27,
            0.42,
            -0.79,
            0.16,
            -0.24,
            -0.21,
            1.49,
            -0.69,
            -0.23,
            -0.76,
            0.23,
            0.89,
            1.0,
            -0.45,
            -0.5,
            -0.11,
            0.28,
            -0.01,
            0.38,
            -0.32,
            0.53,
            -0.97,
            -0.53,
            0.25,
            0.19,
            -0.66,
            0.28,
            1.06,
            -0.88,
            0.63,
            0.97,
            1.32,
            0.06,
            -1.13,
            -0.3,
            -0.08,
            -0.43,
            0.06,
            -1.44,
            -0.94,
            -0.14,
            -0.07,
            -0.32,
            -1.47,
            0.73,
            0.61,
        ],
    }

    df = pd.DataFrame(data)

    ds = xr.Dataset(
        {
            "SeaLevel": ("Year", df["SeaLevel"].values),
            "SOI": ("Year", df["SOI"].values),
        },
        coords={"Year": df["Year"].values},
    )
    return ds


@pytest.fixture()
def data_rain():
    """
    Data from Extremes.jl

    data = Extremes.dataset("rain")
    df = filter(row -> row.Rainfall > threshold, data)
    df[!,:Exceedance] = df[:,:Rainfall] .- threshold
    df[!,:Year] = year.(df[:,:Date])

    Rainfall accumulations at a location in south-west England from 1914 to 1962, studied by Coles (2001) in Chapter 6.

    """
    data = {
        "Year": [
            1914,
            1914,
            1914,
            1914,
            1915,
            1915,
            1915,
            1916,
            1916,
            1916,
            1917,
            1917,
            1918,
            1918,
            1918,
            1920,
            1921,
            1921,
            1922,
            1922,
            1922,
            1922,
            1923,
            1923,
            1924,
            1924,
            1924,
            1924,
            1924,
            1925,
            1925,
            1925,
            1926,
            1926,
            1926,
            1927,
            1927,
            1928,
            1928,
            1928,
            1928,
            1928,
            1928,
            1928,
            1928,
            1929,
            1930,
            1930,
            1930,
            1931,
            1931,
            1931,
            1932,
            1932,
            1933,
            1933,
            1934,
            1934,
            1935,
            1935,
            1936,
            1936,
            1937,
            1937,
            1937,
            1937,
            1937,
            1937,
            1938,
            1938,
            1939,
            1939,
            1939,
            1939,
            1940,
            1941,
            1941,
            1942,
            1942,
            1942,
            1942,
            1943,
            1943,
            1944,
            1945,
            1945,
            1945,
            1945,
            1945,
            1945,
            1945,
            1946,
            1946,
            1946,
            1947,
            1948,
            1948,
            1948,
            1948,
            1948,
            1948,
            1949,
            1949,
            1949,
            1950,
            1950,
            1950,
            1950,
            1950,
            1950,
            1951,
            1951,
            1952,
            1952,
            1952,
            1952,
            1952,
            1953,
            1953,
            1953,
            1954,
            1954,
            1954,
            1954,
            1955,
            1955,
            1955,
            1955,
            1955,
            1956,
            1956,
            1956,
            1956,
            1956,
            1957,
            1957,
            1957,
            1957,
            1958,
            1958,
            1958,
            1958,
            1958,
            1959,
            1959,
            1959,
            1959,
            1959,
            1960,
            1960,
            1961,
            1961,
        ],
        "Exceedance": [
            1.8,
            2.5,
            1.8,
            14.5,
            0.5,
            13.2,
            5.6,
            8.1,
            2.0,
            1.8,
            3.0,
            9.1,
            0.5,
            1.8,
            2.3,
            3.0,
            0.5,
            2.5,
            18.5,
            5.3,
            10.6,
            0.5,
            4.3,
            2.8,
            0.5,
            15.7,
            1.8,
            3.5,
            3.5,
            1.8,
            4.8,
            5.3,
            7.8,
            46.7,
            2.3,
            4.0,
            3.8,
            6.6,
            0.5,
            15.7,
            56.6,
            5.6,
            17.8,
            17.5,
            4.3,
            18.5,
            0.7,
            13.4,
            29.4,
            5.1,
            23.3,
            3.5,
            0.5,
            0.2,
            10.9,
            12.7,
            53.3,
            24.9,
            29.2,
            1.8,
            7.3,
            2.5,
            4.0,
            37.3,
            1.2,
            0.2,
            6.1,
            6.8,
            8.4,
            1.0,
            3.3,
            17.0,
            2.0,
            3.0,
            8.1,
            0.5,
            42.4,
            4.3,
            7.1,
            3.0,
            10.9,
            9.9,
            17.0,
            6.3,
            0.5,
            0.5,
            25.9,
            1.8,
            21.3,
            55.3,
            11.9,
            0.5,
            3.0,
            5.6,
            25.9,
            14.2,
            8.1,
            4.3,
            1.8,
            2.0,
            1.8,
            5.6,
            15.2,
            0.5,
            9.4,
            0.2,
            14.5,
            1.8,
            3.8,
            21.6,
            5.3,
            29.4,
            3.5,
            5.3,
            0.5,
            6.8,
            17.8,
            12.9,
            7.6,
            25.4,
            5.3,
            12.4,
            3.0,
            3.0,
            10.1,
            4.8,
            8.1,
            9.4,
            4.0,
            5.6,
            4.3,
            3.5,
            1.0,
            6.6,
            6.3,
            8.4,
            8.1,
            17.0,
            1.0,
            0.5,
            1.2,
            5.6,
            18.8,
            11.9,
            1.7,
            1.2,
            21.3,
            3.5,
            7.6,
            9.4,
            9.4,
            15.7,
        ],
        "Rainfall": [
            31.8,
            32.5,
            31.8,
            44.5,
            30.5,
            43.2,
            35.6,
            38.1,
            32.0,
            31.8,
            33.0,
            39.1,
            30.5,
            31.8,
            32.3,
            33.0,
            30.5,
            32.5,
            48.5,
            35.3,
            40.6,
            30.5,
            34.3,
            32.8,
            30.5,
            45.7,
            31.8,
            33.5,
            33.5,
            31.8,
            34.8,
            35.3,
            37.8,
            76.7,
            32.3,
            34.0,
            33.8,
            36.6,
            30.5,
            45.7,
            86.6,
            35.6,
            47.8,
            47.5,
            34.3,
            48.5,
            30.7,
            43.4,
            59.4,
            35.1,
            53.3,
            33.5,
            30.5,
            30.2,
            40.9,
            42.7,
            83.3,
            54.9,
            59.2,
            31.8,
            37.3,
            32.5,
            34.0,
            67.3,
            31.2,
            30.2,
            36.1,
            36.8,
            38.4,
            31.0,
            33.3,
            47.0,
            32.0,
            33.0,
            38.1,
            30.5,
            72.4,
            34.3,
            37.1,
            33.0,
            40.9,
            39.9,
            47.0,
            36.3,
            30.5,
            30.5,
            55.9,
            31.8,
            51.3,
            85.3,
            41.9,
            30.5,
            33.0,
            35.6,
            55.9,
            44.2,
            38.1,
            34.3,
            31.8,
            32.0,
            31.8,
            35.6,
            45.2,
            30.5,
            39.4,
            30.2,
            44.5,
            31.8,
            33.8,
            51.6,
            35.3,
            59.4,
            33.5,
            35.3,
            30.5,
            36.8,
            47.8,
            42.9,
            37.6,
            55.4,
            35.3,
            42.4,
            33.0,
            33.0,
            40.1,
            34.8,
            38.1,
            39.4,
            34.0,
            35.6,
            34.3,
            33.5,
            31.0,
            36.6,
            36.3,
            38.4,
            38.1,
            47.0,
            31.0,
            30.5,
            31.2,
            35.6,
            48.8,
            41.9,
            31.7,
            31.2,
            51.3,
            33.5,
            37.6,
            39.4,
            39.4,
            45.7,
        ],
        "Date": [
            "1914-02-07",
            "1914-03-08",
            "1914-12-17",
            "1914-12-30",
            "1915-02-13",
            "1915-02-16",
            "1915-12-14",
            "1916-02-03",
            "1916-08-29",
            "1916-11-04",
            "1917-06-28",
            "1917-08-27",
            "1918-01-15",
            "1918-01-18",
            "1918-11-01",
            "1920-03-12",
            "1921-09-11",
            "1921-12-18",
            "1922-02-05",
            "1922-02-06",
            "1922-02-08",
            "1922-10-25",
            "1923-04-13",
            "1923-10-07",
            "1924-02-11",
            "1924-09-18",
            "1924-10-21",
            "1924-10-31",
            "1924-12-31",
            "1925-10-31",
            "1925-11-07",
            "1925-11-08",
            "1926-01-27",
            "1926-07-13",
            "1926-12-20",
            "1927-01-27",
            "1927-12-10",
            "1928-02-01",
            "1928-07-27",
            "1928-08-02",
            "1928-10-04",
            "1928-11-17",
            "1928-11-18",
            "1928-11-23",
            "1928-12-03",
            "1929-08-19",
            "1930-08-03",
            "1930-09-02",
            "1930-11-02",
            "1931-01-08",
            "1931-06-30",
            "1931-10-07",
            "1932-07-14",
            "1932-10-09",
            "1933-08-04",
            "1933-12-07",
            "1934-10-04",
            "1934-11-13",
            "1935-06-29",
            "1935-12-13",
            "1936-01-21",
            "1936-02-25",
            "1937-07-05",
            "1937-08-02",
            "1937-08-21",
            "1937-11-11",
            "1937-11-24",
            "1937-12-08",
            "1938-04-03",
            "1938-11-04",
            "1939-02-15",
            "1939-07-12",
            "1939-11-02",
            "1939-11-03",
            "1940-08-22",
            "1941-05-16",
            "1941-12-09",
            "1942-01-10",
            "1942-01-29",
            "1942-02-07",
            "1942-05-31",
            "1943-11-16",
            "1943-12-16",
            "1944-10-19",
            "1945-08-08",
            "1945-08-09",
            "1945-09-02",
            "1945-09-06",
            "1945-11-18",
            "1945-11-22",
            "1945-11-26",
            "1946-03-11",
            "1946-07-19",
            "1946-08-15",
            "1947-12-02",
            "1948-07-16",
            "1948-08-24",
            "1948-10-16",
            "1948-10-17",
            "1948-10-22",
            "1948-11-16",
            "1949-08-07",
            "1949-08-19",
            "1949-11-19",
            "1950-02-02",
            "1950-02-14",
            "1950-03-05",
            "1950-05-21",
            "1950-06-08",
            "1950-11-03",
            "1951-07-07",
            "1951-08-15",
            "1952-05-13",
            "1952-05-17",
            "1952-06-20",
            "1952-10-23",
            "1952-10-25",
            "1953-07-24",
            "1953-11-21",
            "1953-11-25",
            "1954-10-17",
            "1954-11-01",
            "1954-11-02",
            "1954-12-18",
            "1955-01-12",
            "1955-07-18",
            "1955-07-28",
            "1955-09-27",
            "1955-12-27",
            "1956-01-31",
            "1956-02-07",
            "1956-03-08",
            "1956-08-07",
            "1956-12-09",
            "1957-02-08",
            "1957-08-23",
            "1957-09-01",
            "1957-10-02",
            "1958-01-04",
            "1958-04-24",
            "1958-10-16",
            "1958-11-13",
            "1958-11-24",
            "1959-01-22",
            "1959-02-25",
            "1959-04-01",
            "1959-10-26",
            "1959-12-03",
            "1960-08-09",
            "1960-10-05",
            "1961-01-14",
            "1961-09-28",
        ],
    }

    df = pd.DataFrame(data)

    ds = xr.Dataset(
        {
            "Year": ("Date", df["Year"].values),
            "Exceedance": ("Date", df["Exceedance"].values),
            "Rainfall": ("Date", df["Rainfall"].values),
        },
        coords={"Date": df["Date"].values},
    )
    return ds


class Testfit:

    def test_stationary_grv(self, data_fre):
        p = fit(data_fre, dist="genextreme", method="ml", vars=["SeaLevel"], dim="Year")

        assert isinstance(p, xr.Dataset)
        assert list(p.coords) == ["dparams"]
        assert p.attrs["method"] == "Maximum likelihood"
        assert (
            (p["SeaLevel_lower"] < p["SeaLevel"])
            & (p["SeaLevel"] < p["SeaLevel_upper"])
        ).values.all()
        np.testing.assert_array_equal(
            p.coords["dparams"].values, np.array(["shape", "loc", "scale"])
        )
        np.testing.assert_array_equal(
            list(p.data_vars), ["SeaLevel", "SeaLevel_lower", "SeaLevel_upper"]
        )
        np.testing.assert_array_almost_equal(
            p.to_array().values,
            np.array(
                [
                    [0.21742419, 1.48234111, 0.14127133],
                    [0.09241807, 1.44956045, 0.12044072],
                    [0.3424303, 1.51512177, 0.16570467],
                ]
            ),
        )

    def test_stationary_gumbel(self, data_fre):
        p = fit(data_fre, dist="gumbel_r", method="ml", vars=["SeaLevel"], dim="Year")

        assert isinstance(p, xr.Dataset)
        assert list(p.coords) == ["dparams"]
        assert p.attrs["method"] == "Maximum likelihood"
        assert (
            (p["SeaLevel_lower"] < p["SeaLevel"])
            & (p["SeaLevel"] < p["SeaLevel_upper"])
        ).values.all()
        np.testing.assert_array_equal(
            p.coords["dparams"].values, np.array(["loc", "scale"])
        )
        np.testing.assert_array_equal(
            list(p.data_vars), ["SeaLevel", "SeaLevel_lower", "SeaLevel_upper"]
        )
        np.testing.assert_array_almost_equal(
            p.to_array().values,
            np.array(
                [
                    [1.4662802, 0.13941243],
                    [1.43506132, 0.11969685],
                    [1.49749908, 0.1623754],
                ]
            ),
        )

    def test_stationary_pareto(self, data_rain):
        p = fit(
            data_rain, dist="genpareto", method="ml", vars=["Exceedance"], dim="Date"
        )

        assert isinstance(p, xr.Dataset)
        assert list(p.coords) == ["dparams"]
        assert p.attrs["method"] == "Maximum likelihood"
        assert (
            (p["Exceedance_lower"] < p["Exceedance"])
            & (p["Exceedance"] < p["Exceedance_upper"])
        ).values.all()
        np.testing.assert_array_equal(
            p.coords["dparams"].values, np.array(["scale", "shape"])
        )
        np.testing.assert_array_equal(
            list(p.data_vars), ["Exceedance", "Exceedance_lower", "Exceedance_upper"]
        )
        np.testing.assert_array_almost_equal(
            p.to_array().values,
            np.array(
                [
                    [7.44019083, 0.1844927],
                    [5.77994415, -0.01385753],
                    [9.57733124, 0.38284292],
                ]
            ),
        )

    def test_non_stationary_cov(self, data_fre):
        p = fit(
            data_fre,
            dist="genextreme",
            method="ml",
            vars=["SeaLevel"],
            dim="Year",
            locationcov=["Year"],
        )

        assert isinstance(p, xr.Dataset)
        assert list(p.coords) == ["dparams"]
        assert p.attrs["method"] == "Maximum likelihood"
        assert (
            (p["SeaLevel_lower"] < p["SeaLevel"])
            & (p["SeaLevel"] < p["SeaLevel_upper"])
        ).values.all()
        np.testing.assert_array_equal(
            p.coords["dparams"].values,
            np.array(["shape", "loc", "loc_Year_covariate", "scale"]),
        )
        np.testing.assert_array_equal(
            list(p.data_vars), ["SeaLevel", "SeaLevel_lower", "SeaLevel_upper"]
        )
        np.testing.assert_array_almost_equal(
            p.to_array().values,
            np.array(
                [
                    [1.25303152e-01, -2.47278806e00, 2.03216246e-03, 1.24325843e-01],
                    [-1.13801539e-02, -4.44905129e00, 1.01748196e-03, 1.05446399e-01],
                    [2.61986457e-01, -4.96524827e-01, 3.04684296e-03, 1.46585520e-01],
                ]
            ),
        )

    def test_non_stationary_cov2(self, data_fre):
        p = fit(
            data_fre,
            dist="genextreme",
            method="ml",
            vars=["SeaLevel"],
            dim="Year",
            locationcov=["Year", "SOI"],
            scalecov=["Year", "SOI"],
        )

        assert isinstance(p, xr.Dataset)
        assert list(p.coords) == ["dparams"]
        assert p.attrs["method"] == "Maximum likelihood"
        assert (
            (p["SeaLevel_lower"] < p["SeaLevel"])
            & (p["SeaLevel"] < p["SeaLevel_upper"])
        ).values.all()
        np.testing.assert_array_equal(
            p.coords["dparams"].values,
            np.array(
                [
                    "shape",
                    "loc",
                    "loc_Year_covariate",
                    "loc_SOI_covariate",
                    "scale",
                    "scale_Year_covariate",
                    "scale_SOI_covariate",
                ]
            ),
        )
        np.testing.assert_array_equal(
            list(p.data_vars), ["SeaLevel", "SeaLevel_lower", "SeaLevel_upper"]
        )
        np.testing.assert_array_almost_equal(
            p.to_array().values,
            np.array(
                [
                    [
                        2.22924013e-01,
                        -2.21521391e00,
                        1.90673297e-03,
                        6.59255626e-02,
                        8.63429815e02,
                        9.95445528e-01,
                        1.30359418e00,
                    ],
                    [
                        7.44730669e-02,
                        -4.01218231e00,
                        9.86305458e-04,
                        3.15112449e-02,
                        6.20212792e-02,
                        9.90579586e-01,
                        1.04009190e00,
                    ],
                    [
                        3.71374958e-01,
                        -4.18245514e-01,
                        2.82716048e-03,
                        1.00339880e-01,
                        12020246.18976905,
                        1.00033537e00,
                        1.63385350e00,
                    ],
                ]
            ),
        )

    def test_non_stationary_cov_pwm(self, data_fre):

        with pytest.raises(
            ValueError,
            match="Probability weighted moment parameter estimation cannot have covariates \\['Year', 'SOI', 'Year2', 'SOI2', 'Year3', 'SOI3'\\]",
        ):

            fit(
                data_fre,
                dist="genextreme",
                method="pwm",
                vars=["SeaLevel"],
                dim="Year",
                locationcov=["Year", "SOI"],
                scalecov=["Year2", "SOI2"],
                shapecov=["Year3", "SOI3"],
            )

    def test_fewer_entries_than_param_param(self, data_fre):

        with pytest.warns(
            UserWarning,
            match="The fitting data contains fewer entries than the number of parameters for the given distribution. "
            "Returned parameters are numpy.nan",
        ):
            p = fit(
                data_fre.isel(Year=slice(1, 2)),
                dist="genextreme",
                method="ml",
                vars=["SeaLevel"],
                dim="Year",
                locationcov=["Year", "SOI"],
                scalecov=["Year", "SOI"],
            )

        assert isinstance(p, xr.Dataset)
        assert list(p.coords) == ["dparams"]
        assert p.attrs["method"] == "Maximum likelihood"
        assert np.isnan(p.to_array().values).all()
        np.testing.assert_array_equal(
            p.coords["dparams"].values,
            np.array(
                [
                    "shape",
                    "loc",
                    "loc_Year_covariate",
                    "loc_SOI_covariate",
                    "scale",
                    "scale_Year_covariate",
                    "scale_SOI_covariate",
                ]
            ),
        )
        np.testing.assert_array_equal(
            list(p.data_vars), ["SeaLevel", "SeaLevel_lower", "SeaLevel_upper"]
        )

    def test_fewer_entries_than_param_rtnlev(self, data_fre):

        with pytest.warns(
            UserWarning,
            match="The fitting data contains fewer entries than the number of parameters for the given distribution. "
            "Returned parameters are numpy.nan",
        ):
            p = return_level(
                data_fre.isel(Year=slice(0, 5)),
                dist="genextreme",
                method="ml",
                vars=["SeaLevel"],
                dim="Year",
                locationcov=["Year", "SOI"],
                scalecov=["Year", "SOI"],
            )

        assert isinstance(p, xr.Dataset)
        assert list(p.coords) == ["return_level"]
        assert p.attrs["method"] == "Maximum likelihood"
        assert np.isnan(p.to_array().values).all()
        np.testing.assert_array_equal(
            p.coords["return_level"].values, np.array([1897, 1898, 1899, 1900, 1901])
        )
        np.testing.assert_array_equal(
            list(p.data_vars), ["SeaLevel", "SeaLevel_lower", "SeaLevel_upper"]
        )


class TestRtnlv:

    def test_stationary(self, data_fre):

        p = return_level(
            data_fre,
            dist="genextreme",
            method="ml",
            vars=["SeaLevel"],
            dim="Year",
            return_period=100,
        )

        assert isinstance(p, xr.Dataset)
        assert list(p.coords) == ["return_level"]
        assert p.attrs["method"] == "Maximum likelihood"
        assert p.to_array().shape[1] == 1
        assert (
            (p["SeaLevel_lower"] < p["SeaLevel"])
            & (p["SeaLevel"] < p["SeaLevel_upper"])
        ).values.all()
        np.testing.assert_array_equal(
            p.coords["return_level"].values, np.array(["return_level"])
        )
        np.testing.assert_array_equal(
            list(p.data_vars), ["SeaLevel", "SeaLevel_lower", "SeaLevel_upper"]
        )
        np.testing.assert_array_almost_equal(
            p.to_array().values, np.array([[1.89310525], [1.81018661], [1.9760239]])
        )

    def test_non_stationary_cov(self, data_fre):

        p = return_level(
            data_fre,
            dist="genextreme",
            method="ml",
            vars=["SeaLevel"],
            dim="Year",
            locationcov=["Year", "SOI"],
            return_period=100,
        )

        assert isinstance(p, xr.Dataset)
        assert list(p.coords) == ["return_level"]
        assert p.attrs["method"] == "Maximum likelihood"
        assert len(data_fre["SeaLevel"]) == p.to_array().shape[1]
        assert (
            (p["SeaLevel_lower"] < p["SeaLevel"])
            & (p["SeaLevel"] < p["SeaLevel_upper"])
        ).values.all()
        np.testing.assert_array_equal(
            p.coords["return_level"].values[:3], np.array([1897, 1898, 1899])
        )
        np.testing.assert_array_equal(
            list(p.data_vars), ["SeaLevel", "SeaLevel_lower", "SeaLevel_upper"]
        )
        np.testing.assert_array_almost_equal(
            p.to_array()[:, :5].values,
            np.array(
                [
                    [1.74899368, 1.81870965, 1.79847143, 1.75642614, 1.79724773],
                    [1.64648291, 1.72510361, 1.70486888, 1.65503795, 1.70372785],
                    [1.85150446, 1.91231569, 1.89207398, 1.85781432, 1.89076761],
                ]
            ),
        )

    def test_recover_nan(self, data_fre):

        data_fre_nan = data_fre.copy(deep=True)
        data_fre_nan["SeaLevel"].isel(Year=slice(1, 3))[:] = np.nan
        data_fre_nan["SOI"].isel(Year=slice(5, 6))[:] = np.nan

        p = return_level(
            data_fre_nan,
            dist="genextreme",
            method="ml",
            vars=["SeaLevel"],
            dim="Year",
            locationcov=["Year", "SOI"],
            return_period=100,
        )

        assert isinstance(p, xr.Dataset)
        assert list(p.coords) == ["return_level"]
        assert p.attrs["method"] == "Maximum likelihood"
        assert len(data_fre["SeaLevel"]) == p.to_array().shape[1]
        np.testing.assert_array_equal(
            p.coords["return_level"].values[:3], np.array([1897, 1898, 1899])
        )
        np.testing.assert_array_equal(
            list(p.data_vars), ["SeaLevel", "SeaLevel_lower", "SeaLevel_upper"]
        )
        np.testing.assert_array_almost_equal(
            p.to_array()[:, :7].values,
            np.array(
                [
                    [
                        1.7545241,
                        np.nan,
                        np.nan,
                        1.76138563,
                        1.80297135,
                        np.nan,
                        1.82715405,
                    ],
                    [
                        1.64451242,
                        np.nan,
                        np.nan,
                        1.65252072,
                        1.700345,
                        np.nan,
                        1.72610061,
                    ],
                    [
                        1.86453578,
                        np.nan,
                        np.nan,
                        1.87025054,
                        1.90559771,
                        np.nan,
                        1.92820748,
                    ],
                ]
            ),
        )


class TestGEV:

    def test_ml_param(self, data_fre):
        p = fit(data_fre, dist="genextreme", method="ML", dim="Year", vars=["SeaLevel"])

        assert (
            (p["SeaLevel_lower"] < p["SeaLevel"])
            & (p["SeaLevel"] < p["SeaLevel_upper"])
        ).values.all()
        np.testing.assert_array_almost_equal(
            p.to_array().values,
            np.array(
                [
                    [0.21742419, 1.48234111, 0.14127133],
                    [0.09241807, 1.44956045, 0.12044072],
                    [0.3424303, 1.51512177, 0.16570467],
                ]
            ),
        )

    def test_pwm_param(self, data_fre):
        p = fit(
            data_fre, dist="genextreme", method="PWM", dim="Year", vars=["SeaLevel"]
        )

        assert (
            (p["SeaLevel_lower"] < p["SeaLevel"])
            & (p["SeaLevel"] < p["SeaLevel_upper"])
        ).values.all()
        np.testing.assert_array_almost_equal(
            p.to_array().values[0], np.array([0.19631625, 1.4807491, 0.13907873])
        )

    def test_bayes_param(self, data_fre):
        p = fit(
            data_fre, dist="genextreme", method="BAYES", dim="Year", vars=["SeaLevel"]
        )

        assert (
            (p["SeaLevel_lower"] < p["SeaLevel"])
            & (p["SeaLevel"] < p["SeaLevel_upper"])
        ).values.all()

    def test_ml_rtnlv(self, data_fre):
        p = return_level(
            data_fre, dist="genextreme", method="ML", dim="Year", vars=["SeaLevel"]
        )

        assert (
            (p["SeaLevel_lower"] < p["SeaLevel"])
            & (p["SeaLevel"] < p["SeaLevel_upper"])
        ).values.all()
        np.testing.assert_array_almost_equal(
            p.to_array().values, np.array([[1.89310525], [1.81018661], [1.9760239]])
        )

    def test_pwm_rtnlv(self, data_fre):
        p = return_level(
            data_fre, dist="genextreme", method="PWM", dim="Year", vars=["SeaLevel"]
        )

        np.testing.assert_array_almost_equal(
            p.to_array().values[0], np.array([1.90204717])
        )
        assert (
            (p["SeaLevel_lower"] < p["SeaLevel"])
            & (p["SeaLevel"] < p["SeaLevel_upper"])
        ).values.all()

    def test_bayes_rtnlv(self, data_fre):
        p = return_level(
            data_fre, dist="genextreme", method="BAYES", dim="Year", vars=["SeaLevel"]
        )

        assert (
            (p["SeaLevel_lower"] < p["SeaLevel"])
            & (p["SeaLevel"] < p["SeaLevel_upper"])
        ).values.all()


class TestGumbel:

    def test_ml_param(self, data_fre):
        p = fit(data_fre, dist="gumbel_r", method="ML", dim="Year", vars=["SeaLevel"])

        assert (
            (p["SeaLevel_lower"] < p["SeaLevel"])
            & (p["SeaLevel"] < p["SeaLevel_upper"])
        ).values.all()
        np.testing.assert_array_almost_equal(
            p.to_array().values,
            np.array(
                [
                    [1.4662802, 0.13941243],
                    [1.43506132, 0.11969685],
                    [1.49749908, 0.1623754],
                ]
            ),
        )

    def test_pwm_param(self, data_fre):
        p = fit(data_fre, dist="gumbel_r", method="PWM", dim="Year", vars=["SeaLevel"])

        assert (
            (p["SeaLevel_lower"] < p["SeaLevel"])
            & (p["SeaLevel"] < p["SeaLevel_upper"])
        ).values.all()
        np.testing.assert_array_almost_equal(
            p.to_array().values[0], np.array([1.46903519, 0.1195187])
        )

    def test_bayes_param(self, data_fre):
        p = fit(
            data_fre, dist="gumbel_r", method="BAYES", dim="Year", vars=["SeaLevel"]
        )

        assert (
            (p["SeaLevel_lower"] < p["SeaLevel"])
            & (p["SeaLevel"] < p["SeaLevel_upper"])
        ).values.all()

    def test_ml_rtnlv(self, data_fre):
        p = return_level(
            data_fre, dist="gumbel_r", method="ML", dim="Year", vars=["SeaLevel"]
        )

        assert (
            (p["SeaLevel_lower"] < p["SeaLevel"])
            & (p["SeaLevel"] < p["SeaLevel_upper"])
        ).values.all()
        np.testing.assert_array_almost_equal(
            p.to_array().values, np.array([[2.10759817], [1.99555233], [2.21964401]])
        )

    def test_pwm_rtnlv(self, data_fre):
        p = return_level(
            data_fre, dist="gumbel_r", method="PWM", dim="Year", vars=["SeaLevel"]
        )

        np.testing.assert_array_almost_equal(
            p.to_array().values[0], np.array([2.01883904])
        )
        assert (
            (p["SeaLevel_lower"] < p["SeaLevel"])
            & (p["SeaLevel"] < p["SeaLevel_upper"])
        ).values.all()

    def test_bayes_rtnlv(self, data_fre):
        p = return_level(
            data_fre, dist="gumbel_r", method="BAYES", dim="Year", vars=["SeaLevel"]
        )

        assert (
            (p["SeaLevel_lower"] < p["SeaLevel"])
            & (p["SeaLevel"] < p["SeaLevel_upper"])
        ).values.all()


class TestPareto:

    def test_ml_param(self, data_rain):
        p = fit(
            data_rain, dist="genpareto", method="ML", dim="Date", vars=["Exceedance"]
        )

        assert (
            (p["Exceedance_lower"] < p["Exceedance"])
            & (p["Exceedance"] < p["Exceedance_upper"])
        ).values.all()
        np.testing.assert_array_almost_equal(
            p.to_array().values,
            np.array(
                [
                    [7.44019083, 0.1844927],
                    [5.77994415, -0.01385753],
                    [9.57733124, 0.38284292],
                ]
            ),
        )

    def test_pwm_param(self, data_rain):
        p = fit(
            data_rain, dist="genpareto", method="PWM", dim="Date", vars=["Exceedance"]
        )

        assert (
            (p["Exceedance_lower"] < p["Exceedance"])
            & (p["Exceedance"] < p["Exceedance_upper"])
        ).values.all()
        np.testing.assert_array_almost_equal(
            p.to_array().values[0], np.array([7.29901897, 0.19651587])
        )

    def test_bayes_param(self, data_rain):
        p = p = fit(
            data_rain, dist="genpareto", method="BAYES", vars=["Exceedance"], dim="Date"
        )

        assert (
            (p["Exceedance_lower"] < p["Exceedance"])
            & (p["Exceedance"] < p["Exceedance_upper"])
        ).values.all()

    def test_ml_rtnlv(self, data_rain):
        p = return_level(
            data_rain,
            dist="genpareto",
            method="ML",
            dim="Date",
            vars=["Exceedance"],
            threshold_pareto=30,
            nobs_pareto=17531,
            nobsperblock_pareto=365,
        )

        assert (
            (p["Exceedance_lower"] < p["Exceedance"])
            & (p["Exceedance"] < p["Exceedance_upper"])
        ).values.all()
        np.testing.assert_array_almost_equal(
            p.to_array().values,
            np.array([[106.32558691], [65.48163774], [147.16953608]]),
        )

    def test_pwm_rtnlv(self, data_rain):
        p = return_level(
            data_rain,
            dist="genpareto",
            method="PWM",
            dim="Date",
            vars=["Exceedance"],
            threshold_pareto=30,
            nobs_pareto=17531,
            nobsperblock_pareto=365,
        )
        np.testing.assert_array_almost_equal(
            p.to_array().values[0], np.array([107.99657119])
        )
        assert (
            (p["Exceedance_lower"] < p["Exceedance"])
            & (p["Exceedance"] < p["Exceedance_upper"])
        ).values.all()

    def test_bayes_rtnlv(self, data_rain):
        p = return_level(
            data_rain,
            dist="genpareto",
            method="BAYES",
            dim="Date",
            vars=["Exceedance"],
            threshold_pareto=30,
            nobs_pareto=17531,
            nobsperblock_pareto=365,
        )

        assert (
            (p["Exceedance_lower"] < p["Exceedance"])
            & (p["Exceedance"] < p["Exceedance_upper"])
        ).values.all()


class TestError:

    def test_non_stationary_cov_pwm(self, data_fre):

        with pytest.raises(
            ValueError,
            match="Probability weighted moment parameter estimation cannot have covariates \\['Year', 'SOI', 'Year2', 'SOI2', 'Year3', 'SOI3'\\]",
        ):

            return_level(
                data_fre,
                dist="genextreme",
                method="pwm",
                vars=["SeaLevel"],
                dim="Year",
                locationcov=["Year", "SOI"],
                scalecov=["Year2", "SOI2"],
                shapecov=["Year3", "SOI3"],
            )

    def test_extremefit_rtnlv_error(self, data_rain):

        with pytest.raises(ValueError, match="Unrecognized distribution: XXX"):
            return_level(
                data_rain,
                dist="XXX",
                method="BAYES",
                dim="Date",
                vars=["Exceedance"],
                threshold_pareto=30,
                nobs_pareto=17531,
                nobsperblock_pareto=365,
            )

    def test_extremefit_param_error(self, data_fre):
        with pytest.raises(ValueError, match="Unrecognized method: YYY"):
            return_level(
                data_rain,
                dist="genpareto",
                method="YYY",
                dim="Date",
                vars=["Exceedance"],
                threshold_pareto=30,
                nobs_pareto=17531,
                nobsperblock_pareto=365,
            )

    def test_confinc_error(self, data_fre):
        with pytest.warns(
            UserWarning, match="There was an error in computing confidence interval."
        ):
            p = fit(
                data_fre, dist="genpareto", method="ML", vars=["SeaLevel"], dim="Year"
            )

        np.testing.assert_array_almost_equal(
            p.to_array().values,
            np.array([[2.87959631, -1.49978975], [np.nan, np.nan], [np.nan, np.nan]]),
        )

    def test_confinc_error_nostat(self):
        dataset = xr.Dataset(
            {"n": ("pos", [1, -1, 1, -1, 1, -1]), "n2": ("pos", [2, -2, 2, -2, 2, -2])},
            coords={"pos": [0, 1, 2, 3, 4, -5]},
        )
        with pytest.warns(
            UserWarning,
            match="There was an error in fitting the data to a genpareto distribution using ML. "
            "Returned parameters are numpy.nan",
        ):

            p = fit(
                dataset,
                dist="genpareto",
                method="ML",
                dim="pos",
                scalecov=["n2"],
                vars=["n"],
            )
            np.testing.assert_array_equal(
                p.to_array().values,
                np.array(
                    [
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                    ]
                ),
            )

    def test_confinc_error_nostat_rtnlv(self):
        dataset = xr.Dataset(
            {"n": ("pos", [1, -1, 1, -1, 1, -1]), "n2": ("pos", [2, -2, 2, -2, 2, -2])},
            coords={"pos": [0, 1, 2, 3, 4, -5]},
        )
        with pytest.warns(
            UserWarning,
            match="There was an error in fitting the data to a genpareto distribution using BAYES. "
            "Returned parameters are numpy.nan",
        ):

            p = fit(
                dataset,
                dist="genpareto",
                method="BAYES",
                dim="pos",
                scalecov=["n2"],
                vars=["n"],
            )

            np.testing.assert_array_equal(
                p.to_array().values,
                np.array(
                    [
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                    ]
                ),
            )

    def test_rtnlv_error_nostat_rtnlv(self):
        dataset = xr.Dataset(
            {"n": ("pos", [1, -1, 1, -1, 1, -1]), "n2": ("pos", [2, -2, 2, -2, 2, -2])},
            coords={"pos": [0, 1, 2, 3, 4, -5]},
        )
        with pytest.warns(
            UserWarning,
            match="There was an error in fitting the data to a genextreme distribution using BAYES. "
            "Returned parameters are numpy.nan",
        ):

            p = return_level(
                dataset,
                dist="genextreme",
                method="BAYES",
                dim="pos",
                scalecov=["n2"],
                vars=["n"],
            )

            np.testing.assert_array_equal(
                p.to_array().values,
                np.array(
                    [
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
            )

    def test_dist_error(self, data_fre):
        with pytest.raises(ValueError, match="Unrecognized distribution: XXX"):
            fit(data_fre, dist="XXX", method="ml", vars=["SeaLevel"], dim="Year")

    def test_method_error(self, data_fre):
        with pytest.raises(ValueError, match="Unrecognized method: YYY"):
            fit(
                data_fre, dist="genextreme", method="YYY", vars=["SeaLevel"], dim="Year"
            )

    def test_vars_error(self, data_fre):
        with pytest.raises(
            ValueError,
            match="XXXX is not a variable in the Dataset. Dataset's variables are: \\['SeaLevel', 'SOI'\\]",
        ):
            fit(data_fre, dist="genextreme", vars=["XXXX"], dim="Year")

    def test_conflev_error(self, data_fre):
        with pytest.raises(
            ValueError,
            match="Confidence level must be strictly smaller than 1 and strictly larger than 0",
        ):
            fit(
                data_fre,
                dist="genextreme",
                vars=["SeaLevel"],
                dim="Year",
                confidence_level=2,
            )

    def test_gumbel_cov_error(self, data_fre):
        with pytest.raises(
            ValueError,
            match="Gumbel distribution has no shape parameter and thus cannot have shape covariates \\['Year'\\]",
        ):
            fit(
                data_fre,
                dist="gumbel_r",
                method="ml",
                vars=["SeaLevel"],
                dim="Year",
                shapecov=["Year"],
            )

    def test_pareto_cov_error(self, data_fre):
        with pytest.raises(
            ValueError,
            match="Pareto distribution has no location parameter and thus cannot have location covariates \\['Year'\\]",
        ):
            return_level(
                data_fre,
                dist="genpareto",
                method="ml",
                vars=["SeaLevel"],
                dim="Year",
                locationcov=["Year"],
            )

    def test_rtmper_error(self, data_fre):
        with pytest.raises(
            ValueError,
            match="Return period has to be strictly larger than 0. Current return period value is -1",
        ):
            return_level(
                data_fre,
                dist="genextreme",
                vars=["SeaLevel"],
                dim="Year",
                return_period=-1,
            )

    def test_para_pareto_error(self, data_rain):
        with pytest.raises(
            ValueError,
            match="'threshold_pareto', 'nobs_pareto', and 'nobsperblock_pareto' must be defined when using dist 'genpareto'.",
        ):
            return_level(
                data_rain,
                dist="genpareto",
                method="ml",
                vars=["Exceedance"],
                dim="Date",
            )
