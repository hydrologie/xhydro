import warnings
from pathlib import Path

import leafmap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xdatasets as xd
from pystac_client.exceptions import APIError
from requests.exceptions import HTTPError

import xhydro as xh


class TestWatershedDelineation:
    m = leafmap.Map(center=(48.63, -74.71), zoom=5, basemap="USGS Hydrography")

    @pytest.mark.parametrize(
        "lng_lat, area",
        [
            ((-73.118597, 46.042467), 23933972552.885937),
            ((-66.153789, 50.265321), 18891676494.940426),
        ],
    )
    def test_watershed_delineation_from_coords(self, lng_lat, area):
        gdf = xh.gis.watershed_delineation(coordinates=lng_lat)
        np.testing.assert_allclose(
            [gdf.to_crs(32198).area.values[0]],
            [area],
            rtol=1e-5,  # FIXME: pip gives slightly different results than conda env
        )

    @pytest.mark.parametrize("area", [18891676494.940426])
    def test_watershed_delineation_from_map(self, area):
        # Richelieu watershed
        self.m.draw_features = [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "Point", "coordinates": [-66.153789, 50.265321]},
            }
        ]
        gdf = xh.gis.watershed_delineation(m=self.m)
        np.testing.assert_allclose(
            [gdf.to_crs(32198).area.values[0]],
            [area],
            rtol=1e-5,  # FIXME: pip gives slightly different results than conda env
        )

    def test_errors(self):
        bad_coordinates = (-35.0, 45.0)
        with pytest.warns(
            UserWarning,
            match=warnings.warn(f"Could not return a watershed boundary for coordinates {bad_coordinates}.", stacklevel=2),
        ):
            xh.gis.watershed_delineation(coordinates=bad_coordinates)

        with pytest.raises(
            ValueError,
            match="Either coordinates or a map with markers must be provided",
        ):
            xh.gis.watershed_delineation()


class TestWatershedOperations:
    gdf = xd.Query(
        **{
            "datasets": {
                "deh_polygons": {
                    "id": ["031501", "042103"],
                    "regulated": ["Natural"],
                }
            }
        }
    ).data.reset_index()

    @pytest.fixture
    def watershed_properties_data(self):
        # Computed using EPSG:6622
        data = {
            "Station": {0: "031501", 1: "042103"},
            "Superficie": {0: 21.868619918823242, 1: 579.4796142578125},
            "area (m2)": {0: 21868619.035204668, 1: 579479639.8084792},
            "perimeter (m)": {0: 27186.996844559395, 1: 283765.05839030433},
            "gravelius (m/m)": {0: 1.6400067113344199, 1: 3.3253311032394937},
            "centroid_lon": {
                0: -72.48631199105834,
                1: -78.37036445281987,
            },
            "centroid_lat": {
                0: 46.22277542928622,
                1: 46.48287117609677,
            },
        }

        df = pd.DataFrame.from_dict(data)
        return df

    def test_watershed_properties(self, watershed_properties_data):
        _properties_name = [
            "area (m2)",
            "perimeter (m)",
            "gravelius (m/m)",
            "centroid_lon",
            "centroid_lat",
        ]

        df_properties = xh.gis.watershed_properties(self.gdf, projected_crs=6622)

        pd.testing.assert_frame_equal(df_properties[_properties_name], watershed_properties_data[_properties_name])

        df_properties_def = xh.gis.watershed_properties(self.gdf)
        pd.testing.assert_frame_equal(
            df_properties_def[_properties_name],
            df_properties[_properties_name],
            rtol=0.02,
        )

    def test_watershed_properties_unique_id(self, watershed_properties_data):
        _properties_name = [
            "area (m2)",
            "perimeter (m)",
            "gravelius (m/m)",
            "centroid_lon",
            "centroid_lat",
        ]
        unique_id = "Station"

        df_properties = xh.gis.watershed_properties(self.gdf, unique_id=unique_id, projected_crs=6622)

        pd.testing.assert_frame_equal(
            df_properties[_properties_name],
            watershed_properties_data.set_index(unique_id)[_properties_name],
        )

    @pytest.mark.parametrize("unique_id", ["Station", None])
    def test_watershed_properties_xarray(self, watershed_properties_data, unique_id):
        ds_properties = xh.gis.watershed_properties(self.gdf, unique_id=unique_id, output_format="xarray", projected_crs=6622)

        unique_id = "Station" if unique_id is not None else "index"

        assert ds_properties.area.attrs["units"] == "m2"
        assert ds_properties.perimeter.attrs["units"] == "m"
        assert ds_properties.gravelius.attrs["units"] == "m/m"
        assert ds_properties.centroid_lon.attrs["units"] == "degrees_east"
        assert ds_properties.centroid_lat.attrs["units"] == "degrees_north"
        assert ds_properties.estimated_area_diff.attrs["units"] == "%"
        assert ds_properties.sizes == {unique_id: 2}

        if unique_id == "Station":
            output_dataset = watershed_properties_data.set_index(unique_id)
        else:
            output_dataset = watershed_properties_data
        output_dataset = output_dataset.to_xarray()
        output_dataset = output_dataset.rename(
            {
                "area (m2)": "area",
                "perimeter (m)": "perimeter",
                "gravelius (m/m)": "gravelius",
            }
        )
        output_dataset["area"].attrs = {"units": "m2"}
        output_dataset["perimeter"].attrs = {"units": "m"}
        output_dataset["gravelius"].attrs = {"units": "m/m"}
        output_dataset["centroid_lon"].attrs = {"units": "degrees_east"}
        output_dataset["centroid_lat"].attrs = {"units": "degrees_north"}

        xr.testing.assert_allclose(ds_properties[[v for v in output_dataset.data_vars]], output_dataset)

    def test_errors(self):
        with pytest.warns(
            UserWarning,
            match="The area calculated from your original source differs",
        ):
            gdf = xh.gis.watershed_delineation(coordinates=(-71.28878, 46.65692))
            xh.gis.watershed_properties(gdf)


class TestSurfaceProperties:
    gdf = xd.Query(
        **{
            "datasets": {
                "deh_polygons": {
                    "id": ["031501", "042103"],
                    "regulated": ["Natural"],
                }
            }
        }
    ).data.reset_index()

    @pytest.fixture
    def surface_properties_data(self):
        # Computed using EPSG:6622
        data = {
            "elevation": {"031501": 45.47, "042103": 358.6},
            "slope": {"031501": 0.4574, "042103": 2.504},
            "aspect": {"031501": 250.6, "042103": 178.3},
        }

        df = pd.DataFrame.from_dict(data).astype("float32")
        df.index.names = ["Station"]
        return df

    @pytest.mark.online
    @pytest.mark.xfail(
        reason="Test is sometimes rate-limited by Microsoft Planetary Computer API.",
        strict=False,
        raises=APIError,
    )
    @pytest.mark.xfail(
        reason="Test may fail with if the server is down.",
        strict=False,
        raises=HTTPError,
        match="404 Client Error",
    )
    def test_surface_properties(self, surface_properties_data):
        _properties_name = ["elevation", "slope", "aspect"]

        df_properties = xh.gis.surface_properties(self.gdf, projected_crs=6622)
        df_properties.index.name = None

        pd.testing.assert_frame_equal(
            df_properties[_properties_name],
            surface_properties_data.reset_index(drop=True)[_properties_name],
            rtol=0.02,
        )

        df_properties_def = xh.gis.surface_properties(self.gdf)
        df_properties_def.index.name = None
        pd.testing.assert_frame_equal(
            df_properties_def[_properties_name],
            df_properties[_properties_name],
            rtol=0.025,
        )

    @pytest.mark.online
    @pytest.mark.xfail(
        reason="Test is sometimes rate-limited by Microsoft Planetary Computer API.",
        strict=False,
        raises=APIError,
    )
    @pytest.mark.xfail(
        reason="Test may fail with if the server is down.",
        strict=False,
        raises=HTTPError,
        match="404 Client Error",
    )
    def test_surface_properties_unique_id(self, surface_properties_data):
        _properties_name = ["elevation", "slope", "aspect"]
        unique_id = "Station"

        df_properties = xh.gis.surface_properties(self.gdf, unique_id=unique_id, projected_crs=6622)

        pd.testing.assert_frame_equal(
            df_properties[_properties_name],
            surface_properties_data[_properties_name],
            rtol=0.02,
        )

    @pytest.mark.online
    @pytest.mark.xfail(
        reason="Test is sometimes rate-limited by Microsoft Planetary Computer API.",
        strict=False,
        raises=APIError,
    )
    @pytest.mark.xfail(
        reason="Test may fail with if the server is down.",
        strict=False,
        raises=HTTPError,
        match="404 Client Error",
    )
    def test_surface_properties_xarray(self, surface_properties_data):
        unique_id = "Station"

        ds_properties = xh.gis.surface_properties(self.gdf, unique_id=unique_id, output_format="xarray", projected_crs=6622)
        ds_properties = ds_properties.drop_vars(list(set(ds_properties.coords) - set(ds_properties.dims)))

        assert ds_properties.elevation.attrs["units"] == "m"
        assert ds_properties.slope.attrs["units"] == "degrees"
        assert ds_properties.aspect.attrs["units"] == "degrees"

        output_dataset = surface_properties_data.to_xarray()
        output_dataset["elevation"].attrs = {"units": "m"}
        output_dataset["slope"].attrs = {"units": "degrees"}
        output_dataset["aspect"].attrs = {"units": "degrees"}

        xr.testing.assert_allclose(ds_properties, output_dataset, rtol=0.02)


@pytest.mark.online
@pytest.mark.xfail(
    reason="Test is sometimes rate-limited by Microsoft Planetary Computer API.",
    strict=False,
    raises=APIError,
)
@pytest.mark.xfail(
    reason="Test may fail with if the server is down.",
    strict=False,
    raises=HTTPError,
    match="404 Client Error",
)
class TestLandClassification:
    gdf = xd.Query(
        **{
            "datasets": {
                "deh_polygons": {
                    "id": ["031501", "042103"],
                    "regulated": ["Natural"],
                }
            }
        }
    ).data.reset_index()

    @pytest.fixture
    def land_classification_data_latest(self):
        data = {
            "pct_built_area": {
                "031501": 0.015609,
                "042103": 1.6e-05,
            },
            "pct_crops": {"031501": 0.716741, "042103": 0.0},
            "pct_trees": {
                "031501": 0.259239,
                "042103": 0.909270,
            },
            "pct_rangeland": {
                "031501": 0.008410,
                "042103": 0.004850,
            },
            "pct_water": {"031501": 0.0, "042103": 0.085444},
            "pct_flooded_vegetation": {"031501": 0.0, "042103": 0.000417},
            "pct_bare_ground": {"031501": 0.0, "042103": 4e-06},
        }

        df = pd.DataFrame.from_dict(data)
        df.index.name = "Station"
        return df

    @pytest.fixture
    def land_classification_data_2018(self):
        data = {
            "pct_built_area": {
                "031501": 0.016057,
                "042103": 3.906517e-05,
            },
            "pct_crops": {"031501": 0.723301, "042103": 0.0},
            "pct_trees": {
                "031501": 0.256347,
                "042103": 0.9106647,
            },
            "pct_rangeland": {
                "031501": 0.004294,
                "042103": 0.004358778,
            },
            "pct_water": {"031501": 0.0, "042103": 0.08474285},
            "pct_flooded_vegetation": {"031501": 0.0, "042103": 0.0001939491},
            "pct_bare_ground": {"031501": 0.0, "042103": 6.883730e-07},
        }

        df = pd.DataFrame.from_dict(data)
        df.index.name = "Station"
        return df

    @pytest.mark.parametrize("year", ["latest", "2018"])
    def test_land_classification(self, land_classification_data_latest, land_classification_data_2018, year):
        if year == "latest":
            df_expected = land_classification_data_latest
        elif year == "2018":
            df_expected = land_classification_data_2018
        else:
            raise ValueError(f"Invalid year argument {year}.")

        for unique_id in ["Station", None]:
            df = xh.gis.land_use_classification(self.gdf, unique_id=unique_id, year=year)
            if unique_id is None:
                df_expected = df_expected.reset_index(drop=True)

            df = df[df_expected.columns]  # Reorder the columns
            pd.testing.assert_frame_equal(df, df_expected, check_exact=False, atol=0.0001)

    @pytest.mark.parametrize("year", ["latest", "2018"])
    def test_land_classification_xarray(self, land_classification_data_latest, land_classification_data_2018, year):
        for unique_id in ["Station", None]:
            if year == "latest":
                df_expected = land_classification_data_latest
            elif year == "2018":
                df_expected = land_classification_data_2018

            else:
                raise ValueError(f"Invalid year argument {year}.")

            if unique_id is None:
                df_expected = df_expected.reset_index(drop=True)

            ds_expected = df_expected.to_xarray()

            ds_classification = xh.gis.land_use_classification(
                self.gdf,
                unique_id=unique_id,
                year=year,
                output_format="xarray",
            )

            for var in ds_classification:
                assert ds_classification[var].attrs["units"] == "percent"

            for var in ds_expected:
                ds_expected[var].attrs = {"units": "percent"}
            if year == "latest":
                ds_expected.attrs = {
                    "year": "2023",
                    "collection": "io-lulc-annual-v02",
                    "spatial_resolution": 10,
                }
            elif year == "2018":
                ds_expected.attrs = {
                    "year": "2018",
                    "collection": "io-lulc-annual-v02",
                    "spatial_resolution": 10,
                }

            for var in ds_classification:
                np.testing.assert_allclose(ds_classification[var], ds_expected[var], atol=0.0001)

    @pytest.mark.parametrize("unique_id,", ["Station", None])
    def test_land_classification_plot(self, unique_id, monkeypatch):
        monkeypatch.setattr(plt, "show", lambda: None)
        xh.gis.land_use_plot(self.gdf, unique_id=unique_id, idx=0)

    def test_errors(self):
        with pytest.raises(
            ValueError,
            match="The provided gpd.GeoDataFrame is missing the crs attribute.",
        ):
            gdf_no_crs = self.gdf.copy()
            gdf_no_crs.crs = None  # Will raise a warning. This can't be helped.
            xh.gis.watershed_properties(gdf_no_crs)
        with pytest.raises(
            TypeError,
            match="Expected year argument foo to be a digit.",
        ):
            xh.gis.land_use_classification(self.gdf, unique_id="Station", year="foo")
        with pytest.raises(
            TypeError,
            match="Expected year argument None to be a digit.",
        ):
            xh.gis.land_use_classification(self.gdf, unique_id="Station", year=None)
        with pytest.raises(
            TypeError,
            match="Expected year argument None to be a digit.",
        ):
            xh.gis.land_use_plot(self.gdf, unique_id="Station", idx=0, year=None)


@pytest.mark.online
@pytest.mark.xfail(
    reason="Test is sometimes rate-limited by Microsoft Planetary Computer API.",
    strict=False,
    raises=APIError,
)
@pytest.mark.xfail(
    reason="Test may fail with if the server is down.",
    strict=False,
    raises=HTTPError,
    match="404 Client Error",
)
class TestToRaven:
    @pytest.mark.parametrize("data", ["coord", "gdf", "file"])
    def test_coords(self, data, tmp_path):
        if data == "coord":
            data = (-73.118597, 46.042467)
        elif data == "gdf":
            data = xh.gis.watershed_delineation(coordinates=(-73.118597, 46.042467))
        elif data == "file":
            gdf = xh.gis.watershed_delineation(coordinates=(-73.118597, 46.042467))
            gdf.to_file(str(Path(tmp_path) / "test.gpkg"), index="HYBAS_ID")
            data = str(Path(tmp_path) / "test.gpkg")

        out = xh.gis.watershed_to_raven_hru(data, unique_id="HYBAS_ID" if not isinstance(data, tuple) else None)

        assert all(
            col in out.columns
            for col in [
                "HRU_ID",
                "geometry",
                "area",
                "latitude",
                "longitude",
                "elevation",
                "SubId",
                "DowSubId",
            ]
        )
        assert out.crs == "EPSG:4326"

    def test_error(self):
        data = xh.gis.watershed_delineation(coordinates=[(-73.118597, 46.042467), (-66.153789, 50.265321)])
        with pytest.raises(
            ValueError,
            match="The input must be a single watershed",
        ):
            xh.gis.watershed_to_raven_hru(data)

        with pytest.warns(
            UserWarning,
            match="The unique_id argument is ignored when using coordinates to delineate a watershed.",
        ):
            xh.gis.watershed_to_raven_hru((-73.118597, 46.042467), unique_id="foo")
