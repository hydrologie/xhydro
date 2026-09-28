import importlib.util
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pooch
import pytest
import xarray as xr
from pystac_client.exceptions import APIError
from requests.exceptions import HTTPError

import xhydro as xh


HAS_LEAFMAP = bool(importlib.util.find_spec("leafmap"))


class TestWatershedDelineation:
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
    @pytest.mark.skipif(not HAS_LEAFMAP, reason="The `leafmap` library is not present in the environment.")
    def test_watershed_delineation_from_map(self, area):
        import leafmap

        m = leafmap.Map(center=(48.63, -74.71), zoom=5, basemap="USGS Hydrography")
        # Richelieu watershed
        m.draw_features = [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "Point", "coordinates": [-66.153789, 50.265321]},
            }
        ]
        gdf = xh.gis.watershed_delineation(m=m)
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
    @pytest.fixture
    def gdf(self, deveraux):
        gdf_files = deveraux.fetch("ravenpy/hru_subset.zip", pooch.Unzip())
        gdf = gpd.read_file([f for f in gdf_files if f.endswith(".shp")][0])
        gdf = gdf.iloc[[0, 100]]
        gdf["Superficie"] = [4.7, 0.6]
        return gdf.to_crs("EPSG:4326").reset_index(drop=True)

    @pytest.fixture
    def watershed_properties_data(self):
        # Computed using EPSG:6622
        data = {
            "SubId": {0: 3, 1: 29},
            "area (m2)": {0: 4744102.281834503, 1: 581212.0153725281},
            "perimeter (m)": {0: 15222.4104833989, 1: 9610.041755029599},
            "gravelius (m/m)": {0: 1.9715213066290012, 1: 3.5559287521610234},
            "centroid_lon": {
                0: -74.07916869661638,
                1: -74.69762970613249,
            },
            "centroid_lat": {
                0: 45.453774218543025,
                1: 45.68703704865826,
            },
        }

        df = pd.DataFrame.from_dict(data)
        return df

    def test_watershed_properties(self, gdf, watershed_properties_data):
        _properties_name = [
            "area (m2)",
            "perimeter (m)",
            "gravelius (m/m)",
            "centroid_lon",
            "centroid_lat",
        ]

        df_properties = xh.gis.watershed_properties(gdf, projected_crs=6622)

        pd.testing.assert_frame_equal(df_properties[_properties_name], watershed_properties_data[_properties_name])

        df_properties_def = xh.gis.watershed_properties(gdf)
        pd.testing.assert_frame_equal(
            df_properties_def[_properties_name],
            df_properties[_properties_name],
            rtol=0.02,
        )

    def test_watershed_properties_unique_id(self, gdf, watershed_properties_data):
        _properties_name = [
            "area (m2)",
            "perimeter (m)",
            "gravelius (m/m)",
            "centroid_lon",
            "centroid_lat",
        ]
        unique_id = "SubId"

        df_properties = xh.gis.watershed_properties(gdf, unique_id=unique_id, projected_crs=6622)

        pd.testing.assert_frame_equal(
            df_properties[_properties_name],
            watershed_properties_data.set_index(unique_id)[_properties_name],
        )

    @pytest.mark.parametrize("unique_id", ["SubId", None])
    def test_watershed_properties_xarray(self, gdf, watershed_properties_data, unique_id):
        ds_properties = xh.gis.watershed_properties(gdf, unique_id=unique_id, output_format="xarray", projected_crs=6622)

        unique_id = "SubId" if unique_id is not None else "index"

        assert ds_properties.area.attrs["units"] == "m2"
        assert ds_properties.perimeter.attrs["units"] == "m"
        assert ds_properties.gravelius.attrs["units"] == "m/m"
        assert ds_properties.centroid_lon.attrs["units"] == "degrees_east"
        assert ds_properties.centroid_lat.attrs["units"] == "degrees_north"
        assert ds_properties.estimated_area_diff.attrs["units"] == "%"
        assert ds_properties.sizes == {unique_id: 2}

        if unique_id == "SubId":
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
    @pytest.fixture
    def gdf(self, deveraux):
        gdf_files = deveraux.fetch("ravenpy/hru_subset.zip", pooch.Unzip())
        gdf = gpd.read_file([f for f in gdf_files if f.endswith(".shp")][0])
        gdf = gdf.iloc[[0, 100]]
        return gdf.to_crs("EPSG:4326").reset_index(drop=True)

    @pytest.fixture
    def surface_properties_data(self):
        # Computed using EPSG:6622
        data = {
            "elevation": {3: 23.089134, 29: 200.55365},
            "slope": {3: 0.2759844, 29: 1.9218631},
            "aspect": {3: 100.66728, 29: 170.05554},
        }

        df = pd.DataFrame.from_dict(data).astype("float32")
        df.index.names = ["SubId"]
        return df

    @pytest.mark.online
    @pytest.mark.xfail(reason="Test is sometimes rate-limited by Microsoft Planetary Computer API.", strict=False, raises=(APIError, HTTPError))
    def test_surface_properties(self, gdf, surface_properties_data):
        _properties_name = ["elevation", "slope", "aspect"]

        df_properties = xh.gis.surface_properties(gdf, projected_crs=6622)
        df_properties.index.name = None

        pd.testing.assert_frame_equal(
            df_properties[_properties_name],
            surface_properties_data.reset_index(drop=True)[_properties_name],
            rtol=0.02,
        )

        df_properties_def = xh.gis.surface_properties(gdf)
        df_properties_def.index.name = None
        # The default CRS may introduce slight differences in the computed surface properties for the watersheds used in the test.
        pd.testing.assert_frame_equal(
            df_properties_def[["elevation"]],
            df_properties[["elevation"]],
            atol=1.5,  # 1.5 m tolerance for elevation differences
        )
        pd.testing.assert_frame_equal(
            df_properties_def[["aspect"]],
            df_properties[["aspect"]],
            atol=7,  # 7 degrees tolerance for aspect differences
        )
        pd.testing.assert_frame_equal(
            df_properties_def[["slope"]],
            df_properties[["slope"]],
            atol=0.15,  # 0.15 degrees tolerance for slope differences
        )

    @pytest.mark.online
    @pytest.mark.xfail(reason="Test is sometimes rate-limited by Microsoft Planetary Computer API.", strict=False, raises=(APIError, HTTPError))
    def test_surface_properties_unique_id(self, gdf, surface_properties_data):
        _properties_name = ["elevation", "slope", "aspect"]
        unique_id = "SubId"

        df_properties = xh.gis.surface_properties(gdf, unique_id=unique_id, projected_crs=6622)

        pd.testing.assert_frame_equal(
            df_properties[_properties_name],
            surface_properties_data[_properties_name],
            rtol=0.02,
        )

    @pytest.mark.online
    @pytest.mark.xfail(reason="Test is sometimes rate-limited by Microsoft Planetary Computer API.", strict=False, raises=(APIError, HTTPError))
    def test_surface_properties_xarray(self, gdf, surface_properties_data):
        unique_id = "SubId"

        ds_properties = xh.gis.surface_properties(gdf, unique_id=unique_id, output_format="xarray", projected_crs=6622)
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
@pytest.mark.xfail(reason="Test is sometimes rate-limited by Microsoft Planetary Computer API.", strict=False, raises=(APIError, HTTPError))
class TestLandClassification:
    @pytest.fixture
    def gdf(self, deveraux):
        gdf_files = deveraux.fetch("ravenpy/hru_subset.zip", pooch.Unzip())
        gdf = gpd.read_file([f for f in gdf_files if f.endswith(".shp")][0])
        gdf = gdf.iloc[[0, 100]]
        return gdf.to_crs("EPSG:4326").reset_index(drop=True)

    @pytest.fixture
    def land_classification_data_latest(self):
        data = {
            "pct_built_area": {
                3: 0.011416490486257928,
                29: 0.0,
            },
            "pct_crops": {3: 0.0008245243128964059, 29: 0.0},
            "pct_trees": {
                3: 0.03511627906976744,
                29: 0.23982758620689656,
            },
            "pct_rangeland": {
                3: 0.0,
                29: 0.012758620689655173,
            },
            "pct_water": {3: 0.9526427061310783, 29: 0.7063793103448276},
            "pct_flooded_vegetation": {3: 0.0, 29: 0.04103448275862069},
        }

        df = pd.DataFrame.from_dict(data)
        df.index.name = "SubId"
        return df

    @pytest.fixture
    def land_classification_data_2018(self):
        data = {
            "pct_built_area": {
                3: 0.014143763213530655,
                29: 0.0,
            },
            "pct_crops": {3: 0.0009725158562367865, 29: 0.0},
            "pct_trees": {
                3: 0.03503171247357294,
                29: 0.296551724137931,
            },
            "pct_rangeland": {
                3: 0.0,
                29: 0.014827586206896552,
            },
            "pct_water": {3: 0.9498520084566596, 29: 0.6832758620689655},
            "pct_flooded_vegetation": {3: 0.0, 29: 0.005344827586206896},
        }

        df = pd.DataFrame.from_dict(data)
        df.index.name = "SubId"
        return df

    @pytest.mark.parametrize("year", ["latest", "2018"])
    def test_land_classification(self, gdf, land_classification_data_latest, land_classification_data_2018, year):
        if year == "latest":
            df_expected = land_classification_data_latest
        elif year == "2018":
            df_expected = land_classification_data_2018
        else:
            raise ValueError(f"Invalid year argument {year}.")

        for unique_id in ["SubId", None]:
            df = xh.gis.land_use_classification(gdf, unique_id=unique_id, year=year)
            if unique_id is None:
                df_expected = df_expected.reset_index(drop=True)

            df = df[df_expected.columns]  # Reorder the columns
            pd.testing.assert_frame_equal(df, df_expected, check_exact=False, atol=0.0001)

    @pytest.mark.parametrize("year", ["latest", "2018"])
    def test_land_classification_xarray(self, gdf, land_classification_data_latest, land_classification_data_2018, year):
        for unique_id in ["SubId", None]:
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
                gdf,
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

    @pytest.mark.parametrize("unique_id", ["SubId", None])
    def test_land_classification_plot(self, gdf, unique_id, monkeypatch):
        monkeypatch.setattr(plt, "show", lambda: None)
        xh.gis.land_use_plot(gdf, unique_id=unique_id, idx=0)

    def test_errors(self, gdf):
        with pytest.raises(
            ValueError,
            match="The provided gpd.GeoDataFrame is missing the crs attribute.",
        ):
            gdf_no_crs = gdf.copy()
            gdf_no_crs.crs = None  # Will raise a warning. This can't be helped.
            xh.gis.watershed_properties(gdf_no_crs)
        with pytest.raises(
            TypeError,
            match="Expected year argument foo to be a digit.",
        ):
            xh.gis.land_use_classification(gdf, unique_id="SubId", year="foo")
        with pytest.raises(
            TypeError,
            match="Expected year argument None to be a digit.",
        ):
            xh.gis.land_use_classification(gdf, unique_id="SubId", year=None)
        with pytest.raises(
            TypeError,
            match="Expected year argument None to be a digit.",
        ):
            xh.gis.land_use_plot(gdf, unique_id="SubId", idx=0, year=None)


@pytest.mark.online
@pytest.mark.xfail(reason="Test is sometimes rate-limited by Microsoft Planetary Computer API.", strict=False, raises=(APIError, HTTPError))
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
