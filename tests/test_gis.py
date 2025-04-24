import warnings

import leafmap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xdatasets as xd
from pystac_client.exceptions import APIError

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
        gdf = xh.gis.watershed_delineation(map=self.m)
        np.testing.assert_allclose(
            [gdf.to_crs(32198).area.values[0]],
            [area],
            rtol=1e-5,  # FIXME: pip gives slightly different results than conda env
        )

    def test_errors(self):
        bad_coordinates = (-35.0, 45.0)
        with pytest.warns(
            UserWarning,
            match=warnings.warn(
                f"Could not return a watershed boundary for coordinates {bad_coordinates}."
            ),
        ):
            xh.gis.watershed_delineation(coordinates=bad_coordinates)


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
        data = {
            "Station": {0: "031501", 1: "042103"},
            "Superficie": {0: 21.868619918823242, 1: 579.4796142578125},
            "area": {0: 21868619.035204668, 1: 579479639.8084792},
            "perimeter": {0: 27186.996844559395, 1: 283765.05839030433},
            "gravelius": {0: 1.6400067113344199, 1: 3.3253311032394937},
            "centroid": {
                0: (-72.48631199105834, 46.22277542928622),
                1: (-78.37036445281987, 46.48287117609677),
            },
        }

        df = pd.DataFrame.from_dict(data)
        return df

    def test_watershed_properties(self, watershed_properties_data):
        _properties_name = ["area", "perimeter", "gravelius", "centroid"]

        df_properties = xh.gis.watershed_properties(self.gdf)

        pd.testing.assert_frame_equal(
            df_properties[_properties_name], watershed_properties_data[_properties_name]
        )

    def test_watershed_properties_unique_id(self, watershed_properties_data):
        _properties_name = ["area", "perimeter", "gravelius", "centroid"]
        unique_id = "Station"

        df_properties = xh.gis.watershed_properties(self.gdf, unique_id=unique_id)

        pd.testing.assert_frame_equal(
            df_properties[_properties_name],
            watershed_properties_data.set_index(unique_id)[_properties_name],
        )

    def test_watershed_properties_xarray(self, watershed_properties_data):
        unique_id = "Station"

        ds_properties = xh.gis.watershed_properties(
            self.gdf, unique_id=unique_id, output_format="xarray"
        )

        assert ds_properties.area.attrs["units"] == "m2"
        assert ds_properties.perimeter.attrs["units"] == "m"
        assert ds_properties.gravelius.attrs["units"] == "m/m"
        assert ds_properties.centroid.attrs["units"] == ("degree_east", "degree_north")

        output_dataset = watershed_properties_data.set_index(unique_id).to_xarray()
        output_dataset["area"].attrs = {"units": "m2"}
        output_dataset["perimeter"].attrs = {"units": "m"}
        output_dataset["gravelius"].attrs = {"units": "m/m"}
        output_dataset["centroid"].attrs = {"units": ("degree_east", "degree_north")}

        xr.testing.assert_allclose(ds_properties, output_dataset)


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
        data = {
            "elevation": {"031501": 46.3385009765625, "042103": 358.54986572265625},
            "slope": {"031501": 0.4634914696216583, "042103": 2.5006439685821533},
            "aspect": {"031501": 241.46539306640625, "042103": 178.55764770507812},
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
    def test_surface_properties(self, surface_properties_data):
        _properties_name = ["elevation", "slope", "aspect"]

        df_properties = xh.gis.surface_properties(self.gdf)
        df_properties.index.name = None

        pd.testing.assert_frame_equal(
            df_properties[_properties_name],
            surface_properties_data.reset_index(drop=True)[_properties_name],
        )

    @pytest.mark.online
    @pytest.mark.xfail(
        reason="Test is sometimes rate-limited by Microsoft Planetary Computer API.",
        strict=False,
        raises=APIError,
    )
    def test_surface_properties_unique_id(self, surface_properties_data):
        _properties_name = ["elevation", "slope", "aspect"]
        unique_id = "Station"

        df_properties = xh.gis.surface_properties(self.gdf, unique_id=unique_id)

        pd.testing.assert_frame_equal(
            df_properties[_properties_name],
            surface_properties_data[_properties_name],
        )

    @pytest.mark.online
    @pytest.mark.xfail(
        reason="Test is sometimes rate-limited by Microsoft Planetary Computer API.",
        strict=False,
        raises=APIError,
    )
    def test_surface_properties_xarray(self, surface_properties_data):
        unique_id = "Station"

        ds_properties = xh.gis.surface_properties(
            self.gdf, unique_id=unique_id, output_format="xarray"
        )
        ds_properties = ds_properties.drop(
            list(set(ds_properties.coords) - set(ds_properties.dims))
        )

        assert ds_properties.elevation.attrs["units"] == "m"
        assert ds_properties.slope.attrs["units"] == "degrees"
        assert ds_properties.aspect.attrs["units"] == "degrees"

        output_dataset = surface_properties_data.to_xarray()
        output_dataset["elevation"].attrs = {"units": "m"}
        output_dataset["slope"].attrs = {"units": "degrees"}
        output_dataset["aspect"].attrs = {"units": "degrees"}

        xr.testing.assert_allclose(ds_properties, output_dataset)


@pytest.mark.online
@pytest.mark.xfail(
    reason="Test is sometimes rate-limited by Microsoft Planetary Computer API.",
    strict=False,
    raises=APIError,
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
                "031501": 0.0153389,
                "042103": 1.61770889e-05,
            },
            "pct_crops": {"031501": 0.71674722, "042103": 0.0},
            "pct_trees": {
                "031501": 0.25940324,
                "042103": 0.90926391,
            },
            "pct_rangeland": {
                "031501": 0.0084097,
                "042103": 0.0048533,
            },
            "pct_water": {"031501": 0.0, "042103": 0.08544583},
            "pct_flooded_vegetation": {"031501": 0.0, "042103": 0.00041716},
            "pct_bare_ground": {"031501": 0.0, "042103": 3.61403049e-06},
        }

        df = pd.DataFrame.from_dict(data)
        df.index.name = "Station"
        return df

    @pytest.fixture
    def land_classification_data_2018(self):
        data = {
            "pct_built_area": {
                "031501": 0.01583815,
                "042103": 3.87217553e-05,
            },
            "pct_crops": {"031501": 0.72329773, "042103": 0.0},
            "pct_trees": {
                "031501": 0.2565528,
                "042103": 0.91066581,
            },
            "pct_rangeland": {
                "031501": 0.00429392,
                "042103": 0.00435766,
            },
            "pct_water": {"031501": 0.0, "042103": 0.08474334},
            "pct_flooded_vegetation": {"031501": 0.0, "042103": 0.00019395},
            "pct_bare_ground": {"031501": 0.0, "042103": 5.1629007e-07},
        }

        df = pd.DataFrame.from_dict(data)
        df.index.name = "Station"
        return df

    @pytest.mark.xfail(
        raises=APIError,
        reason="Test is sometimes rate-limited by Microsoft Planetary Computer API.",
        strict=False,
    )
    @pytest.mark.parametrize("year,", ["latest", "2018"])
    def test_land_classification(
        self, land_classification_data_latest, land_classification_data_2018, year
    ):
        if year == "latest":
            df_expected = land_classification_data_latest
        elif year == "2018":
            df_expected = land_classification_data_2018
        else:
            raise ValueError(f"Invalid year argument {year}.")

        for unique_id in ["Station", None]:
            df = xh.gis.land_use_classification(
                self.gdf, unique_id=unique_id, year=year
            )
            if unique_id is None:
                df_expected = df_expected.reset_index(drop=True)

            pd.testing.assert_frame_equal(
                df, df_expected, check_exact=False, atol=0.0001
            )

    @pytest.mark.parametrize("year,", ["latest", "2018"])
    def test_land_classification_xarray(
        self, land_classification_data_latest, land_classification_data_2018, year
    ):
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

            xr.testing.assert_allclose(ds_classification, ds_expected, atol=0.0001)

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
            gdf_no_crs.crs = None
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
