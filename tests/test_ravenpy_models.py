import datetime as dt
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pooch
import pytest
import xarray as xr
from pystac_client.exceptions import APIError
from shapely import Polygon

import xhydro.gis as xhg
import xhydro.modelling as xhm
from xhydro.modelling import RavenpyModel

try:
    import ravenpy
except ImportError:
    ravenpy = None


@pytest.mark.skipif(ravenpy is None, reason="RavenPy is not installed.")
class TestRavenpyModels:
    # Get data from xhydro-testdata repo
    riviere_rouge_meteo = "ravenpy/ERA5_Riviere_Rouge_global.nc"

    # List of types of data provided to Raven in the meteo file
    data_type = ["TEMP_MAX", "TEMP_MIN", "PRECIP"]
    # Alternate names in the netcdf file (for variables that have different names, map to the name in the file).
    alt_names_meteo = {"TEMP_MIN": "tmin", "TEMP_MAX": "tmax", "PRECIP": "pr"}
    start_date = dt.datetime(1985, 1, 1)
    end_date = dt.datetime(1990, 1, 1)
    hru = {
        "area": 100.0,
        "elevation": 250.5,
        "latitude": 46.0,
        "longitude": -80.75,
    }

    # Station properties. Using the same as for the catchment, but could be different.
    meteo_station_properties = {
        "ALL": {"elevation": 250.5, "latitude": 46.0, "longitude": -80.75}
    }
    rain_snow_fraction = "RAINSNOW_DINGMAN"
    evaporation = "PET_PRIESTLEY_TAYLOR"

    def test_ravenpy_gr4jcn(self, deveraux):
        model_name = "GR4JCN"  # RavenPy already tests all emulators, so we primarily need to check that our call works.
        parameters = [0.529, -3.396, 407.29, 1.072, 16.9, 0.947]
        global_parameter = {"AVG_ANNUAL_SNOW": 30.00}

        rpm = RavenpyModel(
            model_name=model_name,
            parameters=parameters,
            hru=self.hru,
            start_date=self.start_date,
            end_date=self.end_date,
            meteo_file=deveraux.fetch(self.riviere_rouge_meteo),
            data_type=self.data_type,
            alt_names_meteo=self.alt_names_meteo,
            meteo_station_properties=self.meteo_station_properties,
            rain_snow_fraction=self.rain_snow_fraction,  # Test that we can add kwargs
            evaporation=self.evaporation,
            global_parameter=global_parameter,
        )

        qsim = rpm.run()
        assert qsim["q"].shape == (1827,)
        np.testing.assert_almost_equal(
            qsim["q"].values[100:105],
            [0.02460463, 0.02451642, 0.02442908, 0.0243426, 0.0289188],
            decimal=5,
        )

        qsim2 = rpm.get_streamflow()
        assert qsim.equals(qsim2)

        met = rpm.get_inputs()
        assert len(met.time) == 6576
        met = rpm.get_inputs(subset_time=True)
        np.testing.assert_array_equal(met.time, qsim.time)
        assert all(var in met.variables for var in self.alt_names_meteo.values())

    @pytest.mark.online
    @pytest.mark.xfail(
        reason="Test is sometimes rate-limited by Microsoft Planetary Computer API.",
        strict=False,
        raises=APIError,
    )
    def test_ravenpy_from_funcs(self, deveraux, tmp_path):
        meteo = xr.open_dataset(deveraux.fetch(self.riviere_rouge_meteo))
        meteo, cfg = xhm.format_input(
            meteo, model="GR4JCN", save_as=tmp_path / "test.nc"
        )
        hru = xhg.watershed_to_raven_hru((-80.75, 46.0))

        model_config = {
            "model_name": "GR4JCN",
            "parameters": [0.529, -3.396, 407.29, 1.072, 16.9, 0.947],
            "hru": hru,
            "start_date": "1985-01-01",
            "end_date": "1990-01-01",
            "workdir": tmp_path,
            "meteo_station_properties": self.meteo_station_properties,
            **cfg,
        }
        qsim = xhm.hydrological_model(model_config).run()
        assert qsim["q"].shape == (1827,)
        np.testing.assert_almost_equal(
            qsim["q"].values[100:105],
            [4.36429443, 4.34596289, 4.32373373, 4.30464389, 5.13941455],
            decimal=5,
        )

    def test_overwrite(self, deveraux, tmp_path):
        model_config = {
            "model_name": "GR4JCN",
            "parameters": [0.529, -3.396, 407.29, 1.072, 16.9, 0.947],
            "global_parameter": {"AVG_ANNUAL_SNOW": 30.00},
            "hru": self.hru,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "meteo_file": deveraux.fetch(self.riviere_rouge_meteo),
            "data_type": self.data_type,
            "alt_names_meteo": self.alt_names_meteo,
            "meteo_station_properties": self.meteo_station_properties,
            "workdir": tmp_path,
        }

        rpm = xhm.hydrological_model(model_config)
        rpm.run()

        # Try to run the model again
        with pytest.raises(FileExistsError):
            rpm.run()
        rpm.run(overwrite=True)

        # Try to overwrite the model again
        config2 = model_config.copy()
        config2.pop("workdir")
        with pytest.raises(FileExistsError):
            rpm.create_rv(**config2)
        rpm.create_rv(overwrite=True, **config2)

        # Through RavenpyModel, both should work
        RavenpyModel(**model_config, overwrite=True).run(overwrite=True)
        RavenpyModel(**model_config, overwrite=False).run(overwrite=True)
        RavenpyModel(**model_config, overwrite=False).run(overwrite=True)
        with pytest.raises(FileExistsError):
            RavenpyModel(**model_config, overwrite=False).run(overwrite=False)

    def test_fake_ravenpy(self, deveraux):
        with pytest.raises(AttributeError):
            rpm = RavenpyModel(
                model_name="fake_test",
                parameters=[0.529, -3.396, 407.29, 1.072, 16.9, 0.947],
                hru=self.hru,
                start_date=self.start_date,
                end_date=self.end_date,
                meteo_file=deveraux.fetch(self.riviere_rouge_meteo),
                data_type=self.data_type,
                alt_names_meteo=self.alt_names_meteo,
                meteo_station_properties=self.meteo_station_properties,
                rain_snow_fraction=self.rain_snow_fraction,
                evaporation=self.evaporation,
            )

            rpm.run()

    def test_mult_stations(self, deveraux, tmp_path):
        meteo = xr.open_dataset(deveraux.fetch(self.riviere_rouge_meteo))
        meteo2 = meteo.copy().assign_coords({"station": ("station", ["B"])})
        meteo2 = meteo2.assign_coords(
            {
                "latitude": ("station", [45.0]),
                "longitude": ("station", [-80.0]),
                "elevation": ("station", [250.0]),
            }
        )
        meteo2["elevation"].attrs["units"] = "m"
        with xr.set_options(keep_attrs=True):
            meteo2 = meteo2 * 1.15
        meteo = meteo.assign_coords({"station": ("station", ["A"])})
        meteo = meteo.assign_coords(
            {
                "latitude": ("station", [45.5]),
                "longitude": ("station", [-80.5]),
                "elevation": ("station", [250.5]),
            }
        )
        meteo["elevation"].attrs["units"] = "m"
        meteo = xr.concat([meteo, meteo2], dim="station")

        meteo, cfg = xhm.format_input(
            meteo, model="GR4JCN", save_as=tmp_path / "test.nc"
        )

        parameters = [0.529, -3.396, 407.29, 1.072, 16.9, 0.947]
        global_parameter = {"AVG_ANNUAL_SNOW": 30.00}
        qsim = RavenpyModel(
            model_name="GR4JCN",
            parameters=parameters,
            hru=self.hru,
            start_date=self.start_date,
            end_date=self.end_date,
            workdir=tmp_path,
            meteo_station_properties=self.meteo_station_properties,
            rain_snow_fraction=self.rain_snow_fraction,
            evaporation=self.evaporation,
            global_parameter=global_parameter,
            Interpolation="INTERP_NEAREST_NEIGHBOR",
            overwrite=True,
            **cfg,
        ).run()

        assert qsim["q"].shape == (1827,)
        # This should be the same as the first test with 1 station
        np.testing.assert_almost_equal(
            qsim["q"].values[100:105],
            [0.02460463, 0.02451642, 0.02442908, 0.0243426, 0.0289188],
            decimal=5,
        )

        qsim2 = RavenpyModel(
            model_name="GR4JCN",
            parameters=parameters,
            hru=self.hru,
            start_date=self.start_date,
            end_date=self.end_date,
            workdir=tmp_path,
            meteo_station_properties=self.meteo_station_properties,
            rain_snow_fraction=self.rain_snow_fraction,
            evaporation=self.evaporation,
            global_parameter=global_parameter,
            Interpolation="INTERP_AVERAGE_ALL",
            overwrite=True,
            **cfg,
        ).run()

        # This should be different
        np.testing.assert_almost_equal(
            qsim2["q"].values[100:105],
            [0.02570859, 0.0256149, 0.02552219, 0.02543046, 0.03091073],
            decimal=5,
        )

    @pytest.mark.parametrize("input_type", ["gpd", "file", "dict"])
    def test_grid(self, deveraux, tmp_path, input_type):
        ds = xr.open_zarr(
            Path(
                deveraux.fetch(
                    "pmp/CMIP.CCCma.CanESM5.historical.r1i1p1f1.day.gn.zarr.zip",
                    pooch.Unzip(),
                )[0]
            ).parents[0]
        )
        ds_fx = xr.open_zarr(
            Path(
                deveraux.fetch(
                    "pmp/CMIP.CCCma.CanESM5.historical.r1i1p1f1.fx.gn.zarr.zip",
                    pooch.Unzip(),
                )[0]
            ).parents[0]
        )
        ds["orog"] = ds_fx["orog"]
        ds["pr"].attrs = {"units": "mm", "long_name": "precipitation"}
        ds = ds.drop_vars(["height", "prsn"])
        ds = ds.isel(plev=0)
        meteo, cfg = xhm.format_input(ds, model="GR4JCN", save_as=tmp_path / "test.nc")
        if input_type == "gpd":
            meteo["longitude"].attrs = {}
            meteo["latitude"].attrs = {}
            meteo.to_netcdf(tmp_path / "test_bad.nc")

        hru = Polygon(
            [
                (-77.0, 45.0),
                (-77.0, 46.0),
                (-78.0, 46.0),
                (-78.0, 45.0),
            ]
        )
        hru = gpd.GeoDataFrame(
            {
                "geometry": [hru],
                "area": [100.0],
                "elevation": [250.5],
                "latitude": [45.5],
                "longitude": [-77.5],
            },
            crs="EPSG:4326",
        )

        if input_type == "file":
            hru.to_file(str(tmp_path / "hru.shp"))
            hru = tmp_path / "hru.shp"
        elif input_type == "dict":
            hru = hru.iloc[0].to_dict()
            hru["crs"] = "EPSG:4326"

        parameters = [0.529, -3.396, 407.29, 1.072, 16.9, 0.947]
        global_parameter = {"AVG_ANNUAL_SNOW": 30.00}

        if input_type == "gpd":
            cfg2 = cfg.copy()
            cfg2["meteo_file"] = tmp_path / "test_bad.nc"
            with pytest.raises(
                ValueError, match="Could not determine the type of meteorological data"
            ):
                qsim = RavenpyModel(
                    model_name="GR4JCN",
                    parameters=parameters,
                    hru=hru,
                    start_date="2010-01-02",
                    end_date="2010-10-03",
                    workdir=tmp_path,
                    meteo_station_properties=self.meteo_station_properties,
                    rain_snow_fraction=self.rain_snow_fraction,
                    evaporation=self.evaporation,
                    global_parameter=global_parameter,
                    overwrite=True,
                    **cfg2,
                ).run()

        qsim = RavenpyModel(
            model_name="GR4JCN",
            parameters=parameters,
            hru=hru,
            start_date="2010-01-02",
            end_date="2010-10-03",
            workdir=tmp_path,
            meteo_station_properties=self.meteo_station_properties,
            rain_snow_fraction=self.rain_snow_fraction,
            evaporation=self.evaporation,
            global_parameter=global_parameter,
            overwrite=True,
            **cfg,
        ).run()

        assert qsim["q"].shape == (275,)
        np.testing.assert_almost_equal(
            qsim["q"].values[100:105],
            [0.04859913, 0.29512857, 0.90239469, 1.16436519, 1.05067477],
            decimal=5,
        )

    def test_hru_errors(self, deveraux, tmp_path):
        ds = xr.open_zarr(
            Path(
                deveraux.fetch(
                    "pmp/CMIP.CCCma.CanESM5.historical.r1i1p1f1.day.gn.zarr.zip",
                    pooch.Unzip(),
                )[0]
            ).parents[0]
        )
        ds_fx = xr.open_zarr(
            Path(
                deveraux.fetch(
                    "pmp/CMIP.CCCma.CanESM5.historical.r1i1p1f1.fx.gn.zarr.zip",
                    pooch.Unzip(),
                )[0]
            ).parents[0]
        )
        ds["orog"] = ds_fx["orog"]
        ds["pr"].attrs = {"units": "mm", "long_name": "precipitation"}
        ds = ds.drop_vars(["height", "prsn"])
        ds = ds.isel(plev=0)
        meteo, cfg = xhm.format_input(ds, model="GR4JCN", save_as=tmp_path / "test.nc")

        hru = Polygon(
            [
                (-77.0, 45.0),
                (-77.0, 46.0),
                (-78.0, 46.0),
                (-78.0, 45.0),
            ]
        )
        hru = gpd.GeoDataFrame(
            {
                "geometry": [hru],
                "area": [100.0],
                "elevation": [250.5],
                "latitude": [45.5],
                "longitude": [-77.5],
            },
            crs="EPSG:4326",
        )

        parameters = [0.529, -3.396, 407.29, 1.072, 16.9, 0.947]
        global_parameter = {"AVG_ANNUAL_SNOW": 30.00}

        with pytest.raises(ValueError, match="The HRU dataset must contain a geometry"):
            hru_error = hru.copy()
            hru_error = hru_error.drop(columns=["geometry"])
            RavenpyModel(
                model_name="GR4JCN",
                parameters=parameters,
                hru=hru_error,
                start_date="2010-01-02",
                end_date="2010-10-03",
                workdir=tmp_path,
                meteo_station_properties=self.meteo_station_properties,
                rain_snow_fraction=self.rain_snow_fraction,
                evaporation=self.evaporation,
                global_parameter=global_parameter,
                overwrite=True,
                **cfg,
            )

        with pytest.raises(ValueError, match="If using multiple HRUs,"):
            hru_error2 = hru.copy()
            hru_error2 = pd.concat([hru_error2, hru_error2]).reset_index(drop=True)

            RavenpyModel(
                model_name="GR4JCN",
                parameters=parameters,
                hru=hru_error2,
                start_date="2010-01-02",
                end_date="2010-10-03",
                workdir=tmp_path,
                meteo_station_properties=self.meteo_station_properties,
                rain_snow_fraction=self.rain_snow_fraction,
                evaporation=self.evaporation,
                global_parameter=global_parameter,
                overwrite=True,
                **cfg,
            )


@pytest.mark.skipif(ravenpy is None, reason="RavenPy is not installed.")
class TestDistributedRavenpy:
    # Model parameters: X01 to X21
    parameters = [
        -0.15,  # rainsnow_temp
        3.5,  # melt_factor
        3.0,  # refreeze_factor
        0.07,  # snow_swi
        0.4,  # porosity
        0.8,  # field_capacity
        1,  # hbv_beta
        4.0,  # max_perc_rate
        0.5,  # baseflow_coeff_fastres
        0.1,  # baseflow_coeff_slowres
        1,  # time_conc
        5.0,  # precip_lapse
        4.8,  # adiabatic_lapse
        0.1,  # sat_wilt
        1.0,  # baseflow_n
        22.0,  # max_cap_rise_rate
        0.5,  # topsoil_thickness
        0.1,  # hbv_melt_for_corr
        0.0,  # glac_storage_coeff
        1.0,  # rain_corr
        1.0,  # snow_corr
    ]

    @pytest.fixture(scope="class")
    def gridded_meteo(self, deveraux):
        ds = xr.open_zarr(
            Path(
                deveraux.fetch(
                    "pmp/CMIP.CCCma.CanESM5.historical.r1i1p1f1.day.gn.zarr.zip",
                    processor=pooch.Unzip(),
                )[0]
            ).parents[0]
        )
        ds_fx = xr.open_zarr(
            Path(
                deveraux.fetch(
                    "pmp/CMIP.CCCma.CanESM5.historical.r1i1p1f1.fx.gn.zarr.zip",
                    processor=pooch.Unzip(),
                )[0]
            ).parents[0]
        )
        ds["orog"] = ds_fx["orog"]
        ds["pr"].attrs = {"units": "mm", "long_name": "precipitation"}
        ds = ds.drop_vars(["height", "prsn"])
        ds = ds.isel(plev=0)
        meteo, cfg = xhm.format_input(ds, model="HBVEC")
        return meteo, cfg

    @pytest.fixture(scope="class")
    def df(self, deveraux):
        from xhydro.testing.helpers import deveraux as deveraux_branch

        df = gpd.read_file(
            Path(
                deveraux_branch(branch="distributed").fetch(
                    "ravenpy/hru_subset.zip",
                    processor=pooch.Unzip(),
                )[0]
            ).parents[0]
        )

        df.loc[:, "VEG_C"] = "VEG_ALL"
        df.loc[:, "LAND_USE_C"] = "LU_ALL"
        df.loc[:, "SOIL_PROF"] = "DEFAULT_P"
        return df

    @pytest.mark.parametrize("output_sub", ["all", None, "fail"])
    def test_hbvec_basic(self, tmp_path, df, gridded_meteo, output_sub):
        meteo, cfg = gridded_meteo
        meteo.to_netcdf(tmp_path / "test.nc")
        cfg["meteo_file"] = str(tmp_path / "test.nc")

        # Additional modifications to the model
        kwargs = {}

        # Distributed models require a global parameter for the average annual runoff
        kwargs["global_parameter"] = {"AVG_ANNUAL_RUNOFF": 500}

        if output_sub == "fail":
            with pytest.raises(ValueError, match="parameter must be either"):
                qsim = RavenpyModel(
                    model_name="HBVEC",
                    parameters=self.parameters,
                    hru=df,
                    start_date="2010-01-02",
                    end_date="2010-10-05",
                    workdir=tmp_path,
                    overwrite=True,
                    output_subbasins=output_sub,
                    **cfg | kwargs,
                ).run()
        else:
            qsim = RavenpyModel(
                model_name="HBVEC",
                parameters=self.parameters,
                hru=df,
                start_date="2010-01-02",
                end_date="2010-10-05",
                workdir=tmp_path,
                overwrite=True,
                output_subbasins=output_sub,
                **cfg | kwargs,
            ).run()

            assert "q" in qsim
            np.testing.assert_array_equal(qsim.time.min(), np.datetime64("2010-01-02"))
            np.testing.assert_array_equal(qsim.time.max(), np.datetime64("2010-10-05"))

            if output_sub is None:
                # If no output_sub is specified, we get the total flow for all HRUs
                assert qsim["q"].shape == (277,)
            else:
                assert len(qsim["q"].dims) == 2
                assert len(qsim["subbasin_id"]) == 47

    @pytest.mark.parametrize("output_sub", ["qobs", None, "fail"])
    def test_ravenpy_qobs(self, tmp_path, df, gridded_meteo, output_sub):
        meteo, cfg = gridded_meteo
        meteo.to_netcdf(tmp_path / "test.nc")
        cfg["meteo_file"] = str(tmp_path / "test.nc")

        # Additional modifications to the model
        kwargs = {}

        # Distributed models require a global parameter for the average annual runoff
        kwargs["global_parameter"] = {"AVG_ANNUAL_RUNOFF": 500}

        # Create a dummy qobs file
        qobs = xr.DataArray(
            np.array([np.random.rand(100), np.random.rand(100)]).transpose(),
            coords={
                "time": pd.date_range("2010-01-01", periods=100, freq="D"),
                "basin_id": ["13", "17"],
            },
            dims=["time", "basin_id"],
        )
        if output_sub is None:
            # If no output_sub is specified, we get the total flow for all HRUs
            qobs = qobs.assign_coords(station_id=("basin_id", ["020213", "0202017"]))
        qobs.attrs["units"] = "m3/s"
        qobs = qobs.to_dataset(name="qobs")
        if output_sub == "fail":
            qobs = qobs.rename({"basin_id": "abc"})
        qobs.to_netcdf(tmp_path / "qobs.nc")
        kwargs["qobs_file"] = str(tmp_path / "qobs.nc")
        kwargs["alt_name_flow"] = "qobs"

        if output_sub == "fail":
            with pytest.raises(
                ValueError, match="The observed streamflow dataset must contain a "
            ):
                RavenpyModel(
                    model_name="HBVEC",
                    parameters=self.parameters,
                    hru=df,
                    start_date="2010-01-02",
                    end_date="2010-10-05",
                    workdir=tmp_path,
                    overwrite=True,
                    output_subbasins=output_sub,
                    **cfg | kwargs,
                ).run()
        else:
            qsim = RavenpyModel(
                model_name="HBVEC",
                parameters=self.parameters,
                hru=df,
                start_date="2010-01-02",
                end_date="2010-10-05",
                workdir=tmp_path,
                overwrite=True,
                output_subbasins=output_sub,
                **cfg | kwargs,
            ).run()

            if output_sub is None:
                # If no output_sub is specified, we get the total flow for all HRUs
                assert qsim["q"].shape == (277,)
            else:
                assert len(qsim["q"].dims) == 2
                np.testing.assert_array_equal(
                    qsim["subbasin_id"].values, ["sub_13", "sub_17"]
                )

    def test_hbvec_reservoirs(self, tmp_path, df, gridded_meteo):
        meteo, cfg = gridded_meteo
        meteo.to_netcdf(tmp_path / "test.nc")
        cfg["meteo_file"] = str(tmp_path / "test.nc")

        df2 = df.copy()
        df2.loc[df2["HRU_ID"] == 1, "HRU_IsLake"] = 1
        df2.loc[df2["HRU_ID"] == 1, "Lake_Cat"] = 1
        df2.loc[df2["HRU_ID"] == 1, "LakeArea"] = 1000000
        df2.loc[df2["HRU_ID"] == 1, "LakeDepth"] = 1000

        # Additional modifications to the model
        kwargs = {}

        # Distributed models require a global parameter for the average annual runoff
        kwargs["global_parameter"] = {"AVG_ANNUAL_RUNOFF": 500}

        hm = RavenpyModel(
            model_name="HBVEC",
            parameters=self.parameters,
            hru=df2,
            start_date="2010-01-02",
            end_date="2010-10-05",
            workdir=tmp_path,
            overwrite=True,
            output_subbasins="all",
            **cfg | kwargs,
        )

        assert "reservoirs" in hm.emulator_config

        hm2 = RavenpyModel(
            model_name="HBVEC",
            parameters=self.parameters,
            hru=df2,
            start_date="2010-01-02",
            end_date="2010-10-05",
            workdir=tmp_path,
            overwrite=True,
            output_subbasins="all",
            minimum_reservoir_area="20 m2",
            **cfg | kwargs,
        )

        assert "reservoirs" in hm2.emulator_config

        hm_no = RavenpyModel(
            model_name="HBVEC",
            parameters=self.parameters,
            hru=df2,
            start_date="2010-01-02",
            end_date="2010-10-05",
            workdir=tmp_path,
            overwrite=True,
            output_subbasins="all",
            minimum_reservoir_area="20 km2",
            **cfg | kwargs,
        )

        assert "reservoirs" not in hm_no.emulator_config
