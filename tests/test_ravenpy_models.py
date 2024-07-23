import datetime as dt
import warnings

import numpy as np
import pooch
import pytest
import xarray as xr
from raven_hydro import __raven_version__

from xhydro.modelling import RavenpyModel


class TestRavenpyModels:
    """Test configurations of RavenPy models."""

    # Set Github URL for getting files for tests
    GITHUB_URL = "https://github.com/hydrologie/xhydro-testdata"
    BRANCH_OR_COMMIT_HASH = "main"

    # Get data from xhydro-testdata repo
    meteo_file = pooch.retrieve(
        url=f"{GITHUB_URL}/raw/{BRANCH_OR_COMMIT_HASH}/data/ravenpy/ERA5_Riviere_Rouge_global.nc",
        known_hash="md5:de985fa27ddceac690aeb34182a93f11",
    )
    qobs_path = pooch.retrieve(
        url=f"{GITHUB_URL}/raw/{BRANCH_OR_COMMIT_HASH}/data/ravenpy/Debit_Riviere_Rouge.nc",
        known_hash="md5:5b0feedc34333244b1d9e9c251323478",
    )

    # List of types of data provided to Raven in the meteo file
    data_type = ["TEMP_MAX", "TEMP_MIN", "PRECIP"]
    # Alternate names in the netcdf file (for variables that have different names, map to the name in the file).
    alt_names_meteo = {"TEMP_MIN": "tmin", "TEMP_MAX": "tmax", "PRECIP": "pr"}
    alt_names_flow = "qobs"
    start_date = dt.datetime(1985, 1, 1)
    end_date = dt.datetime(1990, 1, 1)
    qobs = xr.open_dataset(qobs_path).qobs.sel(time=slice(start_date, end_date)).values
    drainage_area = np.array([100.0])
    elevation = np.array([250.5])
    latitude = np.array([46.0])
    longitude = np.array([-80.75])

    # Station properties. Using the same as for the catchment, but could be different.
    meteo_station_properties = {
        "ALL": {"elevation": elevation, "latitude": latitude, "longitude": longitude}
    }
    rain_snow_fraction = "RAINSNOW_DINGMAN"
    evaporation = "PET_PRIESTLEY_TAYLOR"

    def test_ravenpy_gr4jcn(self):
        """Test for GR4JCN ravenpy model"""
        model_name = "GR4JCN"
        parameters = [0.529, -3.396, 407.29, 1.072, 16.9, 0.947]
        global_parameter = {"AVG_ANNUAL_SNOW": 30.00}

        rpm = RavenpyModel(
            model_name=model_name,
            parameters=parameters,
            drainage_area=self.drainage_area,
            elevation=self.elevation,
            latitude=self.latitude,
            longitude=self.longitude,
            start_date=self.start_date,
            end_date=self.end_date,
            qobs_path=self.qobs_path,
            alt_names_flow=self.alt_names_flow,
            meteo_file=self.meteo_file,
            data_type=self.data_type,
            alt_names_meteo=self.alt_names_meteo,
            meteo_station_properties=self.meteo_station_properties,
            rain_snow_fraction=self.rain_snow_fraction,
            evaporation=self.evaporation,
            global_parameter=global_parameter,
        )

        qsim = rpm.run()
        assert qsim["streamflow"].shape == (1827,)
        qsim2 = rpm.get_streamflow()
        assert qsim == qsim2
        met = rpm.get_inputs()
        assert len(met.time) == len(qsim.time)

    def test_ravenpy_hmets(self):
        model_name = "HMETS"
        parameters = [
            9.5019,
            0.2774,
            6.3942,
            0.6884,
            1.2875,
            5.4134,
            2.3641,
            0.0973,
            0.0464,
            0.1998,
            0.0222,
            -1.0919,
            2.6851,
            0.3740,
            1.0000,
            0.4739,
            0.0114,
            0.0243,
            0.0069,
            310.7211,
            916.1947,
        ]

        rpm = RavenpyModel(
            model_name=model_name,
            parameters=parameters,
            drainage_area=self.drainage_area,
            elevation=self.elevation,
            latitude=self.latitude,
            longitude=self.longitude,
            start_date=self.start_date,
            end_date=self.end_date,
            qobs_path=self.qobs_path,
            alt_names_flow=self.alt_names_flow,
            meteo_file=self.meteo_file,
            data_type=self.data_type,
            alt_names_meteo=self.alt_names_meteo,
            meteo_station_properties=self.meteo_station_properties,
            rain_snow_fraction=self.rain_snow_fraction,
            evaporation=self.evaporation,
        )

        qsim = rpm.run()
        assert qsim["streamflow"].shape == (1827,)

    def test_ravenpy_mohyse(self):
        """Test for Mohyse ravenpy model"""
        model_name = "Mohyse"
        parameters = [
            1.0,
            0.0468,
            4.2952,
            2.658,
            0.4038,
            0.0621,
            0.0273,
            0.0453,
            0.9039,
            5.6167,
        ]

        rpm = RavenpyModel(
            model_name=model_name,
            parameters=parameters,
            drainage_area=self.drainage_area,
            elevation=self.elevation,
            latitude=self.latitude,
            longitude=self.longitude,
            start_date=self.start_date,
            end_date=self.end_date,
            qobs_path=self.qobs_path,
            alt_names_flow=self.alt_names_flow,
            meteo_file=self.meteo_file,
            data_type=self.data_type,
            alt_names_meteo=self.alt_names_meteo,
            meteo_station_properties=self.meteo_station_properties,
            rain_snow_fraction=self.rain_snow_fraction,
            evaporation=self.evaporation,
        )

        qsim = rpm.run()

        assert qsim["streamflow"].shape == (1827,)

    @pytest.mark.skip(
        reason="Weird error with negative simulated PET in ravenpy for HBVEC."
    )
    def test_ravenpy_hbvec(self):
        """Test for HBVEC ravenpy model"""
        model_name = "HBVEC"
        parameters = [
            0.059,
            4.072,
            2.002,
            0.035,
            0.10,
            0.506,
            3.44,
            38.32,
            0.46,
            0.063,
            2.278,
            4.87,
            0.57,
            0.045,
            0.878,
            18.941,
            2.037,
            0.445,
            0.677,
            1.141,
            1.024,
        ]

        rpm = RavenpyModel(
            model_name=model_name,
            parameters=parameters,
            drainage_area=self.drainage_area,
            elevation=self.elevation,
            latitude=self.latitude,
            longitude=self.longitude,
            start_date=self.start_date,
            end_date=self.end_date,
            qobs_path=self.qobs_path,
            alt_names_flow=self.alt_names_flow,
            meteo_file=self.meteo_file,
            data_type=self.data_type,
            alt_names_meteo=self.alt_names_meteo,
            meteo_station_properties=self.meteo_station_properties,
            rain_snow_fraction=self.rain_snow_fraction,
            evaporation=self.evaporation,
        )

        qsim = rpm.run()

        assert qsim["streamflow"].shape == (1827,)

    @pytest.mark.skip(
        reason="Weird error with negative simulated PET in ravenpy for HYPR."
    )
    def test_ravenpy_hypr(self):
        """Test for HYPR ravenpy model"""
        model_name = "HYPR"
        parameters = [
            -0.186,
            2.92,
            0.031,
            0.439,
            0.465,
            0.117,
            13.1,
            0.404,
            1.21,
            59.1,
            0.166,
            4.10,
            0.822,
            41.5,
            5.85,
            0.69,
            0.924,
            1.64,
            1.59,
            2.51,
            1.14,
        ]

        rpm = RavenpyModel(
            model_name=model_name,
            parameters=parameters,
            drainage_area=self.drainage_area,
            elevation=self.elevation,
            latitude=self.latitude,
            longitude=self.longitude,
            start_date=self.start_date,
            end_date=self.end_date,
            qobs_path=self.qobs_path,
            alt_names_flow=self.alt_names_flow,
            meteo_file=self.meteo_file,
            data_type=self.data_type,
            alt_names_meteo=self.alt_names_meteo,
            meteo_station_properties=self.meteo_station_properties,
            rain_snow_fraction=self.rain_snow_fraction,
            evaporation=self.evaporation,
        )

        qsim = rpm.run()

        assert qsim["streamflow"].shape == (1827,)

    def test_ravenpy_sacsma(self):
        """Test for SAC-SMA ravenpy model"""
        model_name = "SACSMA"
        parameters = [
            0.01,
            0.05,
            0.3,
            0.05,
            0.05,
            0.13,
            0.025,
            0.06,
            0.06,
            1.0,
            40.0,
            0.0,
            0.0,
            0.1,
            0.0,
            0.01,
            1.5,
            0.482,
            4.1,
            1.0,
            1.0,
        ]

        rpm = RavenpyModel(
            model_name=model_name,
            parameters=parameters,
            drainage_area=self.drainage_area,
            elevation=self.elevation,
            latitude=self.latitude,
            longitude=self.longitude,
            start_date=self.start_date,
            end_date=self.end_date,
            qobs_path=self.qobs_path,
            alt_names_flow=self.alt_names_flow,
            meteo_file=self.meteo_file,
            data_type=self.data_type,
            alt_names_meteo=self.alt_names_meteo,
            meteo_station_properties=self.meteo_station_properties,
            rain_snow_fraction=self.rain_snow_fraction,
            evaporation=self.evaporation,
        )

        qsim = rpm.run()

        assert qsim["streamflow"].shape == (1827,)

    def test_ravenpy_blended(self):
        """Test for Blended ravenpy model"""
        model_name = "Blended"
        parameters = [
            0.029,
            2.21,
            2.16,
            0.00023,
            21.74,
            1.56,
            6.21,
            0.91,
            0.035,
            0.25,
            0.00022,
            1.214,
            0.0473,
            0.207,
            0.078,
            -1.34,
            0.22,
            3.84,
            0.29,
            0.483,
            4.1,
            12.83,
            0.594,
            1.65,
            1.71,
            0.372,
            0.0712,
            0.019,
            0.408,
            0.94,
            -1.856,
            2.36,
            1.0,
            1.0,
            0.0075,
            0.53,
            0.0289,
            0.961,
            0.613,
            0.956,
            0.101,
            0.0928,
            0.747,
        ]

        rpm = RavenpyModel(
            model_name=model_name,
            parameters=parameters,
            drainage_area=self.drainage_area,
            elevation=self.elevation,
            latitude=self.latitude,
            longitude=self.longitude,
            start_date=self.start_date,
            end_date=self.end_date,
            qobs_path=self.qobs_path,
            alt_names_flow=self.alt_names_flow,
            meteo_file=self.meteo_file,
            data_type=self.data_type,
            alt_names_meteo=self.alt_names_meteo,
            meteo_station_properties=self.meteo_station_properties,
            rain_snow_fraction=self.rain_snow_fraction,
            evaporation=self.evaporation,
        )

        if __raven_version__ == "3.8.1":
            warnings.warn("Blended model does not work with RavenHydroFramework v3.8.1")
            with pytest.raises(OSError):
                rpm.run()
        else:
            qsim = rpm.run()
            assert qsim["streamflow"].shape == (1827,)

    def test_fake_ravenpy(self):
        """Test for GR4JCN ravenpy model"""
        model_name = "fake_test"
        parameters = [0.529, -3.396, 407.29, 1.072, 16.9, 0.947]

        with pytest.raises(ValueError):
            rpm = RavenpyModel(
                model_name=model_name,
                parameters=parameters,
                drainage_area=self.drainage_area,
                elevation=self.elevation,
                latitude=self.latitude,
                longitude=self.longitude,
                start_date=self.start_date,
                end_date=self.end_date,
                qobs_path=self.qobs_path,
                alt_names_flow=self.alt_names_flow,
                meteo_file=self.meteo_file,
                data_type=self.data_type,
                alt_names_meteo=self.alt_names_meteo,
                meteo_station_properties=self.meteo_station_properties,
                rain_snow_fraction=self.rain_snow_fraction,
                evaporation=self.evaporation,
            )

            rpm.run()
