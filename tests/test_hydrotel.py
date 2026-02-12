import os
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from dotenv import load_dotenv
from xclim.testing.helpers import test_timeseries as timeseries

import xhydro.testing
from xhydro.modelling import Hydrotel, hydrological_model
from xhydro.modelling._hydrotel import _overwrite_csv, _read_csv


# If you want to execute the tests with the actual Hydrotel executable, create a .env file in the tests/ folder with the following variables:
# HYDROTEL_DEMO: path to the DELISLE demo project (copied from https://github.com/INRS-Modelisation-hydrologique/hydrotel/tree/main/DemoProject/DELISLE)
# HYDROTEL_EXECUTABLE: path to the Hydrotel executable
# HYDROTEL_VERSION: version of Hydrotel (e.g. "4.3.6.0000")
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
hydrotel_demo = os.getenv("HYDROTEL_DEMO", None)
hydrotel_executable = os.getenv("HYDROTEL_EXECUTABLE", "command")
hydrotel_version = os.getenv("HYDROTEL_VERSION", None)
if hydrotel_executable != "command" and (hydrotel_version is None or hydrotel_demo is None):
    raise ValueError(
        "If HYDROTEL_EXECUTABLE is set to a path, you must also set HYDROTEL_VERSION and HYDROTEL_DEMO to the"
        " corresponding version and demo project path."
    )


class TestHydrotel:
    @pytest.fixture(scope="class")
    def project_path(self, tmp_path_factory):
        # Create a temporary directory for the project and clean it up after the tests
        tmp_path_factory.mktemp("hydrotel")
        yield tmp_path_factory.getbasetemp() / "hydrotel"
        shutil.rmtree(tmp_path_factory.getbasetemp() / "hydrotel", ignore_errors=True)

    @staticmethod
    def create_project(project_path, replace_meteo=None, **kwargs):
        if hydrotel_demo is None:
            xhydro.testing.utils.fake_hydrotel_project(project_path, **kwargs)
        else:
            shutil.copytree(hydrotel_demo, project_path)
            shutil.rmtree(project_path / "simulation" / "simulation" / "resultat", ignore_errors=True)
            if replace_meteo is not None:
                for f in (project_path / "meteo").glob("*"):
                    f.unlink()

                if replace_meteo in ["station", "grid"]:
                    meteo = timeseries(
                        np.zeros(365 * 2),
                        start="2020-01-01",
                        freq="D",
                        variable="tasmin",
                        as_dataset=True,
                        units="degC",
                    )
                    meteo["tasmax"] = timeseries(
                        np.ones(365 * 2),
                        start="2020-01-01",
                        freq="D",
                        variable="tasmax",
                        units="degC",
                    )
                    meteo["pr"] = timeseries(
                        np.ones(365 * 2) * 10,
                        start="2020-01-01",
                        freq="D",
                        variable="pr",
                        units="mm",
                    )
                    if replace_meteo == "station":
                        meteo = meteo.expand_dims({"stations": [0, 1, 2]})
                        meteo["lat"] = xr.DataArray([45.18, 45.28, 45.38], dims=["stations"])
                        meteo["lon"] = xr.DataArray([-74.18, -74.28, -74.38], dims=["stations"])
                        meteo["z"] = xr.DataArray([0, 0, 0], dims=["stations"])
                        meteo = meteo.assign_coords(coords={"lat": meteo.lat, "lon": meteo.lon, "z": meteo.z})
                    else:
                        meteo = meteo.expand_dims({"lat": [45.18, 45.28, 45.38], "lon": [-74.18, -74.28, -74.38]})
                        meteo["z"] = xr.DataArray([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dims=["lat", "lon"])

                    meteo["lon"].attrs = {"units": "degrees_east"}
                    meteo["lat"].attrs = {"units": "degrees_north"}
                    meteo["z"].attrs = {"units": "m"}

                    return meteo

    def test_options(self, project_path):
        self.create_project(project_path / "fake-for-options")

        model_config = dict(
            model_name="Hydrotel",
            project_dir=project_path / "fake-for-options",
            project_file="DELISLE.csv",
            use_defaults=True,
            executable=hydrotel_executable,
            project_config={"PROJET HYDROTEL VERSION": "2.1.0"},
            simulation_config={"SIMULATION HYDROTEL VERSION": "1.0.5"},
            output_config={"TMAX_JOUR": "1", "OUTPUT_NETCDF": "1"},
        )

        with pytest.warns(FutureWarning, match="Please refer to the DemoProject in"):
            ht = hydrological_model(
                model_config=model_config,
            )

        assert ht.simulation_dir.name == "simulation"
        assert ht.project_dir.name == "fake-for-options"

        # Check that the configuration options have been updated and that the files have been overwritten
        assert ht.project_config["PROJET HYDROTEL VERSION"] == "2.1.0"
        df = _read_csv(ht.config_files["project"])
        assert ht.project_config == df

        assert ht.simulation_config["SIMULATION HYDROTEL VERSION"] == "1.0.5"
        df = _read_csv(ht.config_files["simulation"])
        assert ht.simulation_config == df

        assert ht.output_config["TMAX_JOUR"] == "1"
        df = _read_csv(ht.config_files["output"])
        assert ht.output_config == df

        if hydrotel_executable != "command":
            ds = ht.run()
            # The version in the configuration should not affect the version in the output dataset, which is read from the output file attributes
            assert ds.attrs["Hydrotel_version"] == hydrotel_version
            assert ds.attrs["Hydrotel_config_version"] == "1.0.5"
            assert (project_path / "fake-for-options" / "simulation" / "simulation" / "resultat" / "tmaxjour.nc").exists()

    @pytest.mark.parametrize("test", ["station", "grid", "none", "toomany"])
    def test_get_data(self, project_path, test):
        meteo = self.create_project(
            project_path / f"fake-for-get-data-{test}", replace_meteo=test if test in ["station", "grid"] else None, meteo=True, debit_aval=True
        )

        if test in ["station", "grid"]:
            simulation_config = {
                "FICHIER STATIONS METEO": "meteo\\meteo.nc",
                "FICHIER GRILLE METEO": "",
            }
        elif test == "none":
            if hydrotel_executable == "command":
                simulation_config = {}
            else:
                simulation_config = {
                    "FICHIER STATIONS METEO": "",
                    "FICHIER GRILLE METEO": "",
                }
        else:
            simulation_config = {
                "FICHIER STATIONS METEO": "meteo\\meteo.nc",
                "FICHIER GRILLE METEO": "meteo\\meteo.nc",
            }

        ht = Hydrotel(
            project_dir=project_path / f"fake-for-get-data-{test}",
            project_file="DELISLE.csv",
            executable=hydrotel_executable,
            simulation_config=simulation_config,
            output_config={"OUTPUT_NETCDF": "1"},
        )
        if test in ["station", "grid"]:
            if hydrotel_executable != "command":
                xhydro.modelling.format_input(meteo, "Hydrotel", save_as=project_path / f"fake-for-get-data-{test}" / "meteo" / "meteo.nc")
                ht.run()

            ds, config = ht.get_inputs(return_config=True)
            assert config["TYPE (STATION/GRID/GRID_EXTENT)"] == "STATION"  # It's always reformatted to stations
            assert config["STATION_DIM_NAME"] == "station_id" if hydrotel_executable != "command" else "stations"
            assert config["TMAX_NAME"] == "tasmax"
            assert all(v in ds.variables for v in ["tasmin", "tasmax", "pr"])
            np.testing.assert_array_equal(ds.tasmin, 0)
            np.testing.assert_array_equal(ds.tasmax, 1)

            out = ht.get_streamflow()
            if hydrotel_executable == "command":
                assert all(v in out.variables for v in ["debit_aval"])
                assert set(out.dims) == {"time", "troncon"}
                np.testing.assert_array_equal(out.debit_aval.mean(), 0)
            else:
                assert all(v in out.variables for v in ["q"])
                assert set(out.dims) == {"time", "subbasin_id"}
                np.testing.assert_array_almost_equal(out.q.mean(), 10.15166855)

        elif test == "toomany":
            with pytest.raises(
                ValueError,
                match="Both 'FICHIER GRILLE METEO' and 'FICHIER STATIONS METEO' are specified in the simulation configuration file.",
            ):
                ht.get_inputs()
        else:
            with pytest.raises(
                ValueError,
                match="You must specify either 'FICHIER GRILLE METEO' or 'FICHIER STATIONS METEO'",
            ):
                ht.get_inputs()

    @pytest.mark.parametrize("subset", [True, False])
    def test_input_dates(self, project_path, subset):
        meteo = timeseries(
            np.zeros(365 * 2),
            start="2020-01-01",
            freq="D",
            variable="tasmin",
            as_dataset=True,
            units="degC",
        )
        meteo["tasmax"] = timeseries(
            np.ones(365 * 2),
            start="2020-01-01",
            freq="D",
            variable="tasmax",
            units="degC",
        )
        meteo["pr"] = timeseries(
            np.ones(365 * 2) * 10,
            start="2020-01-01",
            freq="D",
            variable="pr",
            units="mm",
        )
        meteo = meteo.expand_dims({"stations": [0, 1, 2]})
        meteo["lat"] = xr.DataArray([45.18, 45.28, 45.38], dims=["stations"])
        meteo["lon"] = xr.DataArray([-74.18, -74.28, -74.38], dims=["stations"])
        meteo["z"] = xr.DataArray([0, 0, 0], dims=["stations"])
        meteo = meteo.assign_coords(coords={"lat": meteo.lat, "lon": meteo.lon, "z": meteo.z})
        meteo["lon"].attrs = {"units": "degrees_east"}
        meteo["lat"].attrs = {"units": "degrees_north"}
        meteo["z"].attrs = {"units": "m"}
        self.create_project(project_path / f"fake-for-input-dates-{subset}", replace_meteo="grid", meteo=meteo)

        date_debut = "2021-01-01"
        date_fin = "2021-05-31"
        simulation_config = {
            "FICHIER STATIONS METEO": "meteo\\meteo.nc",
            "DATE DEBUT": date_debut,
            "DATE FIN": date_fin,
            "PAS DE TEMPS": 24,
        }
        ht = Hydrotel(
            project_dir=project_path / f"fake-for-input-dates-{subset}",
            project_file="DELISLE.csv",
            executable=hydrotel_executable,
            simulation_config=simulation_config,
            output_config={"OUTPUT_NETCDF": "1"},
        )
        assert ht.simulation_config["DATE DEBUT"] == f"{date_debut} 00:00"
        assert ht.simulation_config["DATE FIN"] == f"{date_fin} 00:00"

        if hydrotel_executable != "command":
            xhydro.modelling.format_input(meteo, "Hydrotel", save_as=project_path / f"fake-for-input-dates-{subset}" / "meteo" / "meteo.nc")
            ht.run()

        ds = ht.get_inputs(subset_time=subset)
        if subset:
            assert ds.time.min().dt.strftime("%Y-%m-%d").item() == date_debut
            assert ds.time.max().dt.strftime("%Y-%m-%d").item() == date_fin
        else:
            assert ds.time.min().dt.strftime("%Y-%m-%d").item() == "2020-01-01"
            assert ds.time.max().dt.strftime("%Y-%m-%d").item() == "2021-12-30"

        if hydrotel_executable != "command":
            out = ht.get_streamflow()
            np.testing.assert_array_equal(out.time.min().dt.strftime("%Y-%m-%d").item(), date_debut)
            np.testing.assert_array_equal(out.time.max().dt.strftime("%Y-%m-%d").item(), "2021-05-30")  # The end date in Hydrotel is exclusive

    def test_standard(self, project_path):
        self.create_project(project_path / "fake-for-standard", debit_aval=True)

        ht = Hydrotel(
            project_dir=project_path / "fake-for-standard",
            project_file="DELISLE.csv",
            executable=hydrotel_executable,
            output_config={"OUTPUT_NETCDF": "1"},
        )

        if hydrotel_executable != "command":
            ht.run()
            ds_orig = None
        else:
            with ht.get_streamflow() as ds_tmp:
                ds_orig = deepcopy(ds_tmp)
            ht._standardise_outputs()
        ds = ht.get_streamflow()

        if ds_orig is not None:
            # To make sure the original dataset was not modified prior to standardisation
            assert list(ds_orig.data_vars) == ["debit_aval"]
            assert set(ds_orig.dims) == {"time", "troncon"}

        assert list(ds.data_vars) == ["q"]
        assert set(ds.dims) == {"time", "subbasin_id"}
        correct_attrs = {
            "units": "m3 s-1",
            "description": "Simulated streamflow at the outlet of the subbasin.",
            "standard_name": "outgoing_water_volume_transport_along_river_channel",
            "long_name": "Simulated streamflow",
            "_original_name": "debit_aval",
            "_original_description": "Debit en aval du troncon",
        }
        assert sorted(set(ds.q.attrs)) == sorted(set(correct_attrs))
        for k, v in correct_attrs.items():
            assert ds.q.attrs[k] == v

        assert ds.attrs["Hydrotel_version"] == "unspecified" if hydrotel_executable == "command" else hydrotel_version
        assert ds.attrs["Hydrotel_config_version"] == "" if hydrotel_executable == "command" else "4.3.1.0000"

        assert "initial_simulation_path" not in ds.attrs

    def test_simname(self, project_path):
        self.create_project(project_path / "fake-for-simname", debit_aval=True)
        with pytest.raises(ValueError, match="folder does not exist"):
            Hydrotel(
                project_path / "fake-for-simname",
                "DELISLE.csv",
                executable="command",
                project_config={"SIMULATION COURANTE": "test"},
            )

        ht = Hydrotel(project_path / "fake-for-simname", "DELISLE.csv", executable="command")
        with pytest.raises(ValueError, match="folder does not exist"):
            ht.update_config(project_config={"SIMULATION COURANTE": "test"})

        simdir = project_path / "fake-for-simname" / "simulation"
        (simdir / "simulation").rename(simdir / "test")
        (simdir / "test" / "simulation.csv").rename(simdir / "test" / "test.csv")
        (simdir / "test" / "simulation.gsb").rename(simdir / "test" / "test.gsb")
        ht2 = Hydrotel(
            project_path / "fake-for-simname",
            "DELISLE.csv",
            executable=hydrotel_executable,
            project_config={"SIMULATION COURANTE": "test"},
            output_config={"OUTPUT_NETCDF": "1"},
        )
        if hydrotel_executable != "command":
            ht2.run()

    @pytest.mark.parametrize("test", ["ok", "pdt", "cfg", "raise"])
    def test_run(self, project_path, test):
        self.create_project(project_path / f"fake-for-run-{test}", meteo=True)

        path_to_executable = hydrotel_executable if test != "raise" else "not_a_command"
        if path_to_executable == "command":
            path_to_executable = "hydrotel"
        ht = Hydrotel(
            project_dir=project_path / f"fake-for-run-{test}",
            project_file="DELISLE.csv",
            executable=path_to_executable,
        )

        if test in ["ok", "pdt"]:
            command = ht.run(dry_run=True)
            assert command == f"{path_to_executable} {ht.config_files['project']} -t 1"
        elif test == "cfg":
            run_options = ["-t 10", "-c", "-s"]
            check_missing = True
            xr_open_kwargs_in = {"chunks": {"time": 10}}
            with pytest.warns(
                FutureWarning,
            ) as w:
                command = ht.run(
                    dry_run=True,
                    run_options=run_options,
                    check_missing=check_missing,
                    xr_open_kwargs_in=xr_open_kwargs_in,
                )
            assert len(w) == 2
            assert command == (f"{path_to_executable} {ht.config_files['project']} -c -s -t 10")
        elif test == "raise":
            with pytest.raises(
                ValueError,
                match="The executable command does not seem to be a valid Hydrotel command",
            ):
                ht.run(dry_run=True)

    # Note that if using the actual executable, this test is quite slow (~2-4 minutes), because HYDROTEL needs to recompute/convert
    # the geomorphological hydrograph at the new time step.
    def test_subdaily(self, project_path):
        meteo = timeseries(
            np.zeros(365 * 2),
            start="2020-01-01",
            freq="3H",
            variable="tasmin",
            as_dataset=True,
            units="degC",
        )
        meteo["tasmax"] = timeseries(
            np.ones(365 * 2),
            start="2020-01-01",
            freq="3H",
            variable="tasmax",
            units="degC",
        )
        meteo["pr"] = timeseries(
            np.ones(365 * 2) * 10,
            start="2020-01-01",
            freq="3H",
            variable="pr",
            units="mm",
        )
        meteo = meteo.expand_dims({"stations": [0, 1, 2]})
        meteo["lat"] = xr.DataArray([45.18, 45.28, 45.38], dims=["stations"])
        meteo["lon"] = xr.DataArray([-74.18, -74.28, -74.38], dims=["stations"])
        meteo["z"] = xr.DataArray([0, 0, 0], dims=["stations"])
        meteo = meteo.assign_coords(coords={"lat": meteo.lat, "lon": meteo.lon, "z": meteo.z})
        meteo["lon"].attrs = {"units": "degrees_east"}
        meteo["lat"].attrs = {"units": "degrees_north"}
        meteo["z"].attrs = {"units": "m"}
        self.create_project(project_path / "fake-for-subdaily", replace_meteo="delete", meteo=meteo)

        ht = Hydrotel(
            project_dir=project_path / "fake-for-subdaily",
            project_file="DELISLE.csv",
            executable=hydrotel_executable,
            simulation_config={
                "FICHIER STATIONS METEO": "meteo\\meteo.nc",
                "DATE DEBUT": "2020-01-01",
                "DATE FIN": "2020-02-01",
                "PAS DE TEMPS": 3,
                "LECTURE ETAT FONTE NEIGE": "",
                "LECTURE ETAT BILAN VERTICAL": "",
                "LECTURE ETAT RUISSELEMENT SURFACE": "",
                "LECTURE ETAT ACHEMINEMENT RIVIERE": "",
            },
            output_config={"OUTPUT_NETCDF": "1"},
        )
        assert ht.simulation_config["DATE DEBUT"] == "2020-01-01 00:00"
        assert ht.simulation_config["DATE FIN"] == "2020-02-01 00:00"

        ds, _ = xhydro.modelling.format_input(meteo, "Hydrotel", save_as=project_path / "fake-for-subdaily" / "meteo" / "meteo.nc")
        assert xr.infer_freq(xr.decode_cf(ds).time) == "3h"

        if hydrotel_executable != "command":
            shutil.rmtree(project_path / "fake-for-subdaily" / "etats", ignore_errors=True)
            ht.run()
            out = ht.get_streamflow()
            assert xr.infer_freq(out.time) == "3h"
            assert out.time.min().dt.strftime("%Y-%m-%d %H:%M").item() == "2020-01-01 00:00"
            assert out.time.max().dt.strftime("%Y-%m-%d %H:%M").item() == "2020-01-31 21:00"

    def test_errors(self, tmpdir):
        # Missing project folder
        with pytest.raises(ValueError, match="The project folder does not exist."):
            Hydrotel("fake", "DELISLE.csv", executable="command")

        # Missing project name
        xhydro.testing.utils.fake_hydrotel_project(tmpdir)
        project_config = {
            "SIMULATION COURANTE": "",
        }
        _overwrite_csv(tmpdir / "DELISLE.csv", project_config)
        with pytest.raises(
            ValueError,
            match="'SIMULATION COURANTE' must be specified",
        ):
            Hydrotel(tmpdir, "DELISLE.csv", executable="command")

        # Simulation does not match project name
        xhydro.testing.utils.fake_hydrotel_project(tmpdir)
        project_config = {
            "SIMULATION COURANTE": "abc",
        }
        Path(tmpdir / "simulation" / "simulation").rename(
            tmpdir / "simulation" / "abc",
        )
        with pytest.raises(
            FileNotFoundError,
            match="/abc/abc.csv",
        ):
            Hydrotel(tmpdir, "DELISLE.csv", executable="command", project_config=project_config)

        xhydro.testing.utils.fake_hydrotel_project(tmpdir)
        # Remove simulation.csv
        Path(tmpdir / "simulation" / "simulation" / "simulation.csv").unlink()
        with pytest.raises(FileNotFoundError):
            Hydrotel(tmpdir, "DELISLE.csv", executable="command")

    def test_bad_overwrite(self, tmpdir):
        xhydro.testing.utils.fake_hydrotel_project(tmpdir)
        with pytest.raises(ValueError, match="Could not find the following keys in the file"):
            _overwrite_csv(
                tmpdir / "simulation" / "simulation" / "output.csv",
                {"foo": "bar"},
            )
