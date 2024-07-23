import os
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from packaging.version import parse
from xclim import __version__ as __xclim_version__
from xclim.testing.helpers import test_timeseries as timeseries

import xhydro.testing
from xhydro.modelling import Hydrotel, hydrological_model
from xhydro.modelling._hydrotel import _overwrite_csv, _read_csv


class TestHydrotel:
    def test_options(self, tmpdir):
        xhydro.testing.utils.fake_hydrotel_project(tmpdir / "fake")

        model_config = dict(
            model_name="Hydrotel",
            project_dir=tmpdir / "fake",
            project_file="SLNO.csv",
            use_defaults=True,
            project_config={"PROJET HYDROTEL VERSION": "2.1.0"},
            simulation_config={"SIMULATION HYDROTEL VERSION": "1.0.5"},
            output_config={"TMAX_JOUR": "1"},
        )
        ht = hydrological_model(
            model_config=model_config,
        )

        assert ht.simulation_dir.name == "simulation"
        assert ht.project_dir.name == "fake"

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

        ht2 = Hydrotel(
            project_dir=tmpdir / "fake", project_file="SLNO.csv", use_defaults=False
        )
        assert ht2.project_config == ht.project_config
        assert ht2.simulation_config == ht.simulation_config
        assert ht2.output_config == ht.output_config

    @pytest.mark.parametrize("test", ["station", "grid", "none", "toomany"])
    def test_get_data(self, tmpdir, test):
        xhydro.testing.utils.fake_hydrotel_project(tmpdir, meteo=True, debit_aval=True)
        if test == "station":
            simulation_config = {"FICHIER STATIONS METEO": "meteo\\SLNO_meteo_GC3H.nc"}
        elif test == "grid":
            simulation_config = {"FICHIER GRILLE METEO": "meteo\\SLNO_meteo_GC3H.nc"}
        elif test == "none":
            simulation_config = {}
        else:
            simulation_config = {
                "FICHIER STATIONS METEO": "meteo\\SLNO_meteo_GC3H.nc",
                "FICHIER GRILLE METEO": "meteo\\SLNO_meteo_GC3H.nc",
            }

        ht = Hydrotel(
            project_dir=tmpdir,
            project_file="SLNO.csv",
            use_defaults=True,
            simulation_config=simulation_config,
        )
        if test in ["station", "grid"]:
            ds = ht.get_inputs()
            assert all(v in ds.variables for v in ["tasmin", "tasmax", "pr"])
            np.testing.assert_array_equal(ds.tasmin, np.zeros([1, 365 * 2]))
            np.testing.assert_array_equal(ds.tasmax.mean(), 1)

            ds = ht.get_streamflow()
            assert all(v in ds.variables for v in ["debit_aval"])
            np.testing.assert_array_equal(ds.dims, ["time", "troncon"])
            np.testing.assert_array_equal(ds.debit_aval.mean(), 0)
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
    def test_input_dates(self, tmpdir, subset):
        meteo = timeseries(
            np.zeros(365 * 10),
            start="2001-01-01",
            freq="D",
            variable="tasmin",
            as_dataset=True,
            units="K",
        )
        meteo["tasmax"] = timeseries(
            np.ones(365 * 10),
            start="2001-01-01",
            freq="D",
            variable="tasmax",
            units="degC",
        )
        meteo["pr"] = timeseries(
            np.ones(365 * 10) * 10,
            start="2001-01-01",
            freq="D",
            variable="pr",
            units="mm",
        )
        meteo = meteo.expand_dims("stations").assign_coords(stations=["010101"])
        meteo = meteo.assign_coords(coords={"lat": 46, "lon": -77})
        for c in ["lat", "lon"]:
            meteo[c] = meteo[c].expand_dims("stations")

        xhydro.testing.utils.fake_hydrotel_project(tmpdir, meteo=meteo)

        date_debut = "2002-01-01"
        date_fin = "2005-12-31"
        simulation_config = {
            "FICHIER STATIONS METEO": r"meteo\SLNO_meteo_GC3H.nc",
            "DATE DEBUT": date_debut,
            "DATE FIN": date_fin,
            "PAS DE TEMPS": 24,
        }
        ht = Hydrotel(
            project_dir=tmpdir,
            project_file="SLNO.csv",
            use_defaults=True,
            simulation_config=simulation_config,
        )

        ds = ht.get_inputs(subset_time=subset)
        if subset:
            assert ds.time.min().dt.strftime("%Y-%m-%d").item() == date_debut
            assert ds.time.max().dt.strftime("%Y-%m-%d").item() == date_fin
        else:
            assert ds.time.min().dt.strftime("%Y-%m-%d").item() == "2001-01-01"
            assert ds.time.max().dt.strftime("%Y-%m-%d").item() == "2010-12-29"

    @pytest.mark.parametrize("test", ["ok", "file", "health"])
    def test_basic(self, tmpdir, test):
        meteo = True
        if test == "health":
            meteo = timeseries(
                np.zeros(365 * 2),
                start="2001-01-01",
                freq="D",
                variable="tasmin",
                as_dataset=True,
                units="K",
            )
            meteo["tasmax"] = timeseries(
                np.ones(365 * 2),
                start="2001-01-01",
                freq="D",
                variable="tasmax",
                units="degC",
            )
            meteo["pr"] = timeseries(
                np.ones(365 * 2) * 10,
                start="2001-01-01",
                freq="D",
                variable="pr",
                units="mm",
            )
            meteo = meteo.expand_dims("stations").assign_coords(stations=["010101"])
            meteo = meteo.assign_coords(coords={"lat": 46, "lon": -77})
            for c in ["lat", "lon"]:
                meteo[c] = meteo[c].expand_dims("stations")
            meteo = meteo.squeeze()
            meteo = meteo.convert_calendar("noleap")
            meteo = xr.concat(
                (
                    meteo.sel(time=slice("2001-01-01", "2001-12-30")),
                    meteo.sel(time=slice("2002-01-01", "2002-12-30")),
                ),
                dim="time",
            )

        xhydro.testing.utils.fake_hydrotel_project(tmpdir, meteo=meteo)
        date_debut = "2001-10-01" if test != "health" else "1999-01-01"
        simulation_config = {
            "FICHIER STATIONS METEO": r"meteo\SLNO_meteo_GC3H.nc",
            "DATE DEBUT": date_debut,
            "DATE FIN": "2002-12-30",
            "PAS DE TEMPS": 24,
        }
        if test == "file":
            Path(tmpdir / "simulation" / "simulation" / "lecture_tempsol.csv").unlink()

        ht = Hydrotel(
            project_dir=tmpdir,
            project_file="SLNO.csv",
            use_defaults=True,
            simulation_config=simulation_config,
        )

        if test == "ok":
            ht._basic_checks()
        elif test == "file":
            with pytest.raises(
                FileNotFoundError,
                match="lecture_tempsol.csv is mentioned in the configuration, but does not exist.",
            ):
                ht._basic_checks()
        elif test == "health":
            with pytest.warns(
                UserWarning,
                match="The following health checks failed:\n  "
                "- The dimension 'stations' is missing.\n  "
                "- The coordinate 'z' is missing.\n  "
                "- The calendar is not 'standard'. Received 'noleap'.\n  "
                "- The start date is not at least 1999-01-01 00:00:00. Received 2001-01-01 00:00:00.\n  "
                "- The variable 'tasmin' does not have the expected units 'degC'. Received 'K'.\n  "
                "- The timesteps are irregular or cannot be inferred by xarray.",
            ):
                ht._basic_checks()

    def test_standard(self, tmpdir):
        xhydro.testing.utils.fake_hydrotel_project(tmpdir, debit_aval=True)

        ht = Hydrotel(tmpdir, "SLNO.csv", use_defaults=True)
        with ht.get_streamflow() as ds_tmp:
            ds_orig = deepcopy(ds_tmp)
        ht._standardise_outputs()
        ds = ht.get_streamflow()

        # To make sure the original dataset was not modified prior to standardisation
        assert list(ds_orig.data_vars) == ["debit_aval"]
        np.testing.assert_array_equal(ds_orig.dims, ["time", "troncon"])

        assert list(ds.data_vars) == ["streamflow"]
        np.testing.assert_array_equal(ds.dims, ["time", "station_id"])
        correct_attrs = {
            "units": (
                "m^3 s-1" if parse(__xclim_version__) < parse("0.48.0") else "m3 s-1"
            ),
            "description": "Streamflow at the outlet of the river reach",
            "standard_name": "outgoing_water_volume_transport_along_river_channel",
            "long_name": "Streamflow",
            "_original_name": "debit_aval",
            "_original_description": "Debit en aval du troncon",
        }
        assert sorted(set(ds.streamflow.attrs)) == sorted(set(correct_attrs))
        for k, v in correct_attrs.items():
            assert ds.streamflow.attrs[k] == v

        assert "initial_simulation_path" not in ds.attrs

    def test_simname(self, tmpdir):
        xhydro.testing.utils.fake_hydrotel_project(tmpdir)
        with pytest.raises(ValueError, match="folder does not exist"):
            Hydrotel(
                tmpdir,
                "SLNO.csv",
                use_defaults=False,
                project_config={"SIMULATION COURANTE": "test"},
            )

        ht = Hydrotel(tmpdir, "SLNO.csv", use_defaults=False)
        with pytest.raises(ValueError, match="folder does not exist"):
            ht.update_config(project_config={"SIMULATION COURANTE": "test"})

        Path(tmpdir / "simulation" / "simulation").rename(
            tmpdir / "simulation" / "test",
        )
        Hydrotel(
            tmpdir,
            "SLNO.csv",
            use_defaults=True,
            project_config={"SIMULATION COURANTE": "test"},
        )

    def test_dates(self, tmpdir):
        xhydro.testing.utils.fake_hydrotel_project(tmpdir)
        ht = Hydrotel(
            tmpdir,
            "SLNO.csv",
            use_defaults=True,
            simulation_config={
                "DATE DEBUT": "2001-01-01",
                "DATE FIN": "2001-12-31 12",
                "LECTURE ETAT FONTE NEIGE": "2001-01-01 03",
                "ECRITURE ETAT FONTE NEIGE": "2001-01-01",
            },
        )
        assert ht.simulation_config["DATE DEBUT"] == "2001-01-01 00:00"
        assert ht.simulation_config["DATE FIN"] == "2001-12-31 12:00"
        assert ht.simulation_config["LECTURE ETAT FONTE NEIGE"] == str(
            Path("etat/fonte_neige_2001010103.csv")
        )
        assert ht.simulation_config["ECRITURE ETAT FONTE NEIGE"] == "2001-01-01 00"

    @pytest.mark.parametrize("test", ["ok", "pdt", "cfg"])
    def test_run(self, tmpdir, test):
        xhydro.testing.utils.fake_hydrotel_project(tmpdir, meteo=True)
        ht = Hydrotel(
            tmpdir,
            "SLNO.csv",
            use_defaults=False,
            simulation_config={
                "DATE DEBUT": "2001-01-01",
                "DATE FIN": "2001-12-31",
                "FICHIER STATIONS METEO": r"meteo\SLNO_meteo_GC3H.nc",
                "PAS DE TEMPS": 24 if test != "pdt" else None,
            },
        )

        if os.name == "nt":
            with pytest.raises(
                ValueError, match="You must specify the path to Hydrotel.exe"
            ):
                ht.run(dry_run=True)
        else:
            if test == "ok":
                command = ht.run(dry_run=True)
                assert command == f"hydrotel {ht.config_files['project']} -t 1"
            elif test == "pdt":
                with pytest.raises(
                    ValueError,
                    match="You must specify 'DATE DEBUT', 'DATE FIN', and 'PAS DE TEMPS'",
                ):
                    ht.run(dry_run=True)
            elif test == "cfg":
                cfg = (
                    pd.read_csv(
                        tmpdir / "meteo" / "SLNO_meteo_GC3H.nc.config",
                        sep=";",
                        header=None,
                        index_col=0,
                    )
                    .replace([np.nan], [None])
                    .squeeze()
                    .to_dict()
                )
                # 1: missing entry
                cfg_bad = deepcopy(cfg)
                cfg_bad["LATITUDE_NAME"] = ""
                pd.DataFrame.from_dict(cfg_bad, orient="index").to_csv(
                    tmpdir / "meteo" / "SLNO_meteo_GC3H.nc.config",
                    sep=";",
                    header=False,
                    columns=[0],
                )
                with pytest.raises(
                    ValueError, match="The configuration file is missing some entries:"
                ):
                    ht.run(dry_run=True)

                # 2: bad type
                cfg_bad = deepcopy(cfg)
                cfg_bad["TYPE (STATION/GRID/GRID_EXTENT)"] = "fake"
                pd.DataFrame.from_dict(cfg_bad, orient="index").to_csv(
                    tmpdir / "meteo" / "SLNO_meteo_GC3H.nc.config",
                    sep=";",
                    header=False,
                    columns=[0],
                )
                with pytest.raises(
                    ValueError,
                    match="The configuration file must specify the type of data",
                ):
                    ht.run(dry_run=True)

                # 3: bad station name
                cfg_bad = deepcopy(cfg)
                cfg_bad["TYPE (STATION/GRID/GRID_EXTENT)"] = "GRID"
                cfg_bad["STATION_DIM_NAME"] = "stations"
                pd.DataFrame.from_dict(cfg_bad, orient="index").to_csv(
                    tmpdir / "meteo" / "SLNO_meteo_GC3H.nc.config",
                    sep=";",
                    header=False,
                    columns=[0],
                )
                with pytest.raises(
                    ValueError,
                    match="STATION_DIM_NAME must be specified if and only if",
                ):
                    ht.run(dry_run=True)

    def test_copypaste(self, tmpdir):
        xhydro.testing.utils.fake_hydrotel_project(tmpdir)
        # Remove simulation.csv
        Path(tmpdir / "simulation" / "simulation" / "simulation.csv").unlink()
        Hydrotel(
            tmpdir,
            "SLNO.csv",
            use_defaults=True,
        )
        assert Path(tmpdir / "simulation" / "simulation" / "simulation.csv").exists()

    def test_errors(self, tmpdir):
        # Missing project folder
        with pytest.raises(ValueError, match="The project folder does not exist."):
            Hydrotel("fake", "SLNO.csv", use_defaults=True)

        # Missing project name
        xhydro.testing.utils.fake_hydrotel_project(tmpdir)
        project_config = {
            "SIMULATION COURANTE": "",
        }
        _overwrite_csv(tmpdir / "SLNO.csv", project_config)
        with pytest.raises(
            ValueError,
            match="'SIMULATION COURANTE' must be specified",
        ):
            Hydrotel(tmpdir, "SLNO.csv", use_defaults=False)

    def test_bad_config(self, tmpdir):
        xhydro.testing.utils.fake_hydrotel_project(tmpdir)
        # overwrite output.csv with simulation.csv
        shutil.copy(
            tmpdir / "simulation" / "simulation" / "simulation.csv",
            tmpdir / "simulation" / "simulation" / "output.csv",
        )
        # Multiple warnings should be raised, so we use a list
        with pytest.warns(UserWarning) as record:
            Hydrotel(tmpdir, "SLNO.csv", use_defaults=False)
        assert len(record) == 2
        assert (
            "configuration file on disk has some entries that might not be valid."
            in record[0].message.args[0]
        )
        assert (
            "file on disk has a different number of entries than the template."
            in record[1].message.args[0]
        )

    def test_bad_overwrite(self, tmpdir):
        xhydro.testing.utils.fake_hydrotel_project(tmpdir)
        # overwrite output.csv with simulation.csv
        with pytest.raises(
            ValueError, match="Could not find the following keys in the template file"
        ):
            _overwrite_csv(
                tmpdir / "simulation" / "simulation" / "output.csv",
                {"foo": "bar"},
            )
