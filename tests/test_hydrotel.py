import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xclim.testing.helpers import test_timeseries as timeseries

import xhydro
from xhydro.modelling import Hydrotel


class TestHydrotel:
    def test_options(self, tmpdir):
        xhydro.testing.utils.fake_hydrotel_project(tmpdir, "fake")
        ht = Hydrotel(
            tmpdir / "fake",
            default_options=True,
            project_options={"PROJET HYDROTEL VERSION": "2.1.0"},
            simulation_options={"SIMULATION HYDROTEL VERSION": "1.0.5"},
            output_options={"TMAX_JOUR": 1},
        )

        assert ht.simulation_name == "simulation"
        assert ht.project.name == "fake"

        # Check that the options have been updated and that the files have been overwritten
        assert ht.project_options["PROJET HYDROTEL VERSION"] == "2.1.0"
        df = (
            pd.read_csv(ht.project / "projet.csv", sep=";", header=None, index_col=0)
            .replace([np.nan], [None])
            .to_dict()[1]
        )
        assert ht.project_options == df

        assert ht.simulation_options["SIMULATION HYDROTEL VERSION"] == "1.0.5"
        df = (
            pd.read_csv(
                ht.project / "simulation" / ht.simulation_name / "simulation.csv",
                sep=";",
                header=None,
                index_col=0,
            )
            .replace([np.nan], [None])
            .to_dict()[1]
        )
        # The simulation options are not the same as the ones in the file because entries in the file are read as strings
        simopt = (
            pd.DataFrame.from_dict(ht.simulation_options.copy(), orient="index")
            .astype(str)
            .replace("None", None)
            .to_dict()[0]
        )
        assert simopt == df

        assert ht.output_options["TMAX_JOUR"] == 1
        df = (
            pd.read_csv(
                ht.project / "simulation" / ht.simulation_name / "output.csv",
                sep=";",
                header=None,
                index_col=0,
            )
            .replace([np.nan], [None])
            .to_dict()[1]
        )
        # The output options are not the same as the ones in the file because entries in the file are read as strings
        outopt = (
            pd.DataFrame.from_dict(ht.output_options.copy(), orient="index")
            .astype(str)
            .replace("None", None)
            .to_dict()[0]
        )
        assert outopt == df

        ht2 = Hydrotel(tmpdir / "fake", default_options=False)
        assert ht2.project_options == ht.project_options
        assert ht2.simulation_options == simopt
        assert ht2.output_options == outopt

    @pytest.mark.parametrize("test", ["station", "grid", "none", "toomany"])
    def test_get_data(self, tmpdir, test):
        xhydro.testing.utils.fake_hydrotel_project(
            tmpdir, "fake", meteo=True, debit_aval=True
        )
        if test == "station":
            simulation_options = {"FICHIER STATIONS METEO": r"meteo\SLNO_meteo_GC3H.nc"}
        elif test == "grid":
            simulation_options = {"FICHIER GRILLE METEO": r"meteo\SLNO_meteo_GC3H.nc"}
        elif test == "none":
            simulation_options = {}
        else:
            simulation_options = {
                "FICHIER STATIONS METEO": r"meteo\SLNO_meteo_GC3H.nc",
                "FICHIER GRILLE METEO": r"meteo\SLNO_meteo_GC3H.nc",
            }

        ht = Hydrotel(
            tmpdir / "fake",
            default_options=True,
            simulation_options=simulation_options,
        )
        if test in ["station", "grid"]:
            ds = ht.get_input()
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
                ht.get_input()
        else:
            with pytest.raises(
                ValueError,
                match="You must specify either 'FICHIER GRILLE METEO' or 'FICHIER STATIONS METEO'",
            ):
                ht.get_input()

    @pytest.mark.parametrize("test", ["ok", "option", "file", "health"])
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

        xhydro.testing.utils.fake_hydrotel_project(tmpdir, "fake", meteo=meteo)
        date_debut = "2001-10-01" if test != "health" else "1999-01-01"
        simulation_options = {
            "FICHIER STATIONS METEO": r"meteo\SLNO_meteo_GC3H.nc",
            "DATE DEBUT": date_debut,
            "DATE FIN": "2002-12-30",
            "PAS DE TEMPS": 24,
        }
        if test == "option":
            df = pd.DataFrame.from_dict(simulation_options, orient="index")
            df = df.replace({None: ""})
            df.to_csv(
                tmpdir / "fake" / "simulation" / "simulation" / "simulation.csv",
                sep=";",
                header=False,
                columns=[0],
            )
        elif test == "file":
            os.remove(
                tmpdir / "fake" / "simulation" / "simulation" / "lecture_tempsol.csv"
            )

        ht = Hydrotel(
            tmpdir / "fake",
            default_options=True if test != "option" else False,
            simulation_options=simulation_options,
        )

        if test == "ok":
            ht._basic_checks()
        elif test == "option":
            with pytest.raises(
                ValueError, match="is missing from the simulation file."
            ):
                ht._basic_checks()
        elif test == "file":
            with pytest.raises(
                FileNotFoundError, match="lecture_tempsol.csv does not exist."
            ):
                ht._basic_checks()
        elif test == "health":
            with pytest.raises(
                ValueError,
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
        xhydro.testing.utils.fake_hydrotel_project(tmpdir, "fake", debit_aval=True)

        ht = Hydrotel(tmpdir / "fake", default_options=True)
        ds_orig = deepcopy(ht.get_streamflow())
        ht._standardise_outputs()
        ds = ht.get_streamflow()

        # To make sure the original dataset was not modified prior to standardisation
        assert list(ds_orig.data_vars) == ["debit_aval"]
        np.testing.assert_array_equal(ds_orig.dims, ["time", "troncon"])

        assert list(ds.data_vars) == ["streamflow"]
        np.testing.assert_array_equal(ds.dims, ["time", "station_id"])
        correct_attrs = {
            "units": "m^3 s-1",
            "description": "Streamflow at the outlet of the river reach",
            "standard_name": "outgoing_water_volume_transport_along_river_channel",
            "long_name": "Streamflow",
            "original_name": "debit_aval",
            "original_description": "Debit en aval du troncon",
        }
        assert sorted(set(ds.streamflow.attrs)) == sorted(set(correct_attrs))
        for k, v in correct_attrs.items():
            assert ds.streamflow.attrs[k] == v

    def test_simname(self, tmpdir):
        xhydro.testing.utils.fake_hydrotel_project(tmpdir, "fake")
        with pytest.raises(
            ValueError, match="The 'simulation/test/' folder does not exist"
        ):
            Hydrotel(
                tmpdir / "fake",
                default_options=False,
                project_options={"SIMULATION COURANTE": "test"},
            )

        ht = Hydrotel(tmpdir / "fake", default_options=False)
        with pytest.raises(
            ValueError, match="The 'simulation/test/' folder does not exist"
        ):
            ht.update_options(project_options={"SIMULATION COURANTE": "test"})

        os.rename(
            tmpdir / "fake" / "simulation" / "simulation",
            tmpdir / "fake" / "simulation" / "test",
        )
        Hydrotel(
            tmpdir / "fake",
            default_options=True,
            project_options={"SIMULATION COURANTE": "test"},
        )

    def test_dates(self, tmpdir):
        xhydro.testing.utils.fake_hydrotel_project(tmpdir, "fake")
        ht = Hydrotel(
            tmpdir / "fake",
            default_options=True,
            simulation_options={
                "DATE DEBUT": "2001-01-01",
                "DATE FIN": "2001-12-31 12",
                "LECTURE ETAT FONTE NEIGE": "2001-01-01 03",
                "ECRITURE ETAT FONTE NEIGE": "2001-01-01",
            },
        )
        assert ht.simulation_options["DATE DEBUT"] == "2001-01-01 00:00"
        assert ht.simulation_options["DATE FIN"] == "2001-12-31 12:00"
        assert ht.simulation_options["LECTURE ETAT FONTE NEIGE"] == str(
            Path("etat/fonte_neige_2001010103.csv")
        )
        assert ht.simulation_options["ECRITURE ETAT FONTE NEIGE"] == "2001-01-01 00"

    @pytest.mark.parametrize("test", ["ok", "pdt", "cfg"])
    def test_run(self, tmpdir, test):
        xhydro.testing.utils.fake_hydrotel_project(tmpdir, "fake", meteo=True)
        ht = Hydrotel(
            tmpdir / "fake",
            default_options=False,
            simulation_options={
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
                assert command == f"hydrotel {ht.project} -t 1"
            elif test == "pdt":
                with pytest.raises(
                    ValueError,
                    match="You must specify 'DATE DEBUT', 'DATE FIN', and 'PAS DE TEMPS'",
                ):
                    ht.run(dry_run=True)
            elif test == "cfg":
                cfg = (
                    pd.read_csv(
                        tmpdir / "fake" / "meteo" / "SLNO_meteo_GC3H.nc.config",
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
                    tmpdir / "fake" / "meteo" / "SLNO_meteo_GC3H.nc.config",
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
                cfg_bad["TYPE (STATION/GRID)"] = "fake"
                pd.DataFrame.from_dict(cfg_bad, orient="index").to_csv(
                    tmpdir / "fake" / "meteo" / "SLNO_meteo_GC3H.nc.config",
                    sep=";",
                    header=False,
                    columns=[0],
                )
                with pytest.raises(
                    ValueError,
                    match="The configuration file must specify 'STATION' or 'GRID'",
                ):
                    ht.run(dry_run=True)

                # 3: bad station name
                cfg_bad = deepcopy(cfg)
                cfg_bad["TYPE (STATION/GRID)"] = "GRID"
                cfg_bad["STATION_DIM_NAME"] = "stations"
                pd.DataFrame.from_dict(cfg_bad, orient="index").to_csv(
                    tmpdir / "fake" / "meteo" / "SLNO_meteo_GC3H.nc.config",
                    sep=";",
                    header=False,
                    columns=[0],
                )
                with pytest.raises(
                    ValueError,
                    match="STATION_DIM_NAME must be specified if and only if",
                ):
                    ht.run(dry_run=True)

    def test_errors(self, tmpdir):
        # Missing project folder
        with pytest.raises(ValueError, match="The project folder does not exist."):
            Hydrotel("fake", default_options=True)

        # Missing project name
        xhydro.testing.utils.fake_hydrotel_project(tmpdir, "fake")
        project_options = {
            "FICHIER ALTITUDE": "physitel/altitude.tif",
            "FICHIER PENTE": "physitel/pente.tif",
        }
        df = pd.DataFrame.from_dict(project_options, orient="index")
        df = df.replace({None: ""})
        df.to_csv(
            tmpdir / "fake" / "projet.csv",
            sep=";",
            header=False,
            columns=[0],
        )
        with pytest.raises(
            ValueError,
            match="If not using default options, 'SIMULATION COURANTE' must be specified in the project files or as a keyword argument in 'project_options'.",
        ):
            Hydrotel(tmpdir / "fake", default_options=False)
