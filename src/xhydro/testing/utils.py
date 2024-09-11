"""Utilities for testing and releasing xhydro."""

import os
import re
import shutil
from io import StringIO
from pathlib import Path
from typing import Optional, TextIO, Union

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from xclim.testing.helpers import test_timeseries as timeseries

__all__ = [
    "fake_hydrotel_project",
    "publish_release_notes",
]


def fake_hydrotel_project(
    project_dir: str | os.PathLike,
    *,
    meteo: bool | xr.Dataset = False,
    debit_aval: bool | xr.Dataset = False,
):
    """Create a fake Hydrotel project in the given directory.

    Parameters
    ----------
    project_dir : str or os.PathLike
        Directory where the project will be created.
    meteo : bool or xr.Dataset
        Fake meteo timeseries. If True, a 2-year timeseries is created. Alternatively, provide a Dataset.
        Leave as False to create a fake file.
    debit_aval : bool or xr.Dataset
        Fake debit_aval timeseries. If True, a 2-year timeseries is created. Alternatively, provide a Dataset.
        Leave as False to create a fake file.

    Notes
    -----
    Uses the directory structure specified in xhydro/testing/data/hydrotel_structure.yml.
    Most files are fake, except for the projet.csv, simulation.csv and output.csv files, which are filled with
    default options taken from xhydro/modelling/data/hydrotel_defaults/.
    """
    project_dir = Path(project_dir)

    with (
        Path(__file__)
        .parent.joinpath("data")
        .joinpath("hydrotel_structure.yml")
        .open() as f
    ):
        struc = yaml.safe_load(f)["structure"]

    default_csv = (
        Path(__file__).parent.parent / "modelling" / "data" / "hydrotel_defaults"
    )

    project_dir.mkdir(exist_ok=True, parents=True)
    for k, v in struc.items():
        if k != "_main":
            project_dir.joinpath(k).mkdir(exist_ok=True, parents=True)
            for file in v:
                if file in ["simulation.csv", "output.csv"]:
                    shutil.copy(default_csv / file, project_dir / k / file)
                elif file is not None and Path(file).suffix not in [".nc", ".config"]:
                    (project_dir / k / file).touch()
    for file in struc["_main"]:
        if file in ["SLNO.csv"]:
            shutil.copy(default_csv / "project.csv", project_dir / file)
        elif file is not None:
            (project_dir / file).touch()

    # Create fake meteo and debit_aval files
    if isinstance(meteo, bool) and meteo:
        meteo = timeseries(
            np.zeros(365 * 2),
            start="2001-01-01",
            freq="D",
            variable="tasmin",
            as_dataset=True,
            units="degC",
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
        meteo = meteo.assign_coords(coords={"lat": 46, "lon": -77, "z": 0})
        for c in ["lat", "lon", "z"]:
            meteo[c] = meteo[c].expand_dims("stations")
    if isinstance(meteo, xr.Dataset):
        meteo.to_netcdf(project_dir / "meteo" / "SLNO_meteo_GC3H.nc")
        cfg = pd.Series(
            {
                "TYPE (STATION/GRID/GRID_EXTENT)": "STATION",
                "STATION_DIM_NAME": "stations",
                "LATITUDE_NAME": "lat",
                "LONGITUDE_NAME": "lon",
                "ELEVATION_NAME": "z",
                "TIME_NAME": "time",
                "TMIN_NAME": "tasmin",
                "TMAX_NAME": "tasmax",
                "PRECIP_NAME": "pr",
            }
        )
        cfg.to_csv(
            project_dir / "meteo" / "SLNO_meteo_GC3H.nc.config",
            sep=";",
            header=False,
            columns=[0],
        )
    else:
        (project_dir / "meteo" / "SLNO_meteo_GC3H.nc").touch()
        (project_dir / "meteo" / "SLNO_meteo_GC3H.nc.config").touch()

    if isinstance(debit_aval, bool) and debit_aval:
        debit_aval = timeseries(
            np.zeros(365 * 2),
            start="2001-01-01",
            freq="D",
            variable="streamflow",
            as_dataset=True,
        )
        debit_aval = debit_aval.expand_dims("troncon").assign_coords(troncon=[0])
        debit_aval = debit_aval.assign_coords(coords={"idtroncon": 0})
        debit_aval["idtroncon"] = debit_aval["idtroncon"].expand_dims("troncon")
        debit_aval = debit_aval.rename({"streamflow": "debit_aval"})
        debit_aval["debit_aval"].attrs = {
            "units": "m3/s",
            "description": "Debit en aval du troncon",
        }
        # Add attributes to the dataset
        debit_aval.attrs = {
            "initial_simulation_path": "path/to/initial/simulation",
        }
    if isinstance(debit_aval, xr.Dataset):
        debit_aval.to_netcdf(
            project_dir / "simulation" / "simulation" / "resultat" / "debit_aval.nc"
        )
    else:
        (
            project_dir / "simulation" / "simulation" / "resultat" / "debit_aval.nc"
        ).touch()


def publish_release_notes(
    style: str = "md",
    file: os.PathLike | StringIO | TextIO | None = None,
    changes: str | os.PathLike | None = None,
) -> str | None:
    """Format release history in Markdown or ReStructuredText.

    Parameters
    ----------
    style : {"rst", "md"}
        Use ReStructuredText (`rst`) or Markdown (`md`) formatting. Default: Markdown.
    file : {os.PathLike, StringIO, TextIO}, optional
        If provided, prints to the given file-like object. Otherwise, returns a string.
    changes : {str, os.PathLike}, optional
        If provided, manually points to the file where the changelog can be found.
        Assumes a relative path otherwise.

    Returns
    -------
    str or None
        Formatted release notes as a string, if `file` is not provided.

    Notes
    -----
    This function exists solely for development purposes.
    Adapted from xclim.testing.utils.publish_release_notes.
    """
    if isinstance(changes, str | Path):
        changes_file = Path(changes).absolute()
    else:
        changes_file = Path(__file__).absolute().parents[2].joinpath("CHANGELOG.rst")

    if not changes_file.exists():
        raise FileNotFoundError("Changes file not found in xhydro file tree.")

    with changes_file.open() as hf:
        changes = hf.read()

    if style == "rst":
        hyperlink_replacements = {
            r":issue:`([0-9]+)`": r"`GH/\1 <https://github.com/hydrologie/xhydro/issues/\1>`_",
            r":pull:`([0-9]+)`": r"`PR/\1 <https://github.com/hydrologie/xhydro/pull/\>`_",
            r":user:`([a-zA-Z0-9_.-]+)`": r"`@\1 <https://github.com/\1>`_",
        }
    elif style == "md":
        hyperlink_replacements = {
            r":issue:`([0-9]+)`": r"[GH/\1](https://github.com/hydrologie/xhydro/issues/\1)",
            r":pull:`([0-9]+)`": r"[PR/\1](https://github.com/hydrologie/xhydro/pull/\1)",
            r":user:`([a-zA-Z0-9_.-]+)`": r"[@\1](https://github.com/\1)",
        }
    else:
        raise NotImplementedError(f"Style {style} not implemented.")

    for search, replacement in hyperlink_replacements.items():
        changes = re.sub(search, replacement, changes)

    if style == "md":
        changes = changes.replace("=========\nChangelog\n=========", "# Changelog")

        titles = {r"\n(.*?)\n([\-]{1,})": "-", r"\n(.*?)\n([\^]{1,})": "^"}
        for title_expression, level in titles.items():
            found = re.findall(title_expression, changes)
            for grouping in found:
                fixed_grouping = (
                    str(grouping[0]).replace("(", r"\(").replace(")", r"\)")
                )
                search = rf"({fixed_grouping})\n([\{level}]{'{' + str(len(grouping[1])) + '}'})"
                replacement = f"{'##' if level == '-' else '###'} {grouping[0]}"
                changes = re.sub(search, replacement, changes)

        link_expressions = r"[\`]{1}([\w\s]+)\s<(.+)>`\_"
        found = re.findall(link_expressions, changes)
        for grouping in found:
            search = rf"`{grouping[0]} <.+>`\_"
            replacement = f"[{str(grouping[0]).strip()}]({grouping[1]})"
            changes = re.sub(search, replacement, changes)

    if not file:
        return changes
    if isinstance(file, os.PathLike):
        file = Path(file).open("w")
    print(changes, file=file)
