import os
import yaml
from pathlib import Path, PureWindowsPath
import subprocess
from copy import deepcopy

import pandas as pd
import xarray as xr
import xclim as xc

from .data_checks import health_check


def initialize(project: str | os.PathLike,
               *,
               simulation_file: str = 'simulation.csv',
               simulation_options: dict = None,
               output_file: str = 'output.csv',
               output_options: dict = None,
               checkups: bool = False,
               checkup_kwargs: dict = None) -> tuple[dict, dict]:
    """
    Initialize the simulation and output files for Hydrotel, and optionally run health checkups on the weather input.

    Parameters
    ----------
    project: str | os.PathLike
        Path to the project folder.
    simulation_file: str
        Name of the simulation file.
    simulation_options: dict
        Dictionary of options to change in the simulation file.
    output_file: str
        Name of the output file.
    output_options: dict
        Dictionary of options to change in the output file.
    checkups: bool
        Whether to run health checkups on the weather input.
    checkup_kwargs: dict
        Keyword arguments to pass to the :py:func:`~xhydro.modelling.data_checks.health_check` function.

    Returns
    -------
    sim_options: dict
        Dictionary of simulation options used.
    out_options: dict
        Dictionary of output options used.

    """
    # If the suffix is missing, add it
    if not Path(simulation_file).suffix:
        simulation_file += '.csv'
    if not Path(output_file).suffix:
        output_file += '.csv'

    # Load from the files, if they exist. Otherwise use default values
    if os.path.isfile(Path(project) / 'simulation' / 'simulation' / simulation_file):
        sim_options = _fix_os_paths(pd.read_csv(Path(project) / 'simulation' / 'simulation' / simulation_file, delimiter=";", header=None,
                                                index_col=0).squeeze().to_dict())
    else:
        with open(Path(__file__).parent / "hydrotel_defaults.yml") as f:
            sim_options = yaml.safe_load(f)["simulation_options"]
    if os.path.isfile(Path(project) / 'simulation' / 'simulation' / output_file):
        out_options = pd.read_csv(Path(project) / 'simulation' / 'simulation' / output_file, delimiter=";", header=None, index_col=0).squeeze().to_dict()
    else:
        with open(Path(__file__).parent / "hydrotel_defaults.yml") as f:
            out_options = yaml.safe_load(f)["output_options"]

    # Update options with the user-provided ones
    simulation_options = deepcopy(_fix_os_paths(simulation_options)) or {}
    for key, value in simulation_options.items():
        sim_options[key] = value

    output_options = deepcopy(output_options) or {}
    for key, value in output_options.items():
        out_options[key] = value

    # Reformat dates
    for key in ['DATE DEBUT', 'DATE FIN']:
        if key in sim_options and not pd.isnull(sim_options[key]):
            sim_options[key] = pd.to_datetime(sim_options[key]).strftime('%Y-%m-%d %H:%M')
    # FIXME: Validate that this part is useful like this
    for key in ['LECTURE ETAT FONTE NEIGE', 'LECTURE ETAT TEMPERATURE DU SOL', 'LECTURE ETAT BILAN VERTICAL',
                'LECTURE ETAT RUISSELEMENT SURFACE', 'LECTURE ETAT ACHEMINEMENT RIVIERE']:
        if key in sim_options and not pd.isnull(sim_options[key]):
            sim_options[key] = sim_options[key].replace(Path(sim_options[key]).stem.split("_")[-1],
                                                        pd.to_datetime(Path(sim_options[key]).stem.split("_")[-1]).strftime('%Y%m%d%H'))
    for key in ['ECRITURE ETAT FONTE NEIGE', 'ECRITURE ETAT TEMPERATURE DU SOL', 'ECRITURE ETAT BILAN VERTICAL',
                'ECRITURE ETAT RUISSELEMENT SURFACE', 'ECRITURE ETAT ACHEMINEMENT RIVIERE']:
        if key in sim_options and not pd.isnull(sim_options[key]):
            # FIXME: Do we realy need to get rid of the minutes?
            sim_options[key] = pd.to_datetime(sim_options[key]).strftime('%Y-%m-%d %H')

    # Make sure that the start and end dates of the simulation are contained in the meteo file
    start_date = pd.to_datetime(sim_options.get('DATE DEBUT', None))
    end_date = pd.to_datetime(sim_options.get('DATE FIN', None))
    weather_file = Path(project) / sim_options.get('FICHIER STATIONS METEO', None)
    ds = xr.open_dataset(weather_file, cache=False)
    if not ((ds.time[0] <= start_date) and (ds.time[-1] >= end_date)).item():
        raise ValueError(f'The start date ({start_date}) or end date ({end_date}) are outside the bounds of the weather file ({weather_file}).')

    if checkups:
        checkup_kwargs = checkup_kwargs or {}
        health_check(ds, model='hydrotel', **checkup_kwargs)

    # Save the simulation options to a file
    df = pd.DataFrame.from_dict(sim_options, orient='index')
    df = df.replace({None: ''})
    df.to_csv(Path(project) / 'simulation' / 'simulation' / simulation_file, sep=';', header=False, columns=[0])

    # Save the output options to a file
    df = pd.DataFrame.from_dict(output_options, orient='index')
    df.to_csv(Path(project) / 'simulation' / 'simulation' / output_file, sep=';', header=False)

    # TODO: Clean up 'etat' folder (missing the files tor really test it)

    return deepcopy(sim_options), deepcopy(out_options)


def run(project: str | os.PathLike,
        *,
        hydrotel_console: str | os.PathLike = None):
    """
    Run the simulation.

    Parameters
    ----------
    project: str | os.PathLike
        Path to the project folder.
    hydrotel_console: str | os.PathLike
        On Windows, path to Hydrotel.exe.

    Returns
    -------

    """
    if os.name == 'nt':  # Windows
        if hydrotel_console is None:
            raise ValueError('You must specify the path to Hydrotel.exe')
    else:
        hydrotel_console = 'hydrotel'

    # TODO: Test it out
    subprocess.check_call(hydrotel_console + ' ' + project + ' -t 1')

    # TODO: Probably combine with standardize_outputs


def standardize_outputs(project: str | os.PathLike,
                        *,
                        streamflow_file: str = 'debit_aval.nc',
                        swe_file: str = 'debit_aval.nc'):
    """

    Parameters
    ----------
    project: str | os.PathLike
        Path to the project folder.
    streamflow_file: str
        Name of the streamflow file.
    swe_file: str
        Name of the SWE file.

    Returns
    -------

    """
    ds = xr.open_dataset(Path(project) / 'simulation' / 'simulation' / 'resultat' / streamflow_file)

    # Rename variables to standard names
    ds = ds.rename_vars({'idtroncon': 'id', 'debit_aval': 'streamflow'})
    ds = ds.assign_coords(id=ds.id.astype(str))
    ds = ds.swap_dims({'troncon': 'id'})

    # Add standard attributes for streamflow and fix units
    ds['streamflow'].attrs['original_name'] = 'debit_aval'
    for attr in ['standard_name', 'long_name', 'description', 'units']:
        if attr in ds['streamflow'].attrs:
            ds['streamflow'].attrs[f"original_{attr}"] = ds['streamflow'].attrs[attr]
    ds['streamflow'].attrs['standard_name'] = 'outgoing_water_volume_transport_along_river_channel'
    ds['streamflow'].attrs['long_name'] = 'Streamflow'
    ds['streamflow'].attrs['description'] = 'Streamflow at the outlet of the river reach'
    ds['streamflow'] = xc.units.convert_units_to(ds['streamflow'], 'm3 s-1')

    # Add standard attributes for SWE and fix units
    # TODO: (currently missing the file)

    return ds


def _fix_os_paths(d: dict):
    """ Convert paths to fit the OS.
    """
    # FIXME: Ugly fix to switch Windows paths to the right OS. Wouldn't work otherwise.
    return {k: str(Path(PureWindowsPath(v).as_posix())) if any(slash in str(v) for slash in ['/', '\\']) else v for k, v in d.items()}
