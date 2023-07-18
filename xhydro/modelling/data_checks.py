import xclim as xc
import xclim.core.dataflags
import xarray as xr


# FIXME: Make this function more generic, missing is not working
def health_check(ds,
                 model: str,
                 *,
                 calendar: str = None,
                 missing: str = None,
                 flags: dict = None,
                 raise_flags: bool = True):

    # Check the dimensions and coordinates
    if model == 'hydrotel':
        if "stations" not in ds.dims:
            raise ValueError("The dimension 'stations' is missing.")
        # TODO: What is important in lat/lon/x/y/z ? Do we raise an error if they are missing?
        for coord in ['lat', 'lon', 'x', 'y', 'z']:
            if coord not in ds.coords and coord in ds:
                ds = ds.assign_coords({coord: ds[coord]})

    else:
        raise NotImplementedError(f'The model {model} is not implemented.')

    # Check the calendar
    if calendar is not None:
        # TODO: Support calendars with multiple names (standard, gregorian, proleptic_gregorian)
        if ds.time.encoding['calendar'] != calendar:
            raise ValueError(f"The calendar is not {calendar}.")

    # Quick check for irregular time steps
    freq = xr.infer_freq(ds.time)
    if freq is None:
        raise ValueError("The timesteps are irregular and cannot be inferred.")

    # Check variables
    if model == 'hydrotel':
        variables = {'pr': 'mm', 'tasmin': 'degC', 'tasmax': 'degC'}
    else:
        raise NotImplementedError(f'The model {model} is not implemented.')

    for v in variables:
        if v not in ds:
            raise ValueError(f"The variable {v} is missing.")
        if ds[v].attrs['units'] != variables[v]:
            ds[v] = xc.units.convert_units_to(ds[v], variables[v])

        # xc.core.missing.missing_any(ds[v], '3h')

        if v in flags:
            xclim.core.dataflags.data_flags(ds[v], ds, flags=flags[v], raise_flags=raise_flags)
