import datetime


def get_julian_day(month, day, year=None):
    """
    Return julian day for a specified date, if year is not specified, uses curent year

    Parameters
    ----------
    month : int
        integer of the target month

    day : int
        integer of the target day

    year : int
        integer of the target year

    Returns
    -------
    int
        julian day (1 - 366)

    Examples
    --------
    >>> import xarray as xr
    >>> cehq_data_path = "/dbfs/mnt/devdlzxxkp01/datasets/xhydro/tests/cehq/zarr"
    >>> ds = xr.open_zarr(cehq_data_path, consolidated=True)
    >>> donnees = Data(ds)
    >>> jj = donnees.get_julian_day(month=9, day=1)
    >>> jj: 244
    >>> jj = donnees.get_julian_day(month=9, day=1, year=2000)
    >>> jj: 245
    """
    if year is None:
        year = datetime.date.today().year

    return datetime.datetime(year, month, day).timetuple().tm_yday


def get_timestep(array_in):
    if len(array_in) < 2:
        # Returns a timestep of one for a one timestep array
        return 1
    timestep = ((array_in[-1] - array_in[0]) / (array_in.size - 1)).values.astype(
        "timedelta64[m]"
    )
    if timestep >= 60 and timestep < 24 * 60:
        timestep = timestep.astype("timedelta64[h]")
    elif timestep >= 24 * 60:
        timestep = timestep.astype("timedelta64[D]")
    return timestep
