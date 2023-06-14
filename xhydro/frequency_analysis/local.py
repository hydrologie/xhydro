import xarray as xr
import numpy as np
import warnings
import copy
from typing import Union
import pandas as pd
import fnmatch


class Data:
    def __init__(self, ds: xr.Dataset):
        """init function takes a dataset as input and initialize an empty
        season dictionary and a list of catchments from dimension id

        Parameters
        ----------
        ds : xr.Dataset
            _description_
        """
        self.data = ds
        self._season = {}
        self._catchments = self.data.id.to_numpy().tolist()

    def _repr_html_(self):
        """Function to show xr.Dataset._repr_html_ when looking at class Data

        Returns
        -------
        xr.Dataset._repr_html_()
        """
        return self.data._repr_html_()

    def copy(self):
        """makes a copy of itself using copy library

        Returns
        -------
        xhydro.frequency_analysis.local.Data()
        """
        return copy.copy(self)

    def select_catchments(self, catchment_list: list):
        """
        select specified catchements from attribute data.
        Also supports the use of a wildcard (*).

        Parameters
        ----------
        catchment_list : List
        List of catchments that will be selcted along the id dimension

        Returns
        -------
        ds : xarray.DataSet
        New dataset with only specified catchments

        Examples
        --------
        >>> import xarray as xr
        >>> cehq_data_path =
            '/dbfs/mnt/devdlzxxkp01/datasets/xhydro/tests/cehq/zarr'
        >>> ds = xr.open_zarr(cehq_data_path, consolidated=True)
        >>> donnees = Data(ds)
        >>> filtered_ds = donnees.select_catchments(
            catchment_list = ['051001','051005'])
        >>> filtered_ds = donnees.select_catchments(
            catchment_list = ['05*','0234*', '023301'])
        """

        # Create a copy of the object
        obj = self.copy()

        def multi_filter(names, patterns: list):
            # sub function to select complete list based on wilcards
            return [
                name
                for name in names
                for pattern in patterns
                if fnmatch.fnmatch(name, pattern)
            ]

        # Getting the full list
        catchment_list = multi_filter(obj.catchments, catchment_list)

        # Setting the list as a class attribute
        obj._catchments = catchment_list

        # Filtering over the list
        obj.data = obj.data.sel(id=self.data.id.isin(catchment_list))
        return obj

    def custom_group_by(self, beg: int, end: int):
        """
        a custum fonction to groupby with specified julian days.

        Parameters
        ----------
        beg : Int
        Julian day of the begining of the period

        end : Int
        Julian day of the end of the period

        Returns
        -------
        ds : xarray.DataSet
        dataset with data grouped by time over specified dates

        Examples
        --------
        >>> import xarray as xr
        >>> cehq_data_path =
            '/dbfs/mnt/devdlzxxkp01/datasets/xhydro/tests/cehq/zarr'
        >>> ds = xr.open_zarr(cehq_data_path, consolidated=True)
        >>> donnees = Data(ds)
        >>> grouped_ds = donnees.custom_group_by(1, 90)

        """

        if beg > end:
            # TODO when beg `end, it means it has to overlap years,
            # for example, from octobre to march
            pass
        else:
            # +1 to include the end
            return self.data.sel(
                time=np.isin(self.data.time.dt.dayofyear, range(beg, end + 1))
            ).groupby("time.year")

    @property
    def season(self):
        return self._season

    @season.setter
    def season(self, liste: list):
        """
        The setter for the season property Issues a Warining
        if a new season is overlapping with another one.
        AlsO issues a warning if the season name was already used,
        and then overwrites it.

        Parameters
        ----------
        liste : List
        List of Name, begining in Julian day, end in Julian day

        Returns
        -------
        ds : xarray.DataSet
        appends Dict self._season with name as key,
        and begining and end as values

        Examples
        --------
        >>> import xarray as xr
        >>> cehq_data_path =
        '/dbfs/mnt/devdlzxxkp01/datasets/xhydro/tests/cehq/zarr'
        >>> ds = xr.open_zarr(cehq_data_path, consolidated=True)
        >>> donnees = Data(ds)
        >>> donnees.season = ['Yearly', 1, 365]
        """

        # TODO a dictionary would probalby be more suited
        name = liste[0]
        beg = liste[1]
        end = liste[2]
        for season, dates in self.season.items():
            if not isinstance(dates, xr.Dataset):
                # We dont check for overlapping if season is a dataset
                if name == season:
                    warnings.warn(
                        "Warning, "
                        + name
                        + " was already defined and has been overwritten"
                    )
                elif (
                    dates[0] <= beg
                    and dates[1] >= beg
                    or dates[0] <= end
                    and dates[1] >= end
                ):
                    warnings.warn("Warning, " + name + " overlapping with " + season)

        self._season[name] = [beg, end]

    def rm_season(self, name: str):
        """
        Fonction to remove a season.
        Isues a Warining if the name is not a valid season.

        Parameters
        ----------
        name : Str
        Name of the season to be removed

        Returns
        -------
        ds : xarray.DataSet
        The dataset is returned with =out the season specified.

        Examples
        --------
        >>> import xarray as xr
        >>> cehq_data_path =
        '/dbfs/mnt/devdlzxxkp01/datasets/xhydro/tests/cehq/zarr'
        >>> ds = xr.open_zarr(cehq_data_path, consolidated=True)
        >>> donnees = Data(ds)
        >>> donnees.season = ['Yearly', 1, 365]
        >>> donnees.rm_season('Yearly')
        """

        try:
            del self._season[name]
        except:
            print("No season named " + name)

    def get_seasons(self):
        """
        Function to list the seasons.

        Returns
        -------
        list : List
        a list of the season names

        Examples
        --------
        >>> import xarray as xr
        >>> cehq_data_path =
        '/dbfs/mnt/devdlzxxkp01/datasets/xhydro/tests/cehq/zarr'
        >>> ds = xr.open_zarr(cehq_data_path, consolidated=True)
        >>> donnees = Data(ds)
        >>> donnees.season = ['Yearly', 1, 365]
        >>> donnees.season = ['Spring', 60, 120]
        >>> ['Yearly', 'Spring']
        """
        return list(self.season.keys())

    def _get_season_values(self, season: str):
        """Function to get the values of a given season

        Parameters
        ----------
        season : str
            name of a previously defined season

        Returns
        -------
        list
            return a list of julian day of begining and end of season
        """
        return self._season[season]

    def get_bool_over_tolerence(self, tol: float, season=None):
        """
        Fonction to check if a season has enough values to be used.
        For each season True will be returned if there is less missing data
        than the specified tolerence.

        Parameters
        ----------
        tol : Float
        Tolerance in decimal form (0.15 for 15%)

        season : String
        Name of the season to be checked

        Returns
        -------
        da : xr.DataArray
        DataArray of boolean

        Examples
        --------
        >>> import xarray as xr
        >>> cehq_data_path =
        '/dbfs/mnt/devdlzxxkp01/datasets/xhydro/tests/cehq/zarr'
        >>> ds = xr.open_zarr(cehq_data_path, consolidated=True)
        >>> donnees = Data(ds)
        >>> donnees.custom_group_by(1, 365.max().where(
            grouped_ds.get_bool_over_tolerence(tolerence, season), drop=True)
        """
        ds = copy.copy(self)
        if season is None:
            # If no season is specified,
            # tolerence will be based on 365 values per year
            # TODO generalize for different time step
            tolerence = 365 * tol
            grouped_ds = ds.data.groupby("time.year").count()

        else:
            season_vals = ds._get_season_values(season)
            season_size = season_vals[1] - season_vals[0] + 1
            # TODO generalize for different time step
            grouped_ds = ds.custom_group_by(season_vals[0], season_vals[1]).count()
            tolerence = season_size * (1 - tol)

        return (grouped_ds.value > tolerence).load()


def get_maximum(self, tolerence: float = None, seasons=None):
    """
    Fonction to tiddy _get_max results.

    Parameters
    ----------
    tolerence : Float
      Tolerance in decimal form (0.15 for 15%)

    seasons : List
      List of season's name to be checked

    Returns
    -------
    df : pd.Dataframe
      Dataframe organised with id,	season,	year,	Maximums

    Examples
    --------
    >>> import xarray as xr
    >>> cehq_data_path =
    '/dbfs/mnt/devdlzxxkp01/datasets/xhydro/tests/cehq/zarr'
    >>> ds = xr.open_zarr(cehq_data_path, consolidated=True)
    >>> donnees = Data(ds)
    >>> donnees.get_maximum(tolerence=0.15, seasons=['Spring'])
    >>> catchment_list = ['023301']
    >>> sub_set = donnees.select_catchments(catchment_list = catchment_list)
    >>> sub_set.season = ['Spring', 60, 182]
    >>> sub_set.get_maximum(tolerence=0.15, seasons=['Spring'])
    >>> id	season	year	Maximums
      0	023301	Spring	1928	231.000000
      1	023301	Spring	1929	241.000000
      2	023301	Spring	1930	317.000000
      ...
    """
    return (
        self._get_max(tolerence=tolerence, seasons=seasons)
        .to_dataframe(name="Maximums")
        .reset_index()[["id", "season", "year", "start_date", "end_date", "Maximums"]]
        .dropna()
    )


def _get_max(self, tolerence=None, seasons=[]):
    """
    Fonction to get maximum value per season, according to a tolerence.

    Parameters
    ----------
    tolerence : Float
      Tolerance in decimal form (0.15 for 15%)

    seasons : List
      List of season's name to be checked

    Returns
    -------
    da : xr.DataArray
      DataArray of maximums

    Examples
    --------
    >>> import xarray as xr
    >>> cehq_data_path =
    '/dbfs/mnt/devdlzxxkp01/datasets/xhydro/tests/cehq/zarr'
    >>> ds = xr.open_zarr(cehq_data_path, consolidated=True)
    >>> donnees = Data(ds)
    >>> donnees.get_maximum(tolerence=0.15, seasons=['Spring'])
    >>> catchment_list = ['023301']
    >>> sub_set = donnees.select_catchments(catchment_list = catchment_list)
    >>> sub_set.season = ['Spring', 60, 182]
    >>> sub_set._get_max(tolerence=0.15, seasons=['Spring'])
        xarray.DataArray 'value' (season: 1, year: 52, id: 1)
    """
    grouped_ds = self.copy()

    def max_over_one_season(grouped_ds, tolerence, season):
        season_vals = grouped_ds._get_season_values(season)
        if isinstance(season_vals, xr.Dataset):
            years = np.unique(season_vals.year)
            bvs = np.unique(season_vals.id)
            max = np.empty((len(years), len(bvs)), dtype=object)
            beg = np.empty((len(years), len(bvs)), dtype=object)
            end = np.empty((len(years), len(bvs)), dtype=object)
            for y, year in enumerate(years):
                for b, bv in enumerate(bvs):
                    dd = season_vals.sel(year=year, id=bv).value.to_numpy().tolist()
                    beg[y, b] = pd.to_datetime(str(year) + str(dd[0]), format="%Y%j")
                    end[y, b] = pd.to_datetime(str(year) + str(dd[1]), format="%Y%j")
                    ds_year = grouped_ds.data.where(
                        grouped_ds.data.time.dt.year == year, drop=True
                    )
                    ds_year = ds_year.sel(id=bv)

                    # +1 to include end
                    ds_period = ds_year.sel(
                        time=np.isin(ds_year.time.dt.dayofyear, range(dd[0], dd[1] + 1))
                    )

                    d = ds_period.value.values
                    timestep = float(ds_year.time.dt.dayofyear.timestep.values.tolist())
                    nb_expected = (dd[1] + 1 - dd[0]) / timestep
                    # nb_expected is used to account for missing and nan
                    if np.count_nonzero(~np.isnan(d)) / nb_expected > (1 - tolerence):
                        max[y, b] = np.nanmax(d)  # .tolist()
                    else:
                        max[y, b] = np.nan

            max_ds = xr.Dataset()

            max_ds.coords["year"] = xr.DataArray(years, dims=("year",))
            max_ds.coords["id"] = xr.DataArray(bvs, dims=("id",))
            max_ds.coords["start_date"] = xr.DataArray(beg, dims=("year", "id"))
            max_ds.coords["end_date"] = xr.DataArray(end, dims=("year", "id"))
            max_ds["value"] = xr.DataArray(max.astype(float), dims=("year", "id"))
            # For each bv
            # For each year
            # check for tolerence
            # get max
            # create a DS
            return max_ds
        else:
            # TODO add year from grouped_ds.data.dt.year
            # and make full str start_date and end_date
            grouped_ds.data.coords["start_date"] = pd.to_datetime(
                str(season_vals[0]), format="%j"
            ).strftime("%m-%d")
            grouped_ds.data.coords["end_date"] = pd.to_datetime(
                str(season_vals[1]), format="%j"
            ).strftime("%m-%d")

            return (
                grouped_ds.custom_group_by(season_vals[0], season_vals[1])
                .max()
                .where(grouped_ds.get_bool_over_tolerence(tolerence, season), drop=True)
            )

    if seasons:
        # Creating a new dimension of season and
        # merging all Dataset from max_over_one_season
        return xr.concat(
            [
                max_over_one_season(grouped_ds, tolerence, season)
                .assign_coords(season=season)
                .expand_dims("season")
                for season in seasons
            ],
            dim="season",
        ).value

    else:
        # TODO Tolerence not used if no period is defined
        return (
            grouped_ds.data.groupby("time.year")
            .max()
            .value.assign_coords(season="Whole year")
            .expand_dims("season")
        )


def calculate_volume(self, dates: Union[list, xr.Dataset] = None, tolerence=0.15):
    ds = self.copy()

    def conversion_factor_to_hm3(timestep):
        # timestep in days
        # TODO check if last date is included
        return float(timestep) * 60 * 60 * 24

    if dates is None:
        # TODO use season dates
        pass
    elif isinstance(dates, list):
        # TODO bool over tolerence takes season, generalise
        with warnings.catch_warnings():  # Removes overlaping warning
            warnings.simplefilter("ignore")
            self.season = ["Volumes", dates[0], dates[1]]
        grouped_ds = (
            ds.custom_group_by(dates[0], dates[1])
            .sum()
            .where(ds.get_bool_over_tolerence(tolerence, "Volumes"), drop=True)
        )
        self.rm_season("Volumes")
        # Transform tp hm³
        # TODO add start and end and clear other attributes
        grouped_ds = (
            grouped_ds
            * xr.apply_ufunc(
                conversion_factor_to_hm3,
                grouped_ds["timestep"],
                input_core_dims=[[]],
                vectorize=True,
            )
            * (dates[1] - dates[0])
            / 1000000
        )

        df = grouped_ds.year.to_dataframe()
        df["beg"] = dates[0]
        df["end"] = dates[1]

        grouped_ds["start_date"] = pd.to_datetime(
            df["year"] * 1000 + df["beg"], format="%Y%j"
        )
        grouped_ds["end_date"] = pd.to_datetime(
            df["year"] * 1000 + df["end"], format="%Y%j"
        )

        grouped_ds["units"] = "hm³"

        return grouped_ds.drop_vars(
            [
                "_last_update_timestamp",
                "aggregation",
                "data_type",
                "data_type",
                "drainage_area",
                "latitude",
                "longitude",
                "name",
                "source",
                "timestep",
                "province",
                "regulated",
            ]
        ).rename_vars({"value": "volume"})
    elif isinstance(dates, xr.Dataset):
        # TODO Make sure DS has same dimensions than target
        vol = np.empty(
            (len(np.unique(ds.data.time.dt.year)), len(ds.data.id)), dtype=object
        )
        beg = np.empty(
            (len(np.unique(ds.data.time.dt.year)), len(ds.data.id)), dtype=object
        )
        end = np.empty(
            (len(np.unique(ds.data.time.dt.year)), len(ds.data.id)), dtype=object
        )
        for y, year in enumerate(np.unique(ds.data.time.dt.year)):
            for b, bv in enumerate(ds.data.id):
                dd = dates.sel(year=year, id=bv).value.to_numpy().tolist()
                beg[y, b] = pd.to_datetime(str(year) + str(dd[0]), format="%Y%j")
                end[y, b] = pd.to_datetime(str(year) + str(dd[1]), format="%Y%j")
                ds_year = ds.data.where(ds.data.time.dt.year == year, drop=True)
                ds_year = ds_year.sel(id=bv)
                # +1 pou inclure la fin,
                # TODO si une seule journe dans ds_period,  ¸a donne 0
                # TODO check for tolerence
                ds_period = ds_year.sel(
                    time=np.isin(ds_year.time.dt.dayofyear, range(dd[0], dd[1] + 1))
                )
                # delta en ns, à rapporter en s (1000000000)
                # puis le tout en hm³ (1000000)
                delta = ds_period.time[-1] - ds_period.time[0]
                delta = delta.to_numpy().tolist() / (1000000000 * 1000000)
                vol[y, b] = sum(ds_period.value.values).tolist() * delta

        vol_ds = xr.Dataset()

        vol_ds.coords["year"] = xr.DataArray(
            np.unique(ds.data.time.dt.year), dims=("year",)
        )
        vol_ds.coords["id"] = xr.DataArray(ds.data.id.to_numpy(), dims=("id",))
        vol_ds.coords["units"] = "hm³"
        vol_ds.coords["start_date"] = xr.DataArray(beg, dims=("year", "id"))
        vol_ds.coords["end_date"] = xr.DataArray(end, dims=("year", "id"))
        vol_ds["volume"] = xr.DataArray(vol, dims=("year", "id"))

        return vol_ds
