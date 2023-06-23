import xarray as xr
import numpy as np
import warnings
import copy
from typing import Union
import pandas as pd
import fnmatch
import scipy.stats
import xhydro as xh

from statsmodels.tools import eval_measures
from xclim.indices.stats import fit, parametric_quantile


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
        catchment_list = multi_filter(obj._catchments, catchment_list)

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

        return (grouped_ds[list(grouped_ds.keys())[0]] > tolerence).load()

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
            .to_dataframe()
            .reset_index()[
                [
                    "id",
                    "season",
                    "year",
                    "start_date",
                    "end_date",
                    list(self.data.keys())[0],
                ]
            ]
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
                        beg[y, b] = pd.to_datetime(
                            str(year) + str(dd[0]), format="%Y%j"
                        )
                        end[y, b] = pd.to_datetime(
                            str(year) + str(dd[1]), format="%Y%j"
                        )
                        ds_year = grouped_ds.data.where(
                            grouped_ds.data.time.dt.year == year, drop=True
                        )
                        ds_year = ds_year.sel(id=bv)

                        # +1 to include end
                        ds_period = ds_year.sel(
                            time=np.isin(
                                ds_year.time.dt.dayofyear, range(dd[0], dd[1] + 1)
                            )
                        )

                        d = ds_period[list(ds_period.keys())[0]].values
                        timestep = xh.get_timestep(ds_year)
                        # timestep = float(
                        # ds_year.time.dt.dayofyear.timestep.values.tolist()
                        # )
                        nb_expected = (dd[1] + 1 - dd[0]) / timestep
                        # nb_expected is used to account for missing and nan
                        if np.count_nonzero(~np.isnan(d)) / nb_expected > (
                            1 - tolerence
                        ):
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
                    .where(
                        grouped_ds.get_bool_over_tolerence(tolerence, season), drop=True
                    )
                )

        if seasons:
            # Creating a new dimension of season and
            # merging all Dataset from max_over_one_season
            ds = xr.concat(
                [
                    max_over_one_season(grouped_ds, tolerence, season)
                    .assign_coords(season=season)
                    .expand_dims("season")
                    for season in seasons
                ],
                dim="season",
            )
            return ds[list(ds.keys())[0]]

        else:
            # TODO Tolerence not used if no period is defined
            return (
                grouped_ds.data.groupby("time.year")
                .max()
                .assign_coords(season="Whole year")
                .expand_dims("season")
            )

    def calculate_volume(self, dates: Union[list, xr.Dataset] = None, tolerence=0.15):
        ds = self.copy()

        def conversion_factor_to_hm3(timestep):
            # flow is in m³/s and we want m³, so we multiply by seconds
            # TODO check if last date is included
            return pd.to_timedelta(1, unit=timestep).total_seconds()

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
                / 1000000  # from m³ to hm³
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
                    "drainage_area",
                    "latitude",
                    "longitude",
                    "name",
                    "source",
                    "timestep",
                    "province",
                    "regulated",
                ]
            ).rename_vars({list(grouped_ds.keys())[0]: "volume"})
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
                    try:
                        dd = dates.sel(year=year, id=bv).value.to_numpy().tolist()
                    except KeyError:
                        # KeyError can occur if ds is incomplete
                        pass
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
                    vol[y, b] = (
                        sum(
                            ds_period.squeeze()[list(ds_period.keys())[0]].values
                        ).tolist()
                        * delta
                    )

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


class Local:
    # TODO list(ds.keys())[0] used multiples time, genewralise ofr all var, not just [0] and do it a better way, ie, in the init
    def __init__(
        self,
        data_ds,
        return_period,
        dates_vol=None,
        dist_list=[
            "expon",
            "gamma",
            "genextreme",
            "genpareto",
            "gumbel_r",
            "pearson3",
            "weibull_min",
        ],
        tolerence=0.15,
        seasons=None,
        min_year=15,
        vars_of_interest=["max", "vol"],
        calculated=False,
    ):
        # TODO if type of data is object instead of float, it will crash, better to convert or to raise a warning  ?
        try:
            # if data is provided
            self.data = data_ds.astype(float)
        except AttributeError:
            # if not
            self.data = data_ds.data.astype(float)
        self.data = data_ds
        self.return_period = return_period
        self.dist_list = dist_list
        self.dates_vol = dates_vol
        self.tolerence = tolerence
        self.seasons = seasons
        self.min_year = min_year
        self.analyse_max = None
        self.analyse_vol = None

        if "max" in vars_of_interest:
            self.analyse_max = self._freq_analys(calculated, var_of_interest="max")
        if "vol" in vars_of_interest:
            self.analyse_vol = self._freq_analys(calculated, var_of_interest="vol")

    def _freq_analys(self, calculated: bool, var_of_interest: str):
        """
        This function is executed upon initialization of calss Local. It performs multiple frequency analysis over the data provided.

        Parameters
        ----------
        self.data  : xhydro.Data
          a dataset containing hydrological data

        self.return_period : list
          list of return periods as float

        self.dist_list : list
          list of distribution supported by scypy.stat

        self.tolerence : float
          percentage of missing value tolerence in decimal form (0.15 for 15%), if above within the season, the maximum for that year will be skipped

        self.seasons : list
          list of seasons names, begining and end of said seasons must have been set previously in the object xhydro.Data

        self.min_year : int
          Minimum number of year. If a station has less year than the minimum, for any given season, the station will be skipped

        Returns
        -------
        self.analyse : xr.Dataset
          A Dataset with dimensions id, season, scipy_dist and return_period indexes with variables Criterions and Quantiles

        Examples
        --------
        >>> TODO Not sure how to set example here


        """

        def get_criterions(data, params, dist):
            data = data[~np.isnan(data)]
            # params = params[~np.isnan(params)]

            LLH = dist.logpdf(data, *params).sum()  # log-likelihood

            aic = eval_measures.aic(llf=LLH, nobs=len(data), df_modelwc=len(params))
            bic = eval_measures.bic(llf=LLH, nobs=len(data), df_modelwc=len(params))
            try:
                aicc = eval_measures.aicc(
                    llf=LLH, nobs=len(data), df_modelwc=len(params)
                )
            except:
                aicc = np.nan

            # logLik = np.sum( stats.gamma.logpdf(data, fitted_params[0], loc=fitted_params[1], scale=fitted_params[2]) )
            # k = len(fitted_params)
            # aic = 2*k - 2*(logLik)
            return {"aic": aic, "bic": bic, "aicc": aicc}

        def fit_one_distrib(ds_max, dist):
            return (
                fit(ds_max.chunk(dict(time=-1)), dist=dist)
                .assign_coords(scipy_dist=dist)
                .expand_dims("scipy_dist")
            )  # .rename('value')

        if calculated:
            ds_calc = self.data.rename({"year": "time"}).load()
        else:
            if var_of_interest == "max":
                ds_calc = (
                    self.data._get_max(self.tolerence, self.seasons)
                    .rename({"year": "time"})
                    .load()
                )
            elif var_of_interest == "vol":
                ds_calc = (
                    self.data.calculate_volume(
                        tolerence=self.tolerence, dates=self.dates_vol
                    )
                    .rename({"year": "time", "volume": "value"})
                    .astype(float)
                    .load()
                )
                ds_calc = ds_calc.value
        ds_calc = ds_calc.dropna(dim="id", thresh=self.min_year)

        quantiles = []
        criterions = []
        parameters = []
        for dist in self.dist_list:
            # FIXME .load() causes issues, but it was added to fix something

            params = fit_one_distrib(ds_calc, dist).load()
            parameters.append(params)
            # quantiles.append(xr.merge([parametric_quantile(params, q=1 - 1.0 / T).rename('value') for T in self.return_period]))
            quantiles.append(
                xr.merge(
                    [
                        parametric_quantile(params, q=1 - 1.0 / T)
                        for T in self.return_period
                    ]
                )
            )
            dist_obj = getattr(scipy.stats, dist)
            # criterions.append(xr.apply_ufunc(get_criterions, ds_calc, params, dist_obj, input_core_dims=[['time'], ['dparams'], []], vectorize=True).to_dataset(name='Criterions'))

            crit = xr.apply_ufunc(
                get_criterions,
                ds_calc,
                params,
                dist_obj,
                input_core_dims=[["time"], ["dparams"], []],
                vectorize=True,
            )
            # If crit is a DataArray, the variable we name it value, if it's a DataSet, it will already have variables names
            if isinstance(crit, xr.DataArray):
                crit.name = "value"
            criterions.append(crit)

        def append_var_names(ds, str):
            try:
                var_list = list(ds.keys())
            except:
                new_name = ds.name + str
                return ds.rename(new_name)
            dict_names = dict(zip(var_list, [s + str for s in var_list]))
            return ds.rename(dict_names)

        # ds_paramters = xr.Dataset()
        ds_paramters = xr.concat(
            parameters, dim="scipy_dist", combine_attrs="drop_conflicts"
        )
        ds_paramters = append_var_names(ds_paramters, "_parameters")

        ds_quantiles = xr.merge(quantiles)
        ds_quantiles = append_var_names(ds_quantiles, "_quantiles")

        ds_criterions = xr.merge(criterions)
        ds_criterions = append_var_names(ds_criterions, "_criterions")

        ds_quantiles = ds_quantiles.rename({"quantile": "return_period"})
        ds_quantiles["return_period"] = 1.0 / (1 - ds_quantiles.return_period)

        return xr.merge([ds_criterions, ds_quantiles, ds_calc, ds_paramters])

    def view_criterions(self, var_of_interest):
        """
        Fonction to get Criterions results from a xhydro.Local object. Output is rounded for easiser visualisation.

        Returns
        -------
        df : pd.Dataframe
          Dataframe organised with id,	season,	year,	scipy_dist

        Examples
        --------
        >>> import xarray as xr
        >>> cehq_data_path = '/dbfs/mnt/devdlzxxkp01/datasets/xhydro/tests/cehq/zarr'
        >>> ds = xr.open_zarr(cehq_data_path, consolidated=True)
        >>> donnees = Data(ds)
        >>> donnees.get_maximum(tolerence=0.15, seasons=['Spring'])
        >>> catchment_list = ['023301']
        >>> sub_set = donnees.select_catchments(catchment_list = catchment_list)
        >>> sub_set.season = ['Spring', 60, 182]
        >>> return_period = np.array([1.01, 2, 2.33, 5, 10, 20, 50, 100, 200, 500, 1000, 10000])
        >>> dist_list = ['expon', 'gamma', 'genextreme', 'genpareto', 'gumbel_r', 'pearson3', 'weibull_min']
        >>> fa = xh.Local(data_ds = sub_set, return_period = return_period, dist_list = dist_list, tolerence = 0.15, seasons = ['Automne', 'Printemps'], min_year = 15)
        >>> fa.view_quantiles()
        >>> id	season	level_2	return_period	Quantiles
            0	023301	Spring	expon	1.01	22.157376
            1	023301	Spring	expon	2.00	87.891419
            2	023301	Spring	expon	2.33	102.585536
            3	023301	Spring	expon	5.00	176.052678
            4	023301	Spring	expon	10.00	242.744095
            ...
        """
        # dataarray to_dataframe uses first diemnsion as nameless index, so depending on the position in dim_order, dimension gets names level_x
        var_list = [s for s in self.analyse_max.keys() if "criterions" in s]

        if var_of_interest == "vol":
            return (
                self.analyse_vol[var_list]
                .to_dataframe(dim_order=["id", "scipy_dist"])[var_list]
                .reset_index()
                .rename(columns={"level_1": "scipy_dist"})
                .round()
            )
        elif var_of_interest == "max":
            return (
                self.analyse_max[var_list]
                .to_dataframe(dim_order=["id", "season", "scipy_dist"])[var_list]
                .reset_index()
                .rename(columns={"level_2": "scipy_dist"})
                .round()
            )
        else:
            return print('use "vol" for volumes or "max" for maximums ')

    def view_quantiles(self, var_of_interest):
        """
        Fonction to get Quantiles results from a xhydro.Local object.

        Returns
        -------
        df : pd.Dataframe
          Dataframe organised with id,	season,	year,	scipy_dist, return_period

        Examples
        --------
        >>> import xarray as xr
        >>> cehq_data_path = '/dbfs/mnt/devdlzxxkp01/datasets/xhydro/tests/cehq/zarr'
        >>> ds = xr.open_zarr(cehq_data_path, consolidated=True)
        >>> donnees = Data(ds)
        >>> donnees.get_maximum(tolerence=0.15, seasons=['Spring'])
        >>> catchment_list = ['023301']
        >>> sub_set = donnees.select_catchments(catchment_list = catchment_list)
        >>> sub_set.season = ['Spring', 60, 182]
        >>> return_period = np.array([1.01, 2, 2.33, 5, 10, 20, 50, 100, 200, 500, 1000, 10000])
        >>> dist_list = ['expon', 'gamma', 'genextreme', 'genpareto', 'gumbel_r', 'pearson3', 'weibull_min']
        >>> fa = xh.Local(data_ds = sub_set, return_period = return_period, dist_list = dist_list, tolerence = 0.15, seasons = ['Automne', 'Printemps'], min_year = 15)
        >>> fa.view_criterions()
        >>> id	season	level_2	Criterions
        >>> 0	023301	Spring	expon	{'aic': 582.9252842821857, 'bic': 586.82777171...
        >>> 1	023301	Spring	gamma	{'aic': 1739.77441499742, 'bic': 1745.62814615...
            ...
        """

        var_list = [s for s in self.analyse_max.keys() if "quantiles" in s]
        if var_of_interest == "vol":
            return (
                self.analyse_vol[var_list]
                .to_dataframe(dim_order=["id", "scipy_dist", "return_period"])[var_list]
                .reset_index()
                .rename(columns={"level_1": "scipy_dist"})
                .round()
            )
        elif var_of_interest == "max":
            return (
                self.analyse_max[var_list]
                .to_dataframe(
                    dim_order=["id", "season", "scipy_dist", "return_period"]
                )[var_list]
                .reset_index()
                .rename(columns={"level_2": "scipy_dist"})
                .round()
            )
        else:
            return print('use "vol" for volumes or "max" for maximums ')

    def view_values(self, var_of_interest):
        """
        Fonction to get values results from a xhydro.Local object.

        Returns
        -------
        df : pd.Dataframe
          Dataframe organised with id,	season,	year,	scipy_dist, return_period

        Examples
        --------
        >>> import xarray as xr
        >>> cehq_data_path = '/dbfs/mnt/devdlzxxkp01/datasets/xhydro/tests/cehq/zarr'
        >>> ds = xr.open_zarr(cehq_data_path, consolidated=True)
        >>> donnees = Data(ds)
        >>> donnees.get_maximum(tolerence=0.15, seasons=['Spring'])
        >>> catchment_list = ['023301']
        >>> sub_set = donnees.select_catchments(catchment_list = catchment_list)
        >>> sub_set.season = ['Spring', 60, 182]
        >>> return_period = np.array([1.01, 2, 2.33, 5, 10, 20, 50, 100, 200, 500, 1000, 10000])
        >>> dist_list = ['expon', 'gamma', 'genextreme', 'genpareto', 'gumbel_r', 'pearson3', 'weibull_min']
        >>> fa = xh.Local(data_ds = sub_set, return_period = return_period, dist_list = dist_list, tolerence = 0.15, seasons = ['Automne', 'Printemps'], min_year = 15)
        >>> fa.view_criterions()
        >>> id	season	level_2	Criterions
        >>> 0	023301	Spring	expon	{'aic': 582.9252842821857, 'bic': 586.82777171...
        >>> 1	023301	Spring	gamma	{'aic': 1739.77441499742, 'bic': 1745.62814615...
            ...
        """
        # TODO  Output as dict is ugly

        if var_of_interest == "vol":
            return self.analyse_vol.value.to_dataframe().dropna().reset_index()
        elif var_of_interest == "max":
            return (
                self.analyse_max.value.to_dataframe()
                .reset_index()[
                    ["id", "season", "time", "start_date", "end_date", "Maximums"]
                ]
                .dropna()
            )
        else:
            return print('use "vol" for volumes or "max" for maximums ')
