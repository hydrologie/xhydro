import xarray as xr
import numpy as np
import warnings
import copy

class Data:

    def __init__(self, 
                 ds: xr.Dataset):
        """init function takes a dataset as input and initialised an empty season dictionary and a list of catchments from dimension id 

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
    
    def select_catchments(self,
                        catchment_list: list):
        """
        select specified catchements from attribute data. Also supports the use of a wildcard (*).
        
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
        >>> cehq_data_path = '/dbfs/mnt/devdlzxxkp01/datasets/xhydro/tests/cehq/zarr'
        >>> ds = xr.open_zarr(cehq_data_path, consolidated=True)
        >>> donnees = Data(ds)
        >>> filtered_ds = donnees.select_catchments(catchment_list = ['051001','051005'])
        >>> filtered_ds = donnees.select_catchments(catchment_list = ['05*','0234*', '023301'])
        """

        # Create a copy of the object
        obj = self.copy()

    

        # sub function to select complete list based on wilcards 
        def multi_filter(names,
                        patterns: list):
            return [name for name in names for pattern in patterns if fnmatch.fnmatch(name, pattern) ]

        # Getting the full list
        catchment_list = multi_filter(obj.catchments, catchment_list)

        # Setting the list as a class attribute
        obj._catchments = catchment_list

        # Filtering over the list
        obj.data = obj.data.sel(id=self.data.id.isin(catchment_list))
        return obj
    
    def custom_group_by(self, 
                      beg: int, 
                      end: int):
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
        >>> cehq_data_path = '/dbfs/mnt/devdlzxxkp01/datasets/xhydro/tests/cehq/zarr'
        >>> ds = xr.open_zarr(cehq_data_path, consolidated=True)
        >>> donnees = Data(ds)
        >>> grouped_ds = donnees.custom_group_by(1, 90)
    
        """
    
        if beg > end:
        # TODO when beg `end, it means it has to overlap years, for example, from octobre to march
            pass
        else:
        # +1 to include the end
            return self.data.sel(time=np.isin(self.data.time.dt.dayofyear, range(beg, end + 1))).groupby('time.year')
        

    @property
    def season(self):
        return self._season
  
    @season.setter
    def season(self, 
             liste: list):
        """
        The setter for the season property Issues a Warining if a new season is overlapping with another one. 
        Alos issues a warning if the season name was already used, mand then overwrites it.
        
        Parameters
        ----------
        liste : List
        List of Name, begining in Julian day, end in Julian day

        Returns
        -------
        ds : xarray.DataSet
        appends Dict self._season with name as key, and begining and end as values
        
        Examples
        --------
        >>> import xarray as xr
        >>> cehq_data_path = '/dbfs/mnt/devdlzxxkp01/datasets/xhydro/tests/cehq/zarr'
        >>> ds = xr.open_zarr(cehq_data_path, consolidated=True)
        >>> donnees = Data(ds)
        >>> donnees.season = ['Yearly', 1, 365]
        """

        #TODO a dictionary would probalby be more suited
        name = liste[0]
        beg = liste[1]
        end = liste[2]
        for season, dates in self.season.items():
            if not isinstance(dates, xr.Dataset):
                # We dont check for overlapping if season is a dataset
                if name == season:
                    warnings.warn('Warning, ' + name + ' was already defined and has been overwritten')
                elif dates[0] <= beg and dates[1] >= beg or dates[0] <= end and dates[1] >= end:
                    warnings.warn('Warning, ' + name + ' overlapping with ' + season)
            
        self._season[name] = [beg, end]


    def rm_season(self, 
                    name: str):
        """
        Fonction to remove a season. Isues a Warining if the name is not a valid season. 
            
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
        >>> cehq_data_path = '/dbfs/mnt/devdlzxxkp01/datasets/xhydro/tests/cehq/zarr'
        >>> ds = xr.open_zarr(cehq_data_path, consolidated=True)
        >>> donnees = Data(ds)
        >>> donnees.season = ['Yearly', 1, 365]
        >>> donnees.rm_season('Yearly')
        """

        try:
            del self._season[name]
        except:
            print('No season named ' + name)
        
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
        >>> cehq_data_path = '/dbfs/mnt/devdlzxxkp01/datasets/xhydro/tests/cehq/zarr'
        >>> ds = xr.open_zarr(cehq_data_path, consolidated=True)
        >>> donnees = Data(ds)
        >>> donnees.season = ['Yearly', 1, 365]
        >>> donnees.season = ['Spring', 60, 120]
        >>> ['Yearly', 'Spring']
        """
        return list(self.season.keys())
    
    def _get_season_values(self, 
                            season: str):
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