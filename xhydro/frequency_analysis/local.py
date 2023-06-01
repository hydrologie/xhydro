import xarray as xr
import numpy as np
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
        # TODO chevauchement d'années ex : année hydrologique, hiver de décembre à mars, etc
            pass
        else:
        # +1 to include the end
            return self.data.sel(time=np.isin(self.data.time.dt.dayofyear, range(beg, end + 1))).groupby('time.year')