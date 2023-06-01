import xarray as xr
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