import datetime as dt
import tempfile

import xhydro.optimal_interpolation.optimal_interpolation_precip as oi
from xhydro.testing.helpers import deveraux


def test_precip_oi():
    start_time = dt.datetime(1970, 1, 1)
    end_time = dt.datetime(1970, 12, 31)

    filename_stations = deveraux().fetch("precip_oi/stations_flags_clean_subset.nc")
    filename_gridded = deveraux().fetch("precip_oi/ERA5_land_3_variables_raw_subset.nc")

    filename_output = tempfile.mkdtemp() + "/ERA5Land_with_OI_flags_validation_subset.nc"

    ds_final, train_idx, valid_idx = oi.main(
        start_time=start_time,
        end_time=end_time,
        filename_stations=filename_stations,
        filename_gridded=filename_gridded,
        filename_output=filename_output,
        grid_resolution=0.1,
        var_name_gridded="tp",
        var_name_stations="precip",
        dims_gridded=("longitude", "latitude", "time"),
        dims_stations="station",
        coords_gridded=("time", "latitude", "longitude"),
        station_req_vars=("latitude", "longitude", "altitude"),
        percentiles=[50],
        p1_bnds=[0.95, 1.0],
        hmax_mult_range_bnds=[0.05, 3],
        var_bg_ratio=0.15,
        variogram_bins=10,
        hmax_divider=2.0,
        ecf_form=1,
        do_cross_validation=False,
        frac_validation=0.0,
    )

    assert ds_final["percentile"] == 50
    assert valid_idx.shape[0] == 0
