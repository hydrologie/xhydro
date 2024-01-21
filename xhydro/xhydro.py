"""Main module."""
from optimal_interpolation import compare_result
from optimal_interpolation import cross_validation
from optimal_interpolation import constants


start_date = '1961-01-01'
end_date = '2019-01-01'
files = [
    constants.DATA_PATH + "Table_Info_Station_Hydro_2020.csv",
    constants.DATA_PATH + "Table_Correspondance_Station_Troncon.csv",
    constants.DATA_PATH + "stations_retenues_validation_croisee.csv",
    constants.DATA_PATH + 'A20_HYDOBS_QCMERI_XXX_DEBITJ_HIS_XXX_XXX_XXX_XXX_XXX_XXX_XXX_XXXXXX_XXX_HC_13102020.nc',
    constants.DATA_PATH + 'A20_HYDREP_QCMERI_XXX_DEBITJ_HIS_XXX_XXX_XXX_XXX_XXX_XXX_HYD_MG24HS_GCQ_SC_18092020.nc'
]

cpu_parrallel = True

cross_validation.execute(start_date, end_date, files, cpu_parrallel)

