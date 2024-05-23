from xhydro.extreme_value_analysis.julia_import import Extremes, jl
from xhydro.extreme_value_analysis.structures.dataitem import *
from xhydro.extreme_value_analysis.structures.data import *
from xhydro.extreme_value_analysis import *
from xhydro.extreme_value_analysis.parameterestimation import *
import pandas as pd
from juliacall import convert as jl_convert


__all__ = ["gevfit", "gumbelfit", "gpfit"]

gevfit_signatures = [
    {'y', 'locationcov', 'logscalecov', 'shapecov'},
    {'y', 'initialvalues', 'locationcov', 'logscalecov', 'shapecov'},
    {'df','datacol','locationcovid', 'logscalecovid', 'shapecovid'},
    {'df','datacol', 'initialvalues', 'locationcovid', 'logscalecovid', 'shapecovid'},
    {'model', 'initialvalues'}
]


def gevfit(**kwargs):
    keys = set(kwargs.keys())

    if keys == gevfit_signatures[0]:
        y, locationcov, logscalecov, shapecov = kwargs['y'], kwargs['locationcov'], kwargs['logscalecov'], kwargs['shapecov']
        return _gevfit_1(y, locationcov=locationcov, logscalecov=logscalecov, shapecov=shapecov)
    elif keys == gevfit_signatures[1]:
        y, initialvalues, locationcov, logscalecov, shapecov = kwargs['y'], kwargs['initialvalues'], kwargs['locationcov'], kwargs['logscalecov'], kwargs['shapecov']
        return _gevfit_2(y, initialvalues, locationcov=locationcov, logscalecov=logscalecov, shapecov=shapecov)
    elif keys == gevfit_signatures[2]:
        df, datacol, locationcovid, logscalecovid, shapecovid = kwargs['df'], kwargs['datacol'], kwargs['locationcovid'], kwargs['logscalecovid'], kwargs['shapecovid']
        return _gevfit_3(df, datacol, locationcovid=locationcovid, logscalecovid=logscalecovid, shapecovid=shapecovid)
    elif keys == gevfit_signatures[3]:
        df, datacol, initialvalues, locationcovid, logscalecovid, shapecovid = kwargs['df'], kwargs['datacol'], kwargs['initialvalues'], kwargs['locationcovid'], kwargs['logscalecovid'], kwargs['shapecovid']
        return _gevfit_4(df, datacol, initialvalues, locationcovid=locationcovid, logscalecovid=logscalecovid, shapecovid=shapecovid)
    elif keys == gevfit_signatures[4]:
        model, initialvalues = kwargs['model'], kwargs['initialvalues']
        return _gevfit_5(model, initialvalues)
    else:
        raise FitError(
            "gevfit() has 5 accepted signatures:\n"
            "1. gevfit(y, locationcov, logscalecov, shapecov)\n"
            "2. gevfit(y, initialvalues, locationcov, logscalecov, shapecov\n"
            "3. gevfit(df, datacol, locationcovid, logscalecovid, shapecovid\n"
            "4. gevfit(df, datacol, initialvalues, locationcovid, logscalecovid, shapecovid\n"
            "5. gevfit(model, initialvalues\n"
        )



def _gevfit_1(y:list[float], locationcov: list[Variable] = [], logscalecov: list[Variable] = [], shapecov: list[Variable] = []):
    jl_y = py_list_float_to_julia_vector_real(y)
    jl_locationcov, jl_logscalecov, jl_shapecov = jl_variable_fit_parameters([locationcov, logscalecov, shapecov])
    return Extremes.gevfit(jl_y, locationcov=jl_locationcov, logscalecov=jl_logscalecov, shapecov=jl_shapecov )

def _gevfit_2(y:list[float], initialvalues:list[float], locationcov: list[Variable] = [], logscalecov: list[Variable] = [], shapecov: list[Variable] = []):
    jl_y, jl_initialvalues = py_list_float_to_julia_vector_real(y), py_list_float_to_julia_vector_real(initialvalues)
    jl_locationcov, jl_logscalecov, jl_shapecov = jl_variable_fit_parameters([locationcov, logscalecov, shapecov])
    return Extremes.gevfit(jl_y, jl_initialvalues, locationcov=jl_locationcov, logscalecov=jl_logscalecov, shapecov=jl_shapecov )

#TODO: test after pd_dataframe_to_jl_dataframe(df) is implemented
def _gevfit_3(df: pd.DataFrame, datacol: str, locationcovid: list[str] = [], logscalecovid: list[str] = [], shapecovid: list[str] = []):
    jl_df = pd_dataframe_to_jl_dataframe(df) 
    jl_datacol = py_str_to_jl_symbol(datacol)
    jl_locationcovid, jl_logscalecovid, jl_shapecovid = jl_symbol_fit_parameters([locationcovid, logscalecovid, shapecovid])
    return Extremes.gevfit(jl_df, jl_datacol, locationcovid = jl_locationcovid, logscalecovid = jl_logscalecovid, shapecovid = jl_shapecovid)

#TODO: test after pd_dataframe_to_jl_dataframe(df) is implemented
def _gevfit_4(df: pd.DataFrame, datacol: str, initialvalues: list[float], locationcovid: list[str] = [], logscalecovid: list[str] = [], shapecovid: list[str] = []):
    jl_df = pd_dataframe_to_jl_dataframe(df) 
    jl_datacol = py_str_to_jl_symbol(datacol)
    jl_initialvalues = py_list_float_to_julia_vector_real(initialvalues)
    jl_locationcovid, jl_logscalecovid, jl_shapecovid = jl_symbol_fit_parameters([locationcovid, logscalecovid, shapecovid])
    return Extremes.gevfit(jl_df, jl_datacol, jl_initialvalues, locationcovid = jl_locationcovid, logscalecovid = jl_logscalecovid, shapecovid = jl_shapecovid)

#TODO: implement when julia.Extremes.BlockMaxima -> python equivalent conversion is implemented
def _gevfit_5(model, initialvalues: list[float]):
    pass


gumbel_signatures = [
    {'y', 'locationcov', 'logscalecov'},
    {'y', 'initialvalues', 'locationcov', 'logscalecov'},
    {'df','datacol','locationcovid', 'logscalecovid'},
    {'df','datacol', 'initialvalues', 'locationcovid', 'logscalecovid'},
    {'model', 'initialvalues'}
]

def gumbelfit(**kwargs):
    keys = set(kwargs.keys())

    if keys == gumbel_signatures[0]:
        y, locationcov, logscalecov = kwargs['y'], kwargs['locationcov'], kwargs['logscalecov']
        return _gumbelfit_1(y, locationcov=locationcov, logscalecov=logscalecov)
    elif keys == gumbel_signatures[1]:
        y, initialvalues, locationcov, logscalecov = kwargs['y'], kwargs['initialvalues'], kwargs['locationcov'], kwargs['logscalecov']
        return _gumbelfit_2(y, initialvalues, locationcov=locationcov, logscalecov=logscalecov)
    elif keys == gumbel_signatures[2]:
        df, datacol, locationcovid, logscalecovid = kwargs['df'], kwargs['datacol'], kwargs['locationcovid'], kwargs['logscalecovid']
        return _gumbelfit_3(df, datacol, locationcovid=locationcovid, logscalecovid=logscalecovid)
    elif keys == gumbel_signatures[3]:
        df, datacol, initialvalues, locationcovid, logscalecovid = kwargs['df'], kwargs['datacol'], kwargs['initialvalues'], kwargs['locationcovid'], kwargs['logscalecovid']
        return _gumbelfit_4(df, datacol, initialvalues, locationcovid=locationcovid, logscalecovid=logscalecovid)
    elif keys == gumbel_signatures[4]:
        model, initialvalues = kwargs['model'], kwargs['initialvalues']
        return _gumbelfit_5(model, initialvalues)
    else:
        raise FitError(
            "gumbelfit() has 5 accepted signatures:\n"
            "1. gumbelfit(y, locationcov, logscalecov)\n"
            "2. gumbelfit(y, initialalues, locationcov, logscalecov\n"
            "3. gumbelfit(df, datacol, locationcovid, logscalecovid\n"
            "4. gumbelfit(df, datacol, initialvalues, locationcovid, logscalecovid\n"
            "5. gumbelfit(model, initialvalues\n"
        )

def _gumbelfit_1(y:list[float], locationcov: list[Variable] = [], logscalecov: list[Variable] = []):
    jl_y = py_list_float_to_julia_vector_real(y)
    jl_locationcov, jl_logscalecov= jl_variable_fit_parameters([locationcov, logscalecov])
    return Extremes.gumbelfit(jl_y, locationcov=jl_locationcov, logscalecov=jl_logscalecov)

def _gumbelfit_2(y:list[float], initialvalues: list[float], locationcov: list[Variable] = [], logscalecov: list[Variable] = []):
    jl_y, jl_initialvalues = py_list_float_to_julia_vector_real(y), py_list_float_to_julia_vector_real(initialvalues)
    jl_locationcov, jl_logscalecov= jl_variable_fit_parameters([locationcov, logscalecov])
    return Extremes.gumbelfit(jl_y, jl_initialvalues, locationcov=jl_locationcov, logscalecov=jl_logscalecov)

def _gumbelfit_3(df:pd.DataFrame, datacol: str, locationcov: list[Variable] = [], logscalecov: list[Variable] = []):
    jl_df = pd_dataframe_to_jl_dataframe(df) 
    jl_datacol = py_str_to_jl_symbol(datacol)
    jl_locationcov, jl_logscalecov= jl_symbol_fit_parameters([locationcov, logscalecov])
    return Extremes.gumbelfit(jl_df, jl_datacol, locationcov=jl_locationcov, logscalecov=jl_logscalecov)

def _gumbelfit_4(df:pd.DataFrame, datacol: str, initialvalues: list[float], locationcov: list[Variable] = [], logscalecov: list[Variable] = []):
    jl_df = pd_dataframe_to_jl_dataframe(df) 
    jl_datacol = py_str_to_jl_symbol(datacol)
    jl_initialvalues = py_list_float_to_julia_vector_real(initialvalues)
    jl_locationcov, jl_logscalecov= jl_symbol_fit_parameters([locationcov, logscalecov])
    return Extremes.gumbelfit(jl_df, jl_datacol, jl_initialvalues, locationcov=jl_locationcov, logscalecov=jl_logscalecov)

#TODO: implement when julia.Extremes.BlockMaxima -> python equivalent conversion is implemented
def _gumbelfit_5(model, initialvalues:list[float]):
    pass





gp_signatures = [
    {'y', 'logscalecov', 'shapecov'},
    {'y', 'initialvalues', 'logscalecov', 'shapecov'},
    {'df','datacol','logscalecovid', 'shapecovid'},
    {'df','datacol', 'initialvalues', 'logscalecovid', 'shapecovid'},
    {'model', 'initialvalues'}
]

def gpfit(**kwargs):
    keys = set(kwargs.keys())

    if keys == gp_signatures[0]:
        y, logscalecov, shapecov = kwargs['y'], kwargs['logscalecov'], kwargs['shapecov']
        return _gpfit_1(y, logscalecov=logscalecov, shapecov=shapecov)
    elif keys == gp_signatures[1]:
        y, initialvalues, logscalecov, shapecov = kwargs['y'], kwargs['initialvalues'], kwargs['logscalecov'], kwargs['shapecov']
        return _gpfit_2(y, initialvalues, logscalecov=logscalecov, shapecov=shapecov)
    elif keys == gp_signatures[2]:
        df, datacol, logscalecovid, shapecovid = kwargs['df'], kwargs['datacol'], kwargs['logscalecovid'], kwargs['shapecovid']
        return _gpfit_3(df, datacol, logscalecovid=logscalecovid, shapecovid=shapecovid)
    elif keys == gp_signatures[3]:
        df, datacol, initialvalues, logscalecovid, shapecovid = kwargs['df'], kwargs['datacol'], kwargs['initialvalues'], kwargs['logscalecovid'], kwargs['shapecovid']
        return _gpfit_4(df, datacol, initialvalues, logscalecovid=logscalecovid, shapecovid=shapecovid)
    elif keys == gp_signatures[4]:
        model, initialvalues = kwargs['model'], kwargs['initialvalues']
        return _gpfit_5(model, initialvalues)
    else:
        raise FitError(
            "gpfit() has 5 accepted signatures:\n"
            "1. gpfit(y, logscalecov, shapecov)\n"
            "2. gpfit(y, initialalues, logscalecov, shapecov\n"
            "3. gpfit(df, datacol, logscalecovid, shapecovid\n"
            "4. gpfit(df, datacol, initialvalues, logscalecovid, shapecovid\n"
            "5. gpfit(model, initialvalues\n"
        )

def _gpfit_1(y:list[float], logscalecov: list[Variable] = [], shapecov: list[Variable] = []):
    jl_y = py_list_float_to_julia_vector_real(y)
    jl_logscalecov, jl_shapecov = jl_variable_fit_parameters([logscalecov, shapecov])
    return Extremes.gevfit(jl_y, logscalecov=jl_logscalecov, shapecov=jl_shapecov )

def _gpfit_2(y:list[float], initialvalues:list[float], logscalecov: list[Variable] = [], shapecov: list[Variable] = []):
    jl_y, jl_initialvalues = py_list_float_to_julia_vector_real(y), py_list_float_to_julia_vector_real(initialvalues)
    jl_logscalecov, jl_shapecov = jl_variable_fit_parameters([logscalecov, shapecov])
    return Extremes.gevfit(jl_y, jl_initialvalues, logscalecov=jl_logscalecov, shapecov=jl_shapecov )

#TODO: test after pd_dataframe_to_jl_dataframe(df) is implemented
def _gpfit_3(df: pd.DataFrame, datacol: str, logscalecovid: list[str] = [], shapecovid: list[str] = []):
    jl_df = pd_dataframe_to_jl_dataframe(df) 
    jl_datacol = py_str_to_jl_symbol(datacol)
    jl_logscalecovid, jl_shapecovid = jl_symbol_fit_parameters([logscalecovid, shapecovid])
    return Extremes.gevfit(jl_df, jl_datacol, logscalecovid = jl_logscalecovid, shapecovid = jl_shapecovid)

#TODO: test after pd_dataframe_to_jl_dataframe(df) is implemented
def _gpfit_4(df: pd.DataFrame, datacol: str, initialvalues: list[float], logscalecovid: list[str] = [], shapecovid: list[str] = []):
    jl_df = pd_dataframe_to_jl_dataframe(df) 
    jl_datacol = py_str_to_jl_symbol(datacol)
    jl_initialvalues = py_list_float_to_julia_vector_real(initialvalues)
    jl_logscalecovid, jl_shapecovid = jl_symbol_fit_parameters([logscalecovid, shapecovid])
    return Extremes.gevfit(jl_df, jl_datacol, jl_initialvalues, logscalecovid = jl_logscalecovid, shapecovid = jl_shapecovid)

#TODO: implement when julia.Extremes.BlockMaxima -> python equivalent conversion is implemented
def _gpfit_5(model, initialvalues: list[float]):
    pass


def jl_symbol_fit_parameters(params: list[list[str]]) -> tuple:
    # python list lists of julia Symbols
    symbols = [[py_str_to_jl_symbol(symbol) for symbol in params[i]] for i in range(len(params))]

    # python tuple of julia vectors of julia Symbols
    jl_params = tuple((jl_convert(jl.Vector[jl.Symbol], symbols[i])) for i in range(len(symbols)))
    return jl_params

def jl_variable_fit_parameters(params: list[list[Variable]]) -> tuple:
    # python list of lists of julia.Extremes Variables
    variables = [[variable.py_variable_to_jl_variable() for variable in params[i]] for i in range(len(params))]

    # python tuple of julia vectors of julia.Extremes Variables
    jl_params = tuple(jl_convert(jl.Vector[jl.Extremes.Variable], variables[i]) for i in range(len(variables)))
    return jl_params


