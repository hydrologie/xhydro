"""Hidden utilities for HYDROTEL and RavenPy models."""

import datetime as dt
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
import xscen as xs
import yaml
from xclim.core.units import convert_units_to


with Path(Path(__file__).parent / "variables.yml").open() as f:
    VARIABLES = yaml.safe_load(f)


def standardize_output(ds, spatial_info: pd.DataFrame | None = None, alt_names: dict[str, str] | None = None) -> xr.Dataset:  # noqa: C901
    """
    Standardize the output dataset by renaming dimensions and variables, adding relevant coordinates, and correcting attributes.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to standardize.
    spatial_info : pd.DataFrame | None, optional
        A dataframe containing the spatial information of the model (RavenpyModel.hru["hru"] or Hydrotel.rhhu).
    alt_names : dict[str, str] | None, optional
        A dictionary mapping original variable names to their standardized names.

    Returns
    -------
    xr.Dataset
        The standardized dataset.
    """
    if alt_names is not None:
        ds = ds.rename({k: v for k, v in alt_names.items() if k in ds})
        if spatial_info is not None:
            spatial_info = spatial_info.rename(columns={k: v for k, v in alt_names.items() if k in spatial_info.columns})

    # Standardize the spatial dimensions
    ordered_dims = [
        "subbasin_id",
        "unit_id",
    ]
    original_dim = [ds[d].dims[0] for d in ordered_dims if d in ds]
    original_dim = original_dim[0] if len(original_dim) > 0 else "not_applicable"

    # Find the correct spatial dimension
    spatial_dim = [d for d in ordered_dims if d in ds]
    if len(spatial_dim) >= 1:
        # Use the first value in the standard dictionary that is present in the dataset.
        spatial_dim = next(o for o in ordered_dims if o in ds)
    else:
        spatial_dim = None

    if spatial_dim is not None and spatial_dim != original_dim:
        ds = ds.swap_dims({original_dim: spatial_dim}).drop_vars(original_dim, errors="ignore")

    # Since Raven v4.1, 'basin_name' starting with 'sub_' has been renamed to 'basin_fullname', while a new 'basin_name' variable
    # without the prefix has been added. This new 'basin_fullname' is not required.
    ds = ds.drop_vars("basin_fullname", errors="ignore")

    # Datasets are exactly the same between Subbasin and Basin levels, so we need to make an educated guess
    is_subbasin = any("Subbasin" in ds[v].attrs.get("long_name", "") for v in ds.data_vars) or any(v in ds.data_vars for v in ["apport_lateral"])
    controlled_spatial = None
    if spatial_dim is not None:
        controlled_spatial = (
            "ComputationalUnit" if spatial_dim != "subbasin_id" else spatial_dim.replace("_id", "").capitalize() if is_subbasin else "DrainageArea"
        )

    # Ensure that all coordinates ending with "_id" are of type string
    for c in [cc for cc in ds.coords if cc.endswith("_id")]:
        ds[c] = ds[c].astype(str)
    if spatial_info is not None:
        for c in [cc for cc in spatial_info.columns if cc.endswith("_id")]:
            spatial_info[c] = spatial_info[c].astype(str)

    # Add relevant coordinates if available.
    if spatial_dim is not None and spatial_info is not None:
        df = spatial_info.drop_duplicates(spatial_dim).set_index(spatial_dim)

        columns = [c for c in list(VARIABLES["coordinates"].keys()) if c in df.columns]

        if spatial_dim == "subbasin_id":
            if is_subbasin is False:
                columns = [c for c in columns if not any(c.startswith(s) for s in ["unit", "subbasin"])]
            else:
                columns = [c for c in columns if not any(c.startswith(s) for s in ["unit"])]

        model = "raven" if "Raven_version" in ds.attrs else "hydrotel" if "HYDROTEL_version" in ds.attrs else "unknown"
        for col in columns:
            attrs = {
                attr: VARIABLES["coordinates"][col].get(f"{attr}_{model.lower()}", VARIABLES["coordinates"][col].get(attr, "unknown"))
                for attr in ["description", "long_name"]
            }
            coord = xr.DataArray(df[col], dims=[spatial_dim], coords={spatial_dim: df.index}, attrs=attrs)

            # Special cases
            if col.endswith("_id"):
                coord = coord.astype(str)
            if "area" in col and model == "raven":
                coord.attrs["units"] = "m2"
                coord = convert_units_to(coord, "km2")

            coord[spatial_dim] = coord[spatial_dim].astype(str)
            ds = ds.assign_coords({col: coord})

        # Also correct the attributes of the spatial dimension
        attrs = {
            attr: VARIABLES["coordinates"][spatial_dim].get(f"{attr}_{model.lower()}", VARIABLES["coordinates"][spatial_dim].get(attr, "unknown"))
            for attr in ["description", "long_name"]
        }
        ds[spatial_dim].attrs = attrs

        # If the dataset has a spatial dimension, add the "timeseries_id" cf_role attribute to it.
        ds = ds.squeeze()
        if spatial_dim in ds.dims:
            ds[spatial_dim].attrs["cf_role"] = "timeseries_id"

    # Manage the other variables
    controlled_vars = [v for v in ds.data_vars if v in VARIABLES["variables"]]
    for v in ds.data_vars:
        if v in controlled_vars:
            if ds[v].attrs.get("units", "unknown") != VARIABLES["variables"][v].get("canonical_units", "unknown"):
                ds[v] = convert_units_to(ds[v], VARIABLES["variables"][v]["canonical_units"])
            ds[v].attrs.update({k: v for k, v in VARIABLES["variables"][v].items() if k != "canonical_units"})
        ds[v].attrs["long_name"] = ds[v].attrs.get("long_name", "").split(" By")[0]
        if controlled_spatial is not None and spatial_dim in ds[v].dims:
            ds[v].attrs["aggregation_level"] = controlled_spatial

    # Since we squeezed the dataset and renamed the spatial dimension, it is preferable to clean the chunking information.
    # Default chunking provided by the hydrological models is often not optimal anyway.
    preferred_chunks = xs.io.estimate_chunks(ds, dims=ds.dims, target_mb=100)
    if "time" in preferred_chunks and len(ds["time"]) > 3:
        time_div = 365 if "D" in xr.infer_freq(ds["time"]) else 720 if "H" in xr.infer_freq(ds["time"]) else 1
        preferred_chunks["time"] = np.min([np.max([int(np.round(preferred_chunks["time"] / time_div) * time_div), time_div]), len(ds["time"])])

    for v in ds.data_vars:
        preferred = {d: preferred_chunks[d] for d in ds[v].dims}
        ds[v] = ds[v].chunk(preferred)
        ds[v].encoding["chunksizes"] = tuple(preferred[d] if d in preferred else ds[d].shape[0] for d in ds[v].dims)
        ds[v].encoding.pop("chunks", None)
        ds[v].encoding["preferred_chunks"] = preferred

    return ds


def aggregate_output(  # noqa: C901
    ds: xr.Dataset, by: Literal["hru", "rhhu", "unit", "subbasin"], to: Literal["subbasin", "drainage_area"]
) -> xr.Dataset:
    """
    Aggregate the model outputs to a different spatial unit. See the Notes section for more details.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to aggregate. The 'standardize_outputs' method must have been called on this dataset beforehand.
    by : {"hru", "rhhu", "unit", "subbasin"}
        The spatial unit to aggregate from.
        "unit" is the generic term for either "hru" or "rhhu", depending on the hydrological model used.
    to : {"subbasin", "drainage_area"}
        The spatial unit to aggregate to.

    Returns
    -------
    xr.Dataset
        The aggregated dataset.
    """
    if by.lower() == to.lower():
        raise ValueError("Invalid aggregation levels.")
    if by in ["hru", "rhhu"]:
        by = "unit"
    to_dim = f"{to}_id" if to != "drainage_area" else "subbasin_id"

    # Load the coordinates
    [ds[c].load() for c in ds.coords]

    # Prepare the chunking information for the output dataset, based on the input
    chunks_out = None
    if ds.chunks is not None:
        chunks_out = {d.replace(f"{by}_id", f"{to_dim}"): ds.chunks[d][0] for d in ds.dims if d in ds.chunks}

    if "subbasin_id" not in ds.coords:
        raise ValueError("The `standardize_outputs` method must be called before using the `aggregate_outputs` method.")

    ds_agg = None
    # Computational Unit --> Subbasin
    # Even if going from computational units to drainage area, it is much more simple to first aggregate to subbasins
    if by == "unit":
        ds_agg = ds.groupby("subbasin_id").map(lambda x: x.weighted(ds["unit_drainage_area"]).mean(dim="unit_id"))

        # Re-add the coordinates and attributes that were lost during the aggregation
        ds_for_coords = ds.swap_dims({"unit_id": "subbasin_id"}).drop_duplicates("subbasin_id")
        ds_for_coords = ds_for_coords.drop_vars([c for c in ds_for_coords.coords if ("unit" in c) or ((c in ds_agg.coords) and c != "subbasin_id")])
        ds_agg = ds_agg.assign_coords({c: ds_for_coords[c] for c in ds_for_coords.coords if not c.startswith(by.lower()) and c not in ds_agg.coords})

    if to == "drainage_area":
        if ds_agg is not None:
            # Take into account the intermediate aggregation to subbasins
            ds = ds_agg

        upsubid = xr.DataArray(
            data=ds["subbasin_id"].values.astype(str),
            coords={c: ds[c] for c in ds.coords if "subbasin_id" in ds[c].dims},
            dims=ds["subbasin_id"].dims,
        )
        upsubid = upsubid.assign_coords({"status": xr.zeros_like(upsubid, dtype=int)}).astype("<U1000")

        # Head subbasins
        upsubid["status"] = xr.where(~ds["subbasin_id"].isin(ds["dowsub_id"]), 1, upsubid["status"])

        # Recursively find the upstream subbasins for each subbasin, starting from the head subbasins and moving downstream
        while upsubid["status"].min() != 2:
            for idx in upsubid["subbasin_id"].where(upsubid["status"] == 1, drop=True):
                if idx.values not in (upsubid["dowsub_id"].sel(subbasin_id=upsubid["subbasin_id"].where(upsubid["status"] <= 1, drop=True)).values):
                    upsubid.loc[upsubid["subbasin_id"] == upsubid["dowsub_id"].sel(subbasin_id=idx)] += f",{upsubid.sel(subbasin_id=idx).values}"
                    upsubid["status"] = xr.where(upsubid["subbasin_id"] == idx, 2, upsubid["status"])
                    upsubid["status"] = xr.where(upsubid["subbasin_id"] == upsubid["dowsub_id"].sel(subbasin_id=idx), 1, upsubid["status"])
        upsubid = upsubid.drop_vars("status")
        ds = ds.assign_coords({"upsubid": upsubid})

        ds_agg = xr.zeros_like(ds)
        ds_agg = ds_agg.transpose("subbasin_id", ...)
        for sbid in ds["subbasin_id"]:
            subset_agg = (
                ds.sel(**{"subbasin_id": str(ds["upsubid"].sel(**{"subbasin_id": sbid}).values).split(",")})
                .weighted(ds["subbasin_drainage_area"])
                .mean(dim="subbasin_id")
            )
            for v in ds.data_vars:
                ds_agg[v].loc[ds_agg["subbasin_id"] == sbid] = subset_agg[v]

        # Remove the coordinates and attributes that are no longer relevant after the aggregation
        ds_agg = ds_agg.drop_vars("upsubid")
        ds_agg = ds_agg.drop_vars([c for c in ds_agg.coords if any(c.startswith(prefix) for prefix in ["unit", "subbasin"] if c != to_dim)])

    for v in ds_agg.data_vars:
        ds_agg[v].attrs["aggregation_level"] = "".join([w.capitalize() for w in to.split("_")])
        ds_agg[v].attrs["history"] = (
            ds_agg[v].attrs.get("history", "")
            + f"{dt.datetime.now().isoformat()}: Aggregated from {by.capitalize()} to {''.join([w.capitalize() for w in to.split('_')])} level."
        )

    try:
        ds_agg = ds_agg.sortby(ds_agg[f"{to_dim}"].astype(int))
    except TypeError:
        ds_agg = ds_agg.sortby(ds_agg[f"{to_dim}"])
    ds_agg = ds_agg.transpose("time", f"{to_dim}", ...)

    # Clean the chunking information after the aggregation
    if chunks_out is not None:
        ds_agg = ds_agg.chunk(chunks_out)
    for v in ds_agg.data_vars:
        ds_agg[v].encoding.pop("chunks", None)
        ds_agg[v].encoding.pop("preferred_chunks", None)
        ds_agg[v].encoding.pop("chunksizes", None)

    return ds_agg
