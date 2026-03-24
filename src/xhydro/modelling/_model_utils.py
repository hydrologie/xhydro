"""Hidden utilities for HYDROTEL and RavenPy models."""

import datetime as dt
from pathlib import Path
from typing import Literal

import pandas as pd
import xarray as xr
import yaml
from xclim.core.units import convert_units_to


with Path(Path(__file__).parent / "variables.yml").open() as f:
    VARIABLES = yaml.safe_load(f)


def standardize_output(ds, spatial_info: pd.DataFrame | None = None, alt_names: dict[str, str] | None = None) -> xr.Dataset:
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
        spatial_info = (
            spatial_info.rename(columns={k: v for k, v in alt_names.items() if k in spatial_info.columns}) if spatial_info is not None else None
        )

    # Step 1: Standardize the spatial dimensions
    ordered_dims = [
        "subbasin_id",
        "hru_id",
        "rhhu_id",
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

    # Add relevant coordinates if available.
    if spatial_dim is not None and spatial_info is not None:
        ds[spatial_dim] = ds[spatial_dim].astype(str)
        df = spatial_info.drop_duplicates(spatial_dim).set_index(spatial_dim)

        columns = [c for c in list(VARIABLES["coordinates"].keys()) if c in df.columns]

        if spatial_dim == "subbasin_id":
            columns = [c for c in columns if not any(s in c for s in ["hru", "rhhu"])]

            # Assess whether the dataset contains subbasin-level or drainage-area-level information
            if "ByDrainageArea" in ds[[v for v in ds.data_vars if spatial_dim in ds[v].dims][0]].attrs.get("long_name", "") or any(
                any(s in v for s in ["q_", "debit"]) for v in ds.data_vars
            ):
                columns = [c for c in columns if not any(s in c for s in ["subbasin"])]

        for col in columns:
            coord = xr.DataArray(df[col], dims=[spatial_dim], coords={spatial_dim: df.index}, attrs=VARIABLES["coordinates"][col])
            # Special handling
            if "_id" in col:
                coord = coord.astype(str)
            if "area" in col and "Raven" in ds.attrs.get("history", ""):
                coord.attrs["units"] = "m2"
                coord = convert_units_to(coord, "km2")

            coord[spatial_dim] = coord[spatial_dim].astype(str)
            ds = ds.assign_coords({col: coord})

        # Also correct the attributes of the spatial dimension
        ds[spatial_dim].attrs = VARIABLES["coordinates"][spatial_dim]

        # If the dataset has a spatial dimension, add the "timeseries_id" cf_role attribute to it.
        ds = ds.squeeze()
        if spatial_dim in ds.dims:
            ds[spatial_dim].attrs["cf_role"] = "timeseries_id"

    # Manage the other variables
    vars = [v for v in ds.data_vars if v in VARIABLES["variables"]]
    for v in vars:
        if ds[v].attrs.get("units", "unknown") != VARIABLES["variables"][v].get("canonical_units", "unknown"):
            ds[v] = convert_units_to(ds[v], VARIABLES["variables"][v]["canonical_units"])
        ds[v].attrs.update({k: v for k, v in VARIABLES["variables"][v].items() if k != "canonical_units"})

    # Since we squeezed the dataset and renamed the spatial dimension, it is preferable to clean the chunking information
    for v in ds.data_vars:
        preferred = ds[v].encoding.get("preferred_chunks", {d: len(ds[v][d]) for d in ds[v].dims})
        if original_dim in preferred.keys():
            preferred[spatial_dim] = preferred.pop(original_dim)
        ds[v] = ds[v].chunk(preferred)
        ds[v].encoding["chunksizes"] = tuple(preferred[d] if d in preferred else ds[d].shape[0] for d in ds[v].dims)
        ds[v].encoding.pop("chunks", None)
        ds[v].encoding["preferred_chunks"] = preferred

    return ds


def aggregate_output(  # noqa: C901
    ds: xr.Dataset, by: Literal["hru", "rhhu", "subbasin"], to: Literal["subbasin", "drainage_area"]
) -> xr.Dataset:
    """
    Aggregate the model outputs to a different spatial unit. See the Notes section for more details.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to aggregate. The 'standardize_outputs' method must have been called on this dataset beforehand.
    by : {"hru", "rhhu", "subbasin"}
        The spatial unit to aggregate from.
    to : {"subbasin", "drainage_area"}
        The spatial unit to aggregate to.

    Returns
    -------
    xr.Dataset
        The aggregated dataset.
    """
    if to.lower() == by.lower():
        raise ValueError("Invalid aggregation levels.")

    info = {
        "hru": {
            "clean": "HRU",
            "dim": "hru_id",
            "area_var": "hru_area",
        },
        "rhhu": {
            "clean": "RHHU",
            "dim": "rhhu_id",
            "area_var": "rhhu_area",
        },
        "subbasin": {
            "clean": "Subbasin",
            "dim": "subbasin_id",
            "area_var": "subbasin_area",
        },
        "drainage_area": {
            "clean": "DrainageArea",
            "dim": "subbasin_id",
            "area_var": "drainage_area",
        },
    }

    if "subbasin_id" not in ds.coords:
        raise ValueError("The `standardize_outputs` method must be called before using the `aggregate_outputs` method.")

    if by in ["hru", "rhhu"]:
        # Even if going from computational units to drainage area, it is much more simple to first aggregate to subbasins
        ds_agg = ds.groupby("subbasin_id").map(lambda x: x.weighted(x[info[by]["area_var"]]).mean(dim=info[by]["dim"]))

        # Re-add the coordinates and attributes that were lost during the aggregation
        ds_for_coords = ds.swap_dims({info[by]["dim"]: "subbasin_id"}).drop_duplicates("subbasin_id")
        ds_for_coords = ds_for_coords.drop_vars(
            [c for c in ds_for_coords.coords if (by.lower() in c) or ((c in ds_agg.coords) and c != "subbasin_id")]
        )
        ds_agg = ds_agg.assign_coords({c: ds_for_coords[c] for c in ds_for_coords.coords if not c.startswith(by.lower()) and c not in ds_agg.coords})

    if to == "drainage_area":
        if by in ["hru", "rhhu"]:
            # Take into account the intermediate aggregation to subbasins
            by_clean = info[by]["clean"]
            by = "subbasin"
            info[by]["clean"] = by_clean
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
                .weighted(ds["subbasin_area"])
                .mean(dim="subbasin_id")
            )
            for v in ds.data_vars:
                ds_agg[v].loc[ds_agg["subbasin_id"] == sbid] = subset_agg[v]

        # Remove the coordinates and attributes that are no longer relevant after the aggregation
        ds_agg = ds_agg.drop_vars("upsubid")
        ds_agg = ds_agg.drop_vars(
            [c for c in ds_agg.coords if any(c.startswith(prefix) for prefix in ["hru", "rhhu", "subbasin"]) and c != "subbasin_id"]
        )

    for v in ds_agg.data_vars:
        ds_agg[v].attrs["long_name"] = ds[v].attrs.get("long_name", "").replace(f"By{info[by]['clean']}", f"By{info[to]['clean']}")
        ds_agg[v].attrs["history"] = f"{dt.datetime.now().isoformat()}: Aggregated from {info[by]['clean']} to {info[to]['clean']} level."

    try:
        ds_agg = ds_agg.sortby(ds_agg[info[to]["dim"]].astype(int))
    except TypeError:
        ds_agg = ds_agg.sortby(ds_agg[info[to]["dim"]])
    ds_agg = ds_agg.transpose("time", info[to]["dim"], ...)

    return ds_agg
