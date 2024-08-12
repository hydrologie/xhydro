"""Module to compute geospatial operations useful in hydrology."""

from __future__ import annotations

import os
import tempfile
import urllib.request
import warnings
from pathlib import Path
from typing import Optional

import cartopy.crs as ccrs
import geopandas as gpd
import leafmap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystac_client
import rasterio
import rasterio.features
import rioxarray  # noqa: F401
import stackstac
import xarray as xr
import xvec  # noqa: F401
from matplotlib.colors import ListedColormap
from pystac.extensions.item_assets import ItemAssetsExtension
from pystac.extensions.projection import ProjectionExtension as proj  # noqa: N813
from shapely import Point
from tqdm.auto import tqdm
from xrspatial import aspect, slope

__all__ = [
    "land_use_classification",
    "land_use_plot",
    "surface_properties",
    "watershed_delineation",
    "watershed_properties",
]


def watershed_delineation(
    coordinates: list[tuple] | tuple | None = None,
    map: leafmap.Map | None = None,
) -> gpd.GeoDataFrame:
    """Calculate watershed delineation from pour point.

    Watershed delineation can be computed at any location in North America using HydroBASINS (hybas_na_lev01-12_v1c).
    The process involves assessing all upstream sub-basins from a specified pour point and consolidating them into a
    unified watershed.

    Parameters
    ----------
    coordinates : list of tuple, tuple, optional
        Geographic coordinates (longitude, latitude) for the location where watershed delineation will be conducted.
    map : leafmap.Map, optional
        If the function receives both a map and coordinates as inputs, it will generate and display watershed
        boundaries on the map. Additionally, any markers present on the map will be transformed into
        corresponding watershed boundaries for each marker.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing the watershed boundaries.
    """
    # level 12 HydroBASINS polygons dataset url (North America only at the moment)
    url = "https://s3.wasabisys.com/hydrometric/shapefiles/polygons.parquet"

    coordinates = [coordinates] if isinstance(coordinates, tuple) else coordinates

    # combine coordinates from both coordinates argument and markers on the map, if they exist
    if map is not None and any(map.draw_features):
        if coordinates is None:
            coordinates = []
        gdf_markers = gpd.GeoDataFrame.from_features(map.draw_features)[["geometry"]]
        gdf_markers = gdf_markers.loc[gdf_markers.type == "Point"]

        marker_coordinates = list(zip(gdf_markers.geometry.x, gdf_markers.geometry.y))
        coordinates = coordinates + marker_coordinates

    # cache and read level 12 HydroBASINS polygons' file
    proxies = urllib.request.getproxies()
    proxy = urllib.request.ProxyHandler(proxies)
    opener = urllib.request.build_opener(proxy)
    urllib.request.install_opener(opener)

    tmp_dir = Path(tempfile.gettempdir()).joinpath("polygons")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    polygon_path = tmp_dir.joinpath(Path(url).name)

    if not polygon_path.is_file():
        urllib.request.urlretrieve(url, polygon_path)  # noqa: S310

    gdf_hydrobasins = gpd.read_parquet(polygon_path)

    # compute watershed boundaries
    if coordinates is not None:
        gdf = (
            pd.concat(
                [
                    _compute_watershed_boundaries(tuple(coords), gdf_hydrobasins)
                    for coords in coordinates
                ]
            )[["HYBAS_ID", "UP_AREA", "geometry"]]
            .rename(columns={"UP_AREA": "Upstream Area (sq. km)."})
            .reset_index(drop=True)
        )

    # plot resulting geodataframe on map, if available
    if map is not None:
        style = {"fillOpacity": 0.65}
        hover_style = {"fillOpacity": 0.9}
        map.add_data(
            gdf,
            column="Upstream Area (sq. km).",
            scheme="Quantiles",
            cmap="YlGnBu",
            hover_style=hover_style,
            style=style,
            legend_title="Upstream Area (sq. km).",
            layer_name="Basins",
        )

    return gdf


def watershed_properties(
    gdf: gpd.GeoDataFrame,
    unique_id: str | None = None,
    projected_crs: int = 6622,
    output_format: str = "geopandas",
) -> gpd.GeoDataFrame | xr.Dataset:
    """Watershed properties extracted from a gpd.GeoDataFrame.

    The calculated properties are :
    - area
    - perimeter
    - gravelius
    - centroid coordinates

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing watershed polygons. Must have a defined .crs attribute.
    unique_id : str, optional
        The column name in the GeoDataFrame that serves as a unique identifier.
    projected_crs : int
        The projected coordinate reference system (crs) to utilize for calculations, such as determining watershed area.
    output_format : str
        One of either `xarray` (or `xr.Dataset`) or `geopandas` (or `gpd.GeoDataFrame`).

    Returns
    -------
    gpd.GeoDataFrame or xr.Dataset
        Output dataset containing the watershed properties.
    """
    if not gdf.crs:
        raise ValueError("The provided gpd.GeoDataFrame is missing the crs attribute.")
    projected_gdf = gdf.to_crs(projected_crs)

    # Calculate watershed properties
    output_dataset = gdf.loc[:, gdf.columns != gdf.geometry.name]
    output_dataset["area"] = projected_gdf.area
    output_dataset["perimeter"] = projected_gdf.length
    output_dataset["gravelius"] = (
        output_dataset.perimeter / 2 / np.sqrt(np.pi * output_dataset.area)
    )
    output_dataset["centroid"] = gdf.centroid.apply(lambda x: (x.x, x.y))

    if unique_id is not None:
        output_dataset.set_index(unique_id, inplace=True)

    if output_format in ("xarray", "xr.Dataset"):
        output_dataset = output_dataset.to_xarray()
        output_dataset["area"].attrs = {"units": "m2"}
        output_dataset["perimeter"].attrs = {"units": "m"}
        output_dataset["gravelius"].attrs = {"units": "m/m"}
        output_dataset["centroid"].attrs = {"units": ("degree_east", "degree_north")}

    return output_dataset


def _compute_watershed_boundaries(
    coordinates: tuple,
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Algorithm for watershed delineation using HydroBASINS (hybas_na_lev01-12_v1c). The process involves assessing
    all upstream sub-basins from a specified pour point and consolidating them into a unified watershed.

    Parameters
    ----------
    coordinates : tuple
        Geographic coordinates (longitude, latitude) for the location where watershed delineation will be conducted.
    gdf : gpd.GeoDataFrame
        HydroBASINS level 12 dataset in GeodataFrame format

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing the watershed boundaries.

    """
    spatial_index = gdf.sindex
    point = Point(coordinates)

    # find index of sub-basin (polygon) on which the selected coordinate/pour point falls on
    possible_matches = gdf.iloc[list(spatial_index.intersection(point.bounds))]
    polygon_index = possible_matches[possible_matches.intersects(point)]

    # find all sub-basins indexes upstream of polygon_index
    gdf_main_basin = gdf[gdf["MAIN_BAS"].isin(polygon_index["MAIN_BAS"])]
    all_sub_basins_indexes = _recursive_upstream_lookup(
        gdf_main_basin, polygon_index["HYBAS_ID"].to_list()
    )
    all_sub_basins_indexes.extend(polygon_index["HYBAS_ID"])

    # create GeoDataFrame from all sub-basins indexes and dissolve to a unique basin (new unique index)
    gdf_sub_basins = gdf_main_basin[
        gdf_main_basin["HYBAS_ID"].isin(set(all_sub_basins_indexes))
    ]
    gdf_basin = gdf_sub_basins.dissolve(by="MAIN_BAS")

    # # keep largest polygon if MultiPolygon
    if (
        gdf_basin.shape[0] > 0
        and gdf_basin.iloc[0].geometry.geom_type == "MultiPolygon"
    ):
        gdf_basin_exploded = gdf_basin.explode()
        gdf_basin = gdf_basin_exploded.loc[[gdf_basin_exploded.area.idxmax()]]

    # Raise warning for invalid or out of extent coordinates
    if gdf_basin.shape[0] == 0:
        warnings.warn(
            f"Could not return a watershed boundary for coordinates {coordinates}."
        )
    return gdf_basin


def _recursive_upstream_lookup(
    gdf: gpd.GeoDataFrame,
    direct_upstream_indexes: list,
    all_upstream_indexes: list | None = None,
):
    """Recursive function to iterate over each upstream sub-basin until all sub-basins in a watershed are identified.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        HydroBASINS level 12 dataset in GeodataFrame format stream of the pour point.
    direct_upstream_indexes : list
        List of all sub-basins indexes directly upstream.
    all_upstream_indexes : list
        Cumulative upstream indexes from `direct_upstream_indexes` accumulated during each iteration.

    Returns
    -------
    all_upstream_indexes
        Cumulative upstream indexes from `direct_upstream_indexes` accumulated during each iteration.
    """
    if all_upstream_indexes is None:
        all_upstream_indexes = []

    # get direct upstream indexes
    direct_upstream_indexes = gdf[gdf["NEXT_DOWN"].isin(direct_upstream_indexes)][
        "HYBAS_ID"
    ].to_list()
    if len(direct_upstream_indexes) > 0:
        all_upstream_indexes.extend(direct_upstream_indexes)
        _recursive_upstream_lookup(gdf, direct_upstream_indexes, all_upstream_indexes)
    return all_upstream_indexes


def _flatten(x, dim="time"):
    # FIXME: assert statements should only be found in test code
    assert isinstance(x, xr.DataArray)  # noqa: S101
    if len(x[dim].values) > len(set(x[dim].values)):
        x = x.groupby(dim).map(stackstac.mosaic)

    return x


def surface_properties(
    gdf: gpd.GeoDataFrame,
    unique_id: str | None = None,
    projected_crs: int = 6622,
    output_format: str = "geopandas",
    operation: str = "mean",
    dataset_date: str = "2021-04-22",
    collection: str = "cop-dem-glo-90",
) -> gpd.GeoDataFrame | xr.Dataset:
    """Surface properties for watersheds.

    Surface properties are calculated using Copernicus's GLO-90 Digital Elevation Model. By default, the dataset
    has a geographic coordinate system (EPSG: 4326) and this function expects a projected crs for more accurate results.

    The calculated properties are :
    - elevation (meters)
    - slope (degrees)
    - aspect ratio (degrees)

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing watershed polygons. Must have a defined .crs attribute.
    unique_id : str, optional
        The column name in the GeoDataFrame that serves as a unique identifier.
    projected_crs : int
        The projected coordinate reference system (crs) to utilize for calculations, such as determining watershed area.
    output_format : str
        One of either `xarray` (or `xr.Dataset`) or `geopandas` (or `gpd.GeoDataFrame`).
    operation : str
        Aggregation statistics such as `mean` or `sum`.
    dataset_date : str
        Date (%Y-%m-%d) for which to select the imagery from the dataset. Date must be available.
    collection : str
        Collection name from the Planetary Computer STAC Catalog. Default is `cop-dem-glo-90`.

    Returns
    -------
    gpd.GeoDataFrame or xr.Dataset
        Output dataset containing the surface properties.
    """
    # Geometries are projected to make calculations more accurate
    projected_gdf = gdf.to_crs(projected_crs)

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
    )

    search = catalog.search(
        collections=[collection],
        bbox=gdf.total_bounds,
    )

    items = list(search.get_items())

    # Create a mosaic of
    da = stackstac.stack(items)
    da = _flatten(
        da, dim="time"
    )  # https://hrodmn.dev/posts/stackstac/#wrangle-the-time-dimension
    ds = (
        da.sel(time=dataset_date)
        .coarsen({"y": 5, "x": 5}, boundary="trim")
        .mean()
        .to_dataset(name="elevation")
        .rio.write_crs("epsg:4326", inplace=True)
        .rio.reproject(projected_crs)
        .isel(band=0)
    )

    # Use Xvec to extract elevation for each geometry in the projected gdf
    da_elevation = ds.xvec.zonal_stats(
        projected_gdf.geometry, x_coords="x", y_coords="y", stats=operation
    )["elevation"].squeeze()

    da_slope = slope(ds.elevation)

    # Use Xvec to extract slope for each geometry in the projected gdf
    da_slope = da_slope.to_dataset(name="slope").xvec.zonal_stats(
        projected_gdf.geometry, x_coords="x", y_coords="y", stats=operation
    )["slope"]

    da_aspect = aspect(ds.elevation)

    # Use Xvec to extract aspect for each geometry in the projected gdf
    da_aspect = da_aspect.to_dataset(name="aspect").xvec.zonal_stats(
        projected_gdf.geometry, x_coords="x", y_coords="y", stats=operation
    )["aspect"]

    output_dataset = xr.merge([da_elevation, da_slope, da_aspect]).astype("float32")

    # Add attributes for each variable
    output_dataset["slope"].attrs = {"units": "degrees"}
    output_dataset["aspect"].attrs = {"units": "degrees"}
    output_dataset["elevation"].attrs = {"units": "m"}

    if unique_id is not None:
        output_dataset = output_dataset.assign_coords(
            {unique_id: ("geometry", gdf[unique_id])}
        )
        output_dataset = output_dataset.swap_dims({"geometry": unique_id})

    if output_format in ("geopandas", "gpd.GeoDataFrame"):
        output_dataset = output_dataset.drop("geometry").to_dataframe()

    return output_dataset


def _merge_stac_dataset(catalog, bbox_of_interest, year):
    search = catalog.search(collections=["io-lulc-9-class"], bbox=bbox_of_interest)
    items = search.item_collection()

    # The STAC metadata contains some information we'll want to use when creating
    # our merged dataset. Get the EPSG code of the first item and the nodata value.
    item = items[0]

    # Create a single DataArray from out multiple resutls with the corresponding
    # rasters projected to a single CRS. Note that we set the dtype to ubyte, which
    # matches our data, since stackstac will use float64 by default.
    stack = (
        stackstac.stack(
            items,
            dtype=np.uint8,
            fill_value=np.uint8(255),
            bounds_latlon=bbox_of_interest,
            epsg=item.properties["proj:epsg"],
            sortby_date=False,
            rescale=False,
        )
        .assign_coords(
            time=pd.to_datetime([item.properties["start_datetime"] for item in items])
            .tz_convert(None)
            .to_numpy()
        )
        .sortby("time")
    )

    merged = stack.squeeze().compute()
    if year == "latest":
        year = str(merged.time.dt.year[-1].values)
    else:
        year = str(year)
        if not year.isdigit():
            raise TypeError(f"Expected year argument {year} to be a digit.")

    merged = merged.sel(time=year).min("time")
    return merged, item


def _count_pixels_from_bbox(
    gdf, idx, catalog, unique_id, values_to_classes, year, pbar
):
    bbox_of_interest = gdf.iloc[[idx]].total_bounds

    merged, item = _merge_stac_dataset(catalog, bbox_of_interest, year)
    epsg = item.properties["proj:epsg"]

    # Mask with polygon
    merged = merged.rio.write_crs(epsg).rio.clip([gdf.to_crs(epsg).iloc[idx].geometry])

    data = merged.data.ravel()
    data = data[data != 0]

    df = pd.DataFrame(
        pd.value_counts(data, sort=False).rename(values_to_classes) / data.shape[0]
    )

    if unique_id is not None:
        column_name = [gdf[unique_id].iloc[idx]]
    else:
        column_name = [idx]
    df.columns = column_name

    pbar.set_description(f"Spatial operations: processing site {column_name[0]}")

    return df.T


def land_use_classification(
    gdf: gpd.GeoDataFrame,
    unique_id: str | None = None,
    output_format: str = "geopandas",
    collection="io-lulc-9-class",
    year: str | int = "latest",
) -> gpd.GeoDataFrame | xr.Dataset:
    """Calculate land use classification.

    Calculate land use classification for each polygon from a gpd.GeoDataFrame. By default,
    the classes are generated from the Planetary Computer's STAC catalog using the
    `10m Annual Land Use Land Cover` dataset.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing watershed polygons. Must have a defined .crs attribute.
    unique_id : str
        GeoDataFrame containing watershed polygons. Must have a defined .crs attribute.
    output_format : str
        One of either `xarray` (or `xr.Dataset`) or `geopandas` (or `gpd.GeoDataFrame`).
    collection : str
        Collection name from the Planetary Computer STAC Catalog.
    year : str | int
        Land use dataset year between 2017 and 2022.

    Returns
    -------
    gpd.GeoDataFrame or xr.Dataset
        Output dataset containing the watershed properties.
    """
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
    )
    collection = catalog.get_collection(collection)
    ia = ItemAssetsExtension.ext(collection)
    x = ia.item_assets["data"]
    class_names = {x["summary"]: x["values"][0] for x in x.properties["file:values"]}

    values_to_classes = {
        v: "_".join(("pct", k.lower().replace(" ", "_")))
        for k, v in class_names.items()
    }

    pbar = tqdm(gdf.index, position=0, leave=True)

    output_dataset = pd.concat(
        [
            _count_pixels_from_bbox(
                gdf, idx, catalog, unique_id, values_to_classes, year, pbar
            )
            for idx in pbar
        ],
        axis=0,
    ).fillna(0)

    if unique_id is not None:
        output_dataset.index.name = unique_id

    if output_format in ("xarray", "xr.Dataset"):
        # TODO : Determine if cf-compliant names exist for physiographical data (area, perimeter, etc.)
        output_dataset = output_dataset.to_xarray()
        for var in output_dataset:
            output_dataset[var].attrs = {"units": "percent"}
    return output_dataset


def land_use_plot(
    gdf: gpd.GeoDataFrame,
    idx: int = 0,
    unique_id: str | None = None,
    collection: str = "io-lulc-9-class",
    year: str | int = "latest",
) -> None:
    """Plot a land use map for a specific year and watershed.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing watershed polygons. Must have a defined .crs attribute.
    idx : int
        Index to select in gpd.GeoDataFrame.
    unique_id : str
        GeoDataFrame containing watershed polygons. Must have a defined .crs attribute.
    collection : str
        Collection name from the Planetary Computer STAC Catalog.
    year : str | int
        Land use dataset year between 2017 and 2022.

    Returns
    -------
    None
        Nothing to return.
    """
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
    )

    collection = catalog.get_collection(collection)
    ia = ItemAssetsExtension.ext(collection)
    x = ia.item_assets["data"]
    class_names = {x["summary"]: x["values"][0] for x in x.properties["file:values"]}

    gdf = gdf.iloc[[idx]]

    if unique_id is not None:
        name = f"- {gdf[unique_id].values[0]}"
    else:
        name = ""

    bbox_of_interest = gdf.total_bounds

    merged, item = _merge_stac_dataset(catalog, bbox_of_interest, year)

    epsg = proj.ext(item).epsg

    class_count = len(class_names)

    with rasterio.open(item.assets["data"].href) as src:
        colormap_def = src.colormap(1)  # get metadata colormap for band 1
        colormap = [
            np.array(colormap_def[i]) / 255 for i in range(class_count)
        ]  # transform to matplotlib color format

    cmap = ListedColormap(colormap)
    fig, ax = plt.subplots(
        figsize=(12, 6),
        dpi=125,
        subplot_kw=dict(projection=ccrs.epsg(epsg)),
        frameon=False,
    )
    p = merged.plot(
        ax=ax,
        transform=ccrs.epsg(epsg),
        cmap=cmap,
        add_colorbar=False,
        vmin=0,
        vmax=class_count,
    )
    ax.set_title(f"Land use classification - {year} {name}")

    cbar = plt.colorbar(p)
    cbar.set_ticks(range(class_count))
    cbar.set_ticklabels(class_names)

    gdf.to_crs(epsg).boundary.plot(ax=ax, alpha=0.9, color="black")

    return fig
