import os
import numpy as np
import geopandas as gpd
import folium
import zipfile
import json
from sentinelsat.sentinel import SentinelAPI, read_geojson, geojson_to_wkt
from matplotlib import pyplot as plt

import rasterio
from rasterio import plot
from rasterio.merge import merge

from getpass import getpass

from rasterio.mask import mask
from shapely.geometry import box
from fiona.crs import from_epsg

from dotenv import load_dotenv

from utils import info_message, warning_message, debug_message


def rescale_frame(frame, n_sig=5):
    """ Scale a single band frame by thresholding around its median

    Args:
        frame ([ndarray]): single band frame to be thresholded
        n_sig (int, optional): Size of threshold range. Defaults to 5.

    Returns:
        [ndarray]: thresholded single band frame
    """
    frame_med = np.median(frame)
    frame_std = np.std(frame)
    frame_flg = abs(frame - frame_med) > n_sig * frame_std
    frame_scl = frame.copy()
    frame_scl[frame_flg] = frame_med

    return frame_scl


def bounding_box_coords(gdf):
    """Derive the bounding box coordinates from the GeoDataframe

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame with set of geometries

    Returns:
        tuple: minimum and maximum longitude and latitude
    """
    # Physical Constraints Min Max
    min_lon = 180
    min_lat = 90
    max_lon = -180
    max_lat = -90

    for geom_ in gdf['geometry']:
        geom_rect_bound = geom_.minimum_rotated_rectangle.bounds
        minlon_, minlat_, maxlon_, maxlat_ = geom_rect_bound

        min_lon = minlon_ if minlon_ < min_lon else min_lon
        min_lat = minlat_ if minlat_ < min_lat else min_lat
        max_lon = maxlon_ if maxlon_ > max_lon else max_lon
        max_lat = maxlat_ if maxlat_ > max_lat else max_lat

    # mid_lon = 0.5 * (min_lon + max_lon)
    # mid_lat = 0.5 * (min_lat + max_lat)

    return min_lon, max_lon, min_lat, max_lat


def geom_to_bounding_box(gdf):
    """Generate a bounding box JSON from Geom

    Find the minimum and maxum longiture and latitude to search for relevant
    Sentinel-2 scenes surrounding the desired geometries

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame with input set of geometries

    Returns:
        dict: dicitonary representing a GeoJSON input with only the bounding box
    """
    min_lon, max_lon, min_lat, max_lat = bounding_box_coords(gdf)
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [min_lon, min_lat],
                            [max_lon, min_lat],
                            [max_lon, max_lat],
                            [min_lon, max_lat],
                            [min_lon, min_lat]
                        ]
                    ]
                }
            }
        ]
    }


def get_scenes_gdf(
        bounding_box, start_date, end_date, platform_name, processing_level,
        min_cloud_coverage, max_cloud_coverage, verbose=False):

    # connect to the API
    api = SentinelAPI(
        user=os.environ.get('COPERNICUS_USERNAME') or input('Username: '),
        password=os.environ.get('COPERNICUS_PASS') or getpass(),
        api_url='https://scihub.copernicus.eu/dhus'
    )

    footprint = geojson_to_wkt(bounding_box)

    products = api.query(
        footprint,
        date=(start_date, end_date),
        platform_name=platform_name,
        processing_level=processing_level,
        cloudcoverpercentage=(min_cloud_coverage, max_cloud_coverage)
    )

    if verbose:
        info_message(products.__len__())

    products_sorted = dict(
        sorted(
            products.items(),
            key=lambda x: x[1]['cloudcoverpercentage']
        )
    )

    scenes = api.to_geodataframe(products_sorted)

    fov_all_idents = [ident_.split('_')[5] for ident_ in scenes.identifier]
    fov_unique_idents = np.unique([
        ident_.split('_')[5] for ident_ in scenes.identifier
    ])

    dict_per_fov = {}
    for ident_ in fov_unique_idents:
        ident_match = np.array(fov_all_idents) == ident_
        dict_per_fov[ident_] = scenes.iloc[ident_match]

    if verbose:
        for key, val in dict_per_fov.items():
            info_message([
                key,
                val.iloc[0].beginposition,
                val.iloc[0].endposition,
                val.iloc[0].cloudcoverpercentage
            ])

    gdf = gpd.GeoDataFrame(
        [val.iloc[0] for _, val in dict_per_fov.items()]
    )

    return gdf, scenes, api


if __name__ == '__main__':
    from argparse import ArgumentParser
    args = ArgumentParser()
    args.add_argument(
        '--geojson',
        type=str,
        default='doberitz_multipolygon.geojson'
    )
    args.add_argument('--start_date', type=str, default='20150623')
    args.add_argument('--end_date', type=str, default='20211231')
    args.add_argument('--platform_name', type=str, default='Sentinel-2')
    args.add_argument('--processing_level', type=str, default='Level-2A')
    args.add_argument('--min_cloud_coverage', type=float, default=0)
    args.add_argument('--max_cloud_coverage', type=float, default=100)
    args.add_argument('--env', type=str, default='.env')
    args.add_argument('--verbose', action='store_true', default=False)
    args.add_argument('--verbose_plot', action='store_true', default=False)
    clargs = args.parse_args()

    start_date = clargs.start_date
    end_date = clargs.end_date

    platform_name = clargs.platform_name
    processing_level = clargs.processing_level

    min_cloud_coverage = clargs.min_cloud_coverage
    max_cloud_coverage = clargs.max_cloud_coverage

    input_geojson_fname = clargs.geojson

    verbose = clargs.verbose
    verbose_plot = clargs.verbose_plot

    load_dotenv(clargs.env)

    gdf_up42_geoms = gpd.read_file(input_geojson_fname)
    bounding_box = geom_to_bounding_box(gdf_up42_geoms)

    gdf_scenes, scenes, api = get_scenes_gdf(
        bounding_box, start_date, end_date, platform_name, processing_level,
        min_cloud_coverage, max_cloud_coverage, verbose=False)
