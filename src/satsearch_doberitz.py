import boto3
import geopandas as gpd
import json
import os

from argparse import ArgumentParser
from dotenv import load_dotenv
from tqdm import tqdm

from satsearch import Search
from satsearch.search import SatSearchError

# TODO: change from utils to .utils when modularizing
from utils import info_message, warning_message, debug_message


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


def get_prefix_filepath(href, collection='sentinel-s2-l2a'):
    prefix = href.replace(f's3://{collection}/', '')

    filename = prefix[prefix.rfind('/')+1:]  # Remove path
    dir_sector = ''.join(prefix.split('/')[1:4])
    dir_date = '-'.join(prefix.split('/')[4:7])

    output_filedir = os.path.join(collection, dir_sector, dir_date)
    output_filepath = os.path.join(output_filedir, filename)

    # Check if filedir exists and create it if not
    if not os.path.exists(output_filedir):
        os.makedirs(output_filedir)

    return prefix, output_filepath


def download_tile_band(href, collection='sentinel-s2-l2a'):
    """Download a specific S3 file URL

    Args:
        href (str): S3 file URL
        collection (str, optional): Earth-AWS collection.
            Defaults to 'sentinel-s2-l2a'.
    """
    prefix, output_filepath = get_prefix_filepath(
        href,
        collection=collection
    )

    # Check if file already exists to skip double downloading
    if not os.path.exists(output_filepath):
        # Download it to current directory
        s3_client.download_file(
            collection,
            prefix,
            output_filepath,
            {'RequestPayer': 'requester'}
        )

    return output_filepath


if __name__ == '__main__':
    """
    Use case:

    python satsearch_doberitz.py \
        --band_names b04 b08\
        --start_date 2020-01-01 \
        --end_date 2020-02-01 \
        --cloud_cover 1 \
        --download\
        --verbose

    OR

    python satsearch_doberitz.py --band_names b04 b08 --start_date 2020-01-01 --end_date 2020-02-01 --cloud_cover 1 --download --verbose
    """
    load_dotenv('.env')
    s3_client = boto3.client('s3')

    args = ArgumentParser()
    args.add_argument(
        '--geojson', type=str, default='doberitz_multipolygon.geojson'
    )
    args.add_argument('--tile_id', type=str)
    args.add_argument('--band_names', nargs='+', default=['B04', 'B08'])
    args.add_argument('--collection', type=str, default='sentinel-s2-l2a')
    args.add_argument('--start_date', type=str, default='2016-01-01')
    args.add_argument('--end_date', type=str, default='2022-01-01')
    args.add_argument('--cloud_cover', type=int, default=10)
    args.add_argument('--download', action='store_true')
    args.add_argument('--download_all', action='store_true')
    args.add_argument('--verbose', action='store_true')
    clargs = args.parse_args()

    # Get Sat-Search URL
    url_earth_search = os.environ.get('STAC_API_URL')

    # Get Input GeoJSON
    input_geojson = clargs.geojson

    # Set Date time start and stop for query
    eo_datetime = f'{clargs.start_date}/{clargs.end_date}'

    # Set cloud cover percentage for query
    eo_query = {
        'eo:cloud_cover': {'lt': clargs.cloud_cover}
    }

    # Load geojson into gpd.GeoDataFrame
    gdf_up42_geoms = gpd.read_file(clargs.geojson)

    # Build a GeoJSON bounding box around AOI(s)
    bounding_box = geom_to_bounding_box(gdf_up42_geoms)

    # Use Sat-Search to idenitify and load all meta data within search field
    search = Search(
        url=url_earth_search,
        intersects=bounding_box['features'][0]['geometry'],
        datetime=eo_datetime,
        query=eo_query,
        collections=[clargs.collection]
    )

    if clargs.verbose:
        info_message(f'Combined search: {search.found()} items')

    # Allocate all meta data for acquisition
    items = search.items()

    if clargs.verbose:
        info_message(items.summary())

    # Allocate MetaData in GeoJSON
    items_geojson = items.geojson()

    # Log all filepaths to queried scenes
    filepaths = {}

    if clargs.download:
        # Loop over GeoJSON Features
        for feat_ in tqdm(items_geojson['features']):
            # Loop over GeoJSON Bands
            filepaths[band_name] = []
            for band_name in tqdm(clargs.band_names):
                # Download the selected bands
                filepath_ = download_tile_band(
                    feat_['assets'][band_name.upper()]['href']
                )
                filepaths[band_name].append(filepath_)
    else:
        # Loop over GeoJSON Features
        for feat_ in tqdm(items_geojson['features']):
            # Loop over GeoJSON Bands
            filepaths[band_name] = []
            for band_name in tqdm(clargs.band_names):
                # Download the selected bands
                href = feat_['assets'][band_name.upper()]['href']
                _, output_filepath = get_prefix_filepath(
                    href, collection=clargs.collection
                )
                # info_message(output_filepath, os.path.exists(output_filepath))
                filepaths[band_name].append(output_filepath)

    for key, val in filepaths.items():
        for fpath_ in filepaths[band_name]:
            info_message(f"{fpath_} :: {os.path.exists(fpath_)}")

    # all_filenames = items.download_assets(requester_pays=True)
    """
    for feat_ in doberitz_feats['features']:
        try:
            search = Search(
                url=url_earth_search,
                intersects=feat_['geometry'],
                # bbox=eo_bbox,
                datetime=eo_datetime,
                query=eo_query,
                collections=['sentinel-s2-l2a']
                # limit=2
            )
            info_message(f'Combined search: {search.found()} items')

            items = search.items()
            info_message(items.summary())

            if clargs.download:
                # Download all Assets
                if clargs.download_all:
                    all_filenames = items.download_assets(requester_pays=True)
                else:
                    # NIR Band
                    b08_filenames = items.download(
                        'B08',
                        # filename_template='assets/${date}/${id}',
                        requester_pays=True
                    )
                    info_message(b08_filenames)

                    # RED Band
                    b04_filenames = items.download(
                        'B04',
                        # filename_template='assets/${date}/${id}',
                        requester_pays=True
                    )
                    info_message(b04_filenames)

        except SatSearchError as err:
            warning_message(err)
    """
