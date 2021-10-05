# from logging import warn
import boto3
import geopandas as gpd
import json
import numpy as np
import os

from argparse import ArgumentParser
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from statsmodels.robust import scale

from fiona.crs import from_epsg
import rasterio
from rasterio import plot
from rasterio.merge import merge
from rasterio.mask import mask
from shapely.geometry import box

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

    # Loop of geometries to find min-max lon-lat coords
    for geom_ in gdf['geometry']:
        geom_rect_bound = geom_.minimum_rotated_rectangle.bounds
        minlon_, minlat_, maxlon_, maxlat_ = geom_rect_bound

        min_lon = minlon_ if minlon_ < min_lon else min_lon
        min_lat = minlat_ if minlat_ < min_lat else min_lat
        max_lon = maxlon_ if maxlon_ > max_lon else max_lon
        max_lat = maxlat_ if maxlat_ > max_lat else max_lat

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

    # Find bounding box coordinates
    min_lon, max_lon, min_lat, max_lat = bounding_box_coords(gdf)

    # Return GeoJSON format with bounding box as AOI
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
    """Generate prefix and filepath for boto3 s3 download request.

    Args:
        href (str): s3 url for single scene from satsearch service
        collection (str, optional): AWS bucket from which to grab jp2 files. Defaults to 'sentinel-s2-l2a'.

    Returns:
        tuple (str, str): `prefix` and `output_filepath` for boto3
    """

    # Isolate the boto3 S3 bucket prefix from the href url
    prefix = href.replace(f's3://{collection}/', '')

    # Isolate the jp2 filename from the prefix
    filename = prefix[prefix.rfind('/') + 1:]

    # Isolate the output file path from the prefix
    dir_resolution = prefix.split('/')[8]
    dir_sector = ''.join(prefix.split('/')[1:4])
    dir_date = '-'.join(prefix.split('/')[4:7])

    # Build the output file directory
    output_filedir = os.path.join(
        collection,
        dir_sector,
        dir_resolution,
        dir_date
    )

    # Build the output file path
    output_filepath = os.path.join(output_filedir, filename)

    # Check if filedir exists and create it if not
    if not os.path.exists(output_filedir):
        os.makedirs(output_filedir)

    return prefix, output_filepath


def download_tile_band(href, collection='sentinel-s2-l2a', s3_client=None):
    """Download a specific S3 file URL

    Args:
        href (str): S3 file URL
        collection (str, optional): Earth-AWS collection.
            Defaults to 'sentinel-s2-l2a'.
    """
    assert(s3_client is not None), 'assign s3_client in __main__'

    # Use the href to form the boto3 S3 prefix and output file path
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


def get_coords_from_geometry(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them

    Args:
        gdf (gpd.GeoDataFrame): gdf with list of geometries for each AOI

    Returns:
        list: coordinates to isolate the AOI
    """

    # Coords will output as list
    coords = []
    for feat_ in json.loads(gdf.to_json())['features']:
        # Loop over each geometry as a json output
        coords.append(feat_['geometry'])

    return coords


def compute_ndvi(band04, band08, n_sig=10):
    """Compute the NDVI image from band08 and band04 values

    Args:
        band04 (dict): Sentinel-2-L2A Band04 raster data
        band08 (dict): Sentinel-2-L2A Band08 raster data
        n_sig (int, optional): Number is sigma to quality as an outlier.
            Defaults to 10.

    Returns:
        tuple (np.array, affine.Affine): NDVI image and its related transform
    """
    # Convert from MAD to STD because Using the MAD is
    #   more agnostic to outliers than STD
    mad2std = 1.4826

    # By definition, the CRS is identical across bands
    gdf_crs = gdf_up42_geoms.to_crs(
        crs=band04['raster'].crs.data
    )

    # Compute the AOI coordinates from the raster crs data
    coords = get_coords_from_geometry(gdf_crs)

    # Mask Band04 data with AOI coords
    band04_masked, _ = mask(
        dataset=band04['raster'],
        shapes=coords,
        crop=True
    )

    # Mask Band08 data with AOI coords
    band08_masked, mask_transform = mask(
        dataset=band08['raster'],
        shapes=coords,
        crop=True
    )

    # Create NDVI from masked Band04 and Band08
    ndvi_masked = np.true_divide(
        band08_masked[0] - band04_masked[0],
        band08_masked[0] + band04_masked[0]
    )

    # FIll in missing data (outside mask) as zeros
    ndvi_masked[np.isnan(ndvi_masked)] = 0

    # median replacement from n_sigma outlier rejection
    med_ndvi = np.median(ndvi_masked.ravel())
    std_ndvi = scale.mad(ndvi_masked.ravel()) * mad2std

    outliers = abs(ndvi_masked - med_ndvi) > n_sig*std_ndvi
    ndvi_masked[outliers] = med_ndvi

    return ndvi_masked, mask_transform


if __name__ == '__main__':
    """
    Use case:

    python satsearch_doberitz.py \
        --band_names b04 b08\
        --start_date 2020-01-01 \
        --end_date 2020-02-01 \
        --cloud_cover 1 \
        --download\
        --verbose\
        --verbose_plot

    OR

    python satsearch_doberitz.py --band_names b04 b08 --start_date 2020-01-01 --end_date 2020-02-01 --cloud_cover 1 --download --verbose --verbose_plot
    """
    args = ArgumentParser()
    args.add_argument(
        '--geojson', type=str, default='doberitz_multipolygon.geojson'
    )
    args.add_argument('--scene_id', type=str)
    args.add_argument('--band_names', nargs='+', default=['B04', 'B08'])
    args.add_argument('--collection', type=str, default='sentinel-s2-l2a')
    args.add_argument('--start_date', type=str, default='2020-01-01')
    args.add_argument('--end_date', type=str, default='2020-02-01')
    args.add_argument('--cloud_cover', type=int, default=1)
    args.add_argument('--n_sig', type=float, default=10)
    args.add_argument('--download', action='store_true')
    args.add_argument('--download_all', action='store_true')
    args.add_argument('--env_filename', type=str, default='.env')
    args.add_argument('--verbose', action='store_true')
    args.add_argument('--verbose_plot', action='store_true')
    clargs = args.parse_args()

    load_dotenv(clargs.env_filename)
    s3_client = boto3.client('s3')

    # n_sig = 10
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

    band_names = [band_name_.upper() for band_name_ in clargs.band_names]
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
            for band_name_ in tqdm(band_names):
                if not band_name_ in filepaths.keys():
                    filepaths[band_name_] = []
                # Download the selected bands
                filepath_ = download_tile_band(
                    feat_['assets'][band_name_.upper()]['href'],
                    s3_client=s3_client
                )
                filepaths[band_name_].append(filepath_)
    else:
        # Loop over GeoJSON Features
        for feat_ in items_geojson['features']:
            # Loop over GeoJSON Bands
            for band_name_ in band_names:
                if not band_name_ in filepaths.keys():
                    filepaths[band_name_] = []
                # Download the selected bands
                href = feat_['assets'][band_name_.upper()]['href']
                _, output_filepath = get_prefix_filepath(
                    href, collection=clargs.collection
                )
                filepaths[band_name_].append(output_filepath)

    # Load all data into JSON data structure
    jp2_data = {}
    for band_name_, filepaths_ in filepaths.items():
        # loop over band names
        for fpath_ in filepaths_:
            # loop over file paths
            if not os.path.exists(fpath_):
                warning_message(f"{fpath_} does not exist")
                continue

            # Use filepath to identify scene_id, res, and date
            _, scene_id_, res_, date_, _ = fpath_.split('/')

            # Build up data structure for easier access later
            if scene_id_ not in jp2_data.keys():
                jp2_data[scene_id_] = {}
            if res_ not in jp2_data[scene_id_].keys():
                jp2_data[scene_id_][res_] = {}
            if date_ not in jp2_data[scene_id_][res_].keys():
                jp2_data[scene_id_][res_][date_] = {}
            if band_name_ not in jp2_data[scene_id_][res_][date_].keys():
                jp2_data[scene_id_][res_][date_][band_name_] = {}

            if clargs.verbose:
                info_message(f"{fpath_} :: {os.path.exists(fpath_)}")

            raster_ = rasterio.open(fpath_, driver='JP2OpenJPEG')
            # rast_data_ = raster_.read()

            jp2_data[scene_id_][res_][date_][band_name_]['raster'] = raster_
            # jp2_data[scene_id_][res_][date_][band_name_]['data'] = rast_data_

    # Compute NDVI for each Scene, Resolution, and Date
    for scene_id_, res_dict_ in jp2_data.items():
        for res_, date_dict_ in res_dict_.items():
            for date_, band_data_ in date_dict_.items():

                # Compute NDVI for individual scene, res, date
                ndvi_masked_, mask_transform_ = compute_ndvi(
                    band_data_['B04'],
                    band_data_['B08'],
                    n_sig=clargs.n_sig
                )

                # Store the NDVI and masked transform in data struct
                jp2_data[scene_id_][res_][date_]['ndvi'] = ndvi_masked_
                jp2_data[scene_id_][res_][date_]['transform'] = mask_transform_

                if clargs.verbose_plot:
                    # Sanity Check with imshow
                    plt.figure()
                    plt.imshow(ndvi_masked_)
                    plt.title(f"NDVI Image: {scene_id_} - {res_} - {date_}")

                    # Sanity Check with visual histogram
                    plt.figure()
                    plt.hist(
                        ndvi_masked_.ravel()[(ndvi_masked_.ravel() != 0)],
                        bins=100
                    )
                    plt.title(f"NDVI Hist: {scene_id_} - {res_} - {date_}")

    if clargs.verbose_plot:
        plt.show()
