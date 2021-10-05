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

    # Connect to the API
    api = SentinelAPI(
        user=os.environ.get('COPERNICUS_USERNAME') or input('Username: '),
        password=os.environ.get('COPERNICUS_PASS') or getpass(),
        api_url='https://scihub.copernicus.eu/dhus'
    )

    footprint = geojson_to_wkt(bounding_box)

    products = api.query(
        footprint,
        date=(start_date, end_date),
        platformname=platform_name,
        processinglevel=processing_level,
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

    scene_all_idents = [ident_.split('_')[5] for ident_ in scenes.identifier]
    scene_unique_idents = np.unique([
        ident_.split('_')[5] for ident_ in scenes.identifier
    ])

    dict_per_scene = {}
    for ident_ in scene_unique_idents:
        ident_match = np.array(scene_all_idents) == ident_
        dict_per_scene[ident_] = scenes.iloc[ident_match]

    if verbose:
        for key, val in dict_per_scene.items():
            info_message([
                key,
                val.iloc[0].beginposition,
                val.iloc[0].endposition,
                val.iloc[0].cloudcoverpercentage
            ])

    gdf = gpd.GeoDataFrame(
        [val.iloc[0] for _, val in dict_per_scene.items()]
    )

    return gdf, scenes, api


def download_data(api, scenes, verbose=False):
    # Download Multiple UUIDs
    sentinel2_data = {}

    for k, (key, val) in enumerate(scenes.T.items()):
        if verbose:
            info_message([k, key, val.title])

        try:
            info_message(f"Attempting to Download: {val.uuid}")
            sentinel2_data[val.uuid] = api.download(val.uuid)
        except Exception as err:
            info_message(err)

        if val.uuid in sentinel2_data.keys() \
                and os.path.exists(sentinel2_data[val.uuid]['path']):
            info_message(f"Download Completed Sucessfully: {val.identifier}")
        else:
            info_message(f"File for {val.uuid} does not exist")

        if verbose:
            for key, val in sentinel2_data.items():
                info_message(val['title'])

    download_dir = r'./sentinelsat'
    if not os.path.exists(download_dir):
        os.mkdir(download_dir)

    # Unzip Multiple UUIDs
    for key, val in sentinel2_data.items():
        if not os.path.exists(f"sentinelsat/{val['title']}"):
            info_message(f"Unzipping {key} into {val['title']}")
            zip_ref = zipfile.ZipFile(val['path'], 'r')
            zip_ref.extractall(f"sentinelsat/{val['title']}")
            zip_ref.close()
            info_message("Unzip successful")

    if verbose:
        for key, val in sentinel2_data.items():
            info_message(os.listdir(f"sentinelsat/{val['title']}")[0])


def get_directory_structure(ident='T33UUU', file_ext='.jp2'):
    scene_unique_idents = np.unique([
        ident_.split('_')[5] for ident_ in scenes.identifier
    ])

    idx_ident = np.where(np.array(scene_unique_idents == ident))[0][0]
    gdf_ident = gdf_scenes.iloc[idx_ident]
    filename_date = gdf_ident.title.split('_')[2]

    dir_ident = os.path.join(
        'sentinelsat', gdf_ident.title, gdf_ident.filename
    )

    dir_granule = os.listdir(os.path.join(dir_ident, 'GRANULE'))[0]

    file_structure = {}
    file_structure['ident_dir'] = {}
    for res_ in [10, 20, 60]:
        res_dir = f'R{res_}m'
        file_structure['ident_dir'][res_dir] = os.path.join(
            dir_ident, 'GRANULE', dir_granule, 'IMG_DATA', res_dir
        )

    file_structure['jp2_files'] = {}
    for res_ in [10, 20, 60]:
        res_dir = f'R{res_}m'
        file_structure['jp2_files'][res_dir] = os.listdir(
            file_structure['ident_dir'][res_dir]
        )

    file_structure['jp2_filebase'] = {}
    for res_ in [10, 20, 60]:
        res_dir = f'R{res_}m'
        file_structure['jp2_filebase'][res_dir] = '_'.join(
            file_structure['jp2_files'][res_dir][0].split('_')[:2]
        )

    file_structure['jp2_filenames'] = {}
    for res_ in [10, 20, 60]:
        res_dir = f'R{res_}m'
        file_structure['jp2_filenames'][res_dir] = {}
        for k in range(12):
            band_label = f"B{k+1:0>2}"
            if band_label == 'B10':
                continue

            if band_label == 'B08':
                band_label = 'B8A'

            ident_dirname = '_'.join([
                file_structure['jp2_filebase'][res_dir],
                band_label,
                res_dir[1:]
            ]) + file_ext

            file_structure['jp2_filenames'][res_dir][band_label] = os.path.join(
                file_structure['ident_dir'][res_dir], ident_dirname
            )

    file_structure['scl_filename'] = {}
    file_structure['scl_filepath'] = {}
    for res_ in [10, 20, 60]:
        res_dir = f'R{res_}m'
        file_structure['scl_filename'][res_dir] = (
            f'{ident}_{filename_date}_SCL_{res_dir[1:]}.jp2'
        )
        file_structure['scl_filepath'][res_dir] = os.path.join(
            file_structure['ident_dir'][res_dir],
            file_structure['scl_filename'][res_dir]
        )

    return file_structure


def get_raster_data(file_structure):

    jp2_raster = {}
    scl_raster = {}
    scl_data = {}
    for res_ in [10, 20, 60]:
        res_dir = f'R{res_}m'
        jp2_raster[res_dir] = {}
        filenames_ = file_structure['jp2_filenames'][res_dir]
        for band_label, fname_ in filenames_.items():
            if not os.path.exists(fname_):
                continue

            jp2_raster[res_dir][band_label] = rasterio.open(
                fname_, driver='JP2OpenJPEG'
            )

        if not os.path.exists(file_structure['scl_filepath'][res_dir]):
            continue

        scl_raster[res_dir] = rasterio.open(
            file_structure['scl_filepath'][res_dir]
        )

        scl_data[res_dir] = scl_raster[res_dir].read()[0]

    return jp2_raster, scl_raster, scl_data


if __name__ == '__main__':
    from argparse import ArgumentParser
    args = ArgumentParser()
    args.add_argument(
        '--geojson',
        type=str,
        default='doberitz_multipolygon.geojson'
    )
    args.add_argument('--start_date', type=str, default='20190101')
    args.add_argument('--end_date', type=str, default='20200101')
    args.add_argument('--platform_name', type=str, default='Sentinel-2')
    args.add_argument('--processing_level', type=str, default='Level-2A')
    args.add_argument('--min_cloud_coverage', type=float, default=0)
    args.add_argument('--max_cloud_coverage', type=float, default=100)
    args.add_argument('--env', type=str, default='.env')
    args.add_argument('--download_all', action='store_true', default=False)
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

    if verbose_plot:
        f, ax = plt.subplots(figsize=(15, 15))
        gdf_scenes.plot(column='uuid', cmap=None, alpha=0.5, ax=ax)
        gdf_scenes.apply(lambda x: ax.annotate(text=x.identifier.split(
            '_')[5], xy=x.geometry.centroid.coords[0], ha='center'), axis=1)
        gdf_up42_geoms.plot(ax=ax, alpha=0.5, color='orange')
        plt.show()

    download_data(api, gdf_scenes, verbose=verbose)

    # if verbose_plot:
    #     # Follow specific Scence IDs
    #     gdfs = gpd.GeoDataFrame([gdf_T33UUU])
    #     fig, ax = plt.subplots(figsize=(15, 15))
    #     gdfs.plot(column='uuid', cmap=None, alpha=0.5, ax=ax)
    #     gdfs.apply(
    #         lambda x: ax.annotate(
    #             text=x.identifier.split('_')[5],
    #             xy=x.geometry.centroid.coords[0],
    #             ha='center'
    #         ),
    #         axis=1
    #     )

    #     gdf_up42_geoms.plot(ax=ax, alpha=0.5, color='orange')
    #     plt.show()

    files_T33UUU = get_directory_structure(
        ident='T33UUU', file_ext='.jp2'
    )

    jp2_T33UUU_raster, scl_T33UUU_raster, scl_T33UUU_data = get_raster_data(
        file_structure=files_T33UUU
    )
