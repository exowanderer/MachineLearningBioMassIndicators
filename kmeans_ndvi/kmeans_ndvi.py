import boto3
import geopandas as gpd
import numpy as np
import os
import rasterio

from datetime import datetime
# from logging import warn
# from logging import debug, warning
from tqdm import tqdm

from satsearch import Search

# TODO: change from utils to .utils when modularizing
from .utils import (
    geom_to_bounding_box,
    get_prefix_filepath,
    download_tile_band,
    compute_ndvi,
    kmeans_spatial_cluster,
    kmeans_temporal_cluster,
    info_message,
    warning_message,
    debug_message
)


class KMeansNDVI(object):
    """[summary]

    Args:
        object ([type]): [description]
    """

    def __init__(
            self, geojson, start_date='2020-01-01', end_date='2020-02-01',
            cloud_cover=1, collection='sentinel-s2-l2a',
            band_names=['B04', 'B08'], download=False, n_clusters=5, n_sig=10,
            quantile_range=(1, 99), verbose=False, verbose_plot=False,
            hist_bins=100):
        """[summary]

        Args:
            geojson ([type]): [description]
            start_date (str, optional): [description]. Defaults to '2020-01-01'.
            end_date (str, optional): [description]. Defaults to '2020-02-01'.
            cloud_cover (int, optional): [description]. Defaults to 1.
            collection (str, optional): [description]. Defaults to 'sentinel-s2-l2a'.
            band_names (list, optional): [description]. Defaults to ['B04', 'B08'].
            download (bool, optional): [description]. Defaults to False.
            n_clusters (int, optional): [description]. Defaults to 5.
            n_sig (int, optional): [description]. Defaults to 10.
            quantile_range (tuple, optional): [description]. Defaults to (1, 99).
            verbose (bool, optional): [description]. Defaults to False.
            verbose_plot (bool, optional): [description]. Defaults to False.
            hist_bins (int, optional): [description]. Defaults to 100.
        """
        self.scenes = {}  # Data structure for JP2 Data
        self.s3_client = boto3.client('s3')
        self.geojson = geojson
        self.start_date = start_date
        self.end_date = end_date
        self.cloud_cover = cloud_cover
        self.collection = collection
        self.band_names = band_names
        self.download = download
        self.n_clusters = n_clusters
        self.n_sig = n_sig
        self.quantile_range = quantile_range
        self.verbose = verbose
        self.verbose_plot = verbose_plot
        self.hist_bins = hist_bins

    def search_earth_aws(self):
        """[summary]
        """

        # Get Sat-Search URL
        self.url_earth_search = os.environ.get('STAC_API_URL')

        # Set Date time start and stop for query
        eo_datetime = f'{self.start_date}/{self.end_date}'

        # Set cloud cover percentage for query
        eo_query = {
            'eo:cloud_cover': {'lt': self.cloud_cover}
        }

        # Load geojson into gpd.GeoDataFrame
        self.gdf = gpd.read_file(self.geojson)

        # Build a GeoJSON bounding box around AOI(s)
        bounding_box = geom_to_bounding_box(self.gdf)

        # Use Sat-Search to idenitify and load all meta data within search field
        self.search = Search(
            url=self.url_earth_search,
            intersects=bounding_box['features'][0]['geometry'],
            datetime=eo_datetime,
            query=eo_query,
            collections=[self.collection]
        )

    def download_and_acquire_images(self):
        """Cycle through geoJSON to download files (if download is True) and return list of files for later storage
        """
        assert(self.s3_client is not None), \
            'Please assign and allocate an s3_client'

        self.search_earth_aws()

        if self.verbose:
            info_message(f'Combined search: {self.search.found()} items')

        # Allocate all meta data for acquisition
        self.items = self.search.items()

        if self.verbose:
            info_message(self.items.summary())

        info_message("Allocating metadata in geoJSON")
        self.items_geojson = self.items.geojson()

        # Log all filepaths to queried scenes
        if self.download:
            # Loop over GeoJSON Features
            for feat_ in tqdm(self.items_geojson['features']):
                # Loop over GeoJSON Bands
                for band_name_ in tqdm(self.band_names):
                    # if not band_name_ in filepaths.keys():
                    #     filepaths[band_name_] = []
                    # Download the selected bands
                    _ = download_tile_band(  # filepath_
                        feat_['assets'][band_name_.upper()]['href'],
                        s3_client=self.s3_client
                    )
                    # filepaths[band_name_].append(filepath_)

        self.filepaths = {}
        # Loop over GeoJSON Features to Allocate all requested files
        for feat_ in self.items_geojson['features']:
            # Loop over GeoJSON Bands
            for band_name_ in self.band_names:
                if not band_name_ in self.filepaths.keys():
                    self.filepaths[band_name_] = []

                # Download the selected bands
                href = feat_['assets'][band_name_.upper()]['href']
                _, output_filepath = get_prefix_filepath(
                    href, collection=self.collection
                )
                self.filepaths[band_name_].append(output_filepath)

    def load_data_into_struct(self):
        """Load all files in filepaths into data structure self.scenes
        """

        for band_name_, filepaths_ in self.filepaths.items():
            # loop over band names
            for fpath_ in filepaths_:
                # loop over file paths
                if not os.path.exists(fpath_):
                    warning_message(f"{fpath_} does not exist")
                    continue

                # Use filepath to identify scene_id, res, and date
                _, scene_id_, res_, date_, _ = fpath_.split('/')

                # Adjust month from 1 to 2 digits if necessary
                year_, month_, day_ = date_.split('-')
                month_ = f"{month_:0>2}"
                day_ = f"{day_:0>2}"
                date_ = f"{year_}-{month_}-{day_}"

                # Build up data structure for easier access later
                if scene_id_ not in self.scenes.keys():
                    self.scenes[scene_id_] = {}
                if res_ not in self.scenes[scene_id_].keys():
                    self.scenes[scene_id_][res_] = {}
                if date_ not in self.scenes[scene_id_][res_].keys():
                    self.scenes[scene_id_][res_][date_] = {}
                if band_name_ not in self.scenes[scene_id_][res_][date_].keys():
                    self.scenes[scene_id_][res_][date_][band_name_] = {}

                if self.verbose:
                    info_message(f"{fpath_} :: {os.path.exists(fpath_)}")

                raster_ = {}
                raster_['raster'] = rasterio.open(fpath_, driver='JP2OpenJPEG')
                self.scenes[scene_id_][res_][date_][band_name_] = raster_

    def compute_ndvi_for_all(self):
        """Cycle over self.scenes and compute NDVI for each scene and date_
        """
        for scene_id_, res_dict_ in tqdm(self.scenes.items()):
            for res_, date_dict_ in tqdm(res_dict_.items()):
                for date_, band_data_ in tqdm(date_dict_.items()):
                    if not 'B04' in band_data_.keys() and \
                            not 'B08' in band_data_.keys():
                        warning_message(
                            'NDVI cannot be computed without both Band04 and Band08'
                        )
                        continue

                    # Compute NDVI for individual scene, res, date
                    ndvi_masked_, mask_transform_ = compute_ndvi(
                        band_data_['B04'],
                        band_data_['B08'],
                        gdf=self.gdf,
                        n_sig=self.n_sig,
                        scene_id=scene_id_,
                        res=res_,
                        date=date_,
                        bins=self.hist_bins,
                        verbose=self.verbose,
                        verbose_plot=self.verbose_plot
                    )

                    # Store the NDVI and masked transform in data struct
                    self.scenes[scene_id_][res_][date_]['ndvi'] = ndvi_masked_
                    self.scenes[scene_id_][res_][date_]['transform'] = mask_transform_

    def allocate_ndvi_timeseries(self):
        """Allocate NDIV images per scene and date into time series
        """
        for scene_id_, res_dict_ in tqdm(self.scenes.items()):
            for res_, date_dict_ in tqdm(res_dict_.items()):
                timestamps_ = []
                timeseries_ = []
                for date_, dict_ in tqdm(date_dict_.items()):
                    if 'ndvi' not in dict_.keys():
                        continue

                    if date_ != 'timeseries':
                        timestamps_.append(datetime.fromisoformat(date_))
                        timeseries_.append(dict_['ndvi'])

                timeseries_ = np.array(timeseries_)

                timeseries_dict = {}
                timeseries_dict['ndvi'] = timeseries_
                timeseries_dict['timestamps'] = timestamps_

                self.scenes[scene_id_][res_]['timeseries'] = timeseries_dict

    def compute_spatial_kmeans(self):
        """Cycle through all NDVI and Compute NDVI for each Scene, Resolution, 
            and Date
        """
        for scene_id_, res_dict_ in tqdm(self.scenes.items()):
            for res_, date_dict_ in tqdm(res_dict_.items()):
                for date_, band_data_ in tqdm(date_dict_.items()):
                    if not 'B04' in band_data_.keys() and \
                            not 'B08' in band_data_.keys():
                        warning_message(
                            'NDVI cannot be computed without both Band04 and Band08'
                        )
                        continue

                    # Compute K-Means Spatial Clustering per Image
                    kmeans_ = kmeans_spatial_cluster(
                        self.scenes[scene_id_][res_][date_]['ndvi'],
                        n_clusters=self.n_clusters,
                        quantile_range=self.quantile_range,
                        verbose=self.verbose,
                        verbose_plot=self.verbose_plot,
                        scene_id=scene_id_,
                        res=res_,
                        date=date_
                    )

                    # Store the NDVI and masked transform in data struct
                    self.scenes[scene_id_][res_][date_]['kmeans'] = kmeans_

    def compute_temporal_kmeans(self):
        """Cycle over all NDVI time series and Compute NDVI for each Scene,
            Resolution, and Date
        """
        for scene_id_, res_dict_ in tqdm(self.scenes.items()):
            for res_, date_dict_ in tqdm(res_dict_.items()):
                if 'timeseries' not in date_dict_.keys():
                    continue
                if date_dict_['timeseries']['ndvi'].size == 0:
                    warning_message(
                        f'Temporal NDVI does not exist for {scene_id_} at {res_}'
                    )
                    continue

                # Compute K-Means Spatial Clustering per Image
                kmeans_ = kmeans_temporal_cluster(
                    date_dict_['timeseries']['ndvi'],
                    n_clusters=self.n_clusters,
                    quantile_range=self.quantile_range,
                    verbose=self.verbose,
                    verbose_plot=self.verbose_plot,
                    scene_id=scene_id_,
                    res=res_
                )

                # Store the NDVI and masked transform in data struct
                self.scenes[scene_id_][res_]['timeseries']['kmeans'] = kmeans_
