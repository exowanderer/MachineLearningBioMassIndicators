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


class SentinelAOI(object):

    def __init__(
            self, geojson: str,
            start_date: str = '2020-01-01', end_date: str = '2020-02-01',
            cloud_cover: int = 1, collection: str = 'sentinel-s2-l2a',
            band_names: list = ['B04', 'B08'], download: bool = False,
            verbose: bool = False, quiet=False):
        """[summary]

        Args:
            geojson (str): filepath to geojson for AOI
            start_date (str, optional): start date for STAC query.
                Defaults to '2020-01-01'.
            end_date (str, optional): end date for STAC Query.
                Defaults to '2020-02-01'.
            cloud_cover (int, optional): Percent cloud cover maximum.
                Defaults to 1.
            collection (str, optional): S3 bucket collection for STAC_API_URL.
                Defaults to 'sentinel-s2-l2a'.
            band_names (list, optional): Sentinel-2 band names.
                Defaults to ['B04', 'B08'].
            download (bool, optional): Flag whether to initate a download
                (costs money). Defaults to False.
            verbose (bool, optional): Flag whether to output extra print
                statemetns to stdout. Defaults to False.
        """
        self.scenes = {}  # Data structure for JP2 Data
        self.s3_client = boto3.client('s3')  # AWS download client

        self.geojson = geojson
        self.start_date = start_date
        self.end_date = end_date
        self.cloud_cover = cloud_cover
        self.collection = collection
        self.band_names = band_names
        self.download = download
        self.verbose = verbose
        self.quiet = quiet

        if self.quiet:
            # Force all output to be supressed
            # Useful when iterating over instances
            self.verbose = False

    def search_earth_aws(self):
        """Organize input parameters and call search query to AWS STAC API
        """

        # Get Sat-Search URL
        self.url_earth_search = os.environ.get('STAC_API_URL')

        # Set Date time start and stop for query
        eo_datetime = f'{self.start_date}/{self.end_date}'

        # Set cloud cover percentage for query
        eo_query = {
            'eo:cloud_cover': {'lt': self.cloud_cover}
        }

        # Load geojson inot gpd.GeoDataFrame
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
        """Cycle through geoJSON to download files (if download is True)
            and return list of files for later storage
        """
        # Check if s3_client is properly configured
        assert(self.s3_client is not None), \
            'Please assign and allocate an s3_client'

        # Call and store STAC Sentinel-2 query
        self.search_earth_aws()

        if self.verbose:
            info_message(f'Combined search: {self.search.found()} items')

        # Allocate all meta data for acquisition
        self.items = self.search.items()

        if self.verbose:
            info_message(self.items.summary())

        if not self.quiet:
            info_message("Allocating metadata in geoJSON")

        # Store STAC Sentinel-2 query as geoJSON
        self.items_geojson = self.items.geojson()

        # Log all filepaths to queried scenes
        if self.download:
            # Loop over GeoJSON Features
            feat_iter = tqdm(
                self.items_geojson['features'], disable=self.quiet
            )
            for feat_ in feat_iter:
                # Loop over GeoJSON Bands
                band_iter = tqdm(self.band_names, disable=self.quiet)
                for band_name_ in band_iter:
                    # Download the selected bands
                    _ = download_tile_band(  # filepath_
                        feat_['assets'][band_name_.upper()]['href'],
                        s3_client=self.s3_client
                    )

        # Loop over GeoJSON Features to Allocate all requested files
        self.filepaths = {}
        for feat_ in self.items_geojson['features']:
            # Loop over GeoJSON Bands
            for band_name_ in self.band_names:
                if not band_name_ in self.filepaths.keys():
                    # Check if this is the first file per band
                    self.filepaths[band_name_] = []

                # Download the selected bands
                href = feat_['assets'][band_name_.upper()]['href']
                _, output_filepath = get_prefix_filepath(
                    href, collection=self.collection
                )
                # Storoe the file name in a per band structure
                self.filepaths[band_name_].append(output_filepath)

    def load_data_into_struct(self):
        """Load all files in filepaths inot data structure self.scenes
        """

        for band_name_, filepaths_ in self.filepaths.items():
            # loop over band names
            for fpath_ in filepaths_:
                # loop over file paths
                if not os.path.exists(fpath_):
                    # If a file does not exist, then skip it
                    warning_message(
                        f"File Does not exist {fpath_}"
                        "\n Suggest using --download to acquire it "
                        "(costs money)"
                    )
                    continue

                # Use filepath to identify scene_id, res, and date
                _, scene_id_, res_, date_, _ = fpath_.split('/')

                # Adjust month from 1 to 2 digits if necessary
                year_, month_, day_ = date_.split('-')
                month_ = f"{month_:0>2}"  # datetime.datetime requires this
                day_ = f"{day_:0>2}"  # datetime.datetime requires this
                date_ = f"{year_}-{month_}-{day_}"

                # Build up data structure for easier access later
                if scene_id_ not in self.scenes.keys():
                    # Set scenes are blank dict per scene
                    self.scenes[scene_id_] = {}
                if res_ not in self.scenes[scene_id_].keys():
                    # Set resolutions dict are blank dict per resolution
                    self.scenes[scene_id_][res_] = {}
                if date_ not in self.scenes[scene_id_][res_].keys():
                    # Set dates dict are blank dict per date
                    self.scenes[scene_id_][res_][date_] = {}
                if band_name_ not in self.scenes[scene_id_][res_][date_].keys():
                    # Set band dict are blank dict per band
                    self.scenes[scene_id_][res_][date_][band_name_] = {}

                if self.verbose:
                    info_message(f"{fpath_} :: {os.path.exists(fpath_)}")

                # Load the JP2 file
                raster_ = {}  # Aid to maintain 79 characters per line
                raster_['raster'] = rasterio.open(fpath_, driver='JP2OpenJPEG')

                # Store the raster in the self.scenes data structure
                self.scenes[scene_id_][res_][date_][band_name_] = raster_

    def __add__(self, instance):
        """Concatenate to this SentinelAOI instance the data from a second
            SentinelAOI instance

        Args:
            scenes (SentinelAOI): SentinelAOI instance to be concatenated
        """
        for scene_id_, res_dict_ in instance.scenes.items():
            if not isinstance(res_dict_, dict):
                # Corner case: if res_dict_ is not a dict
                self.scenes[scene_id_] = res_dict_
                continue
            for res_, date_dict_ in res_dict_.items():
                if not isinstance(date_dict_, dict):
                    # Corner case: if date_dict_ is not a dict
                    self.scenes[scene_id_][res_] = date_dict_
                    continue
                for date_, band_data_ in date_dict_.items():
                    if not isinstance(band_data_, dict):
                        # Corner case: if band_data_ is not a dict
                        self.scenes[scene_id_][res_][date_] = band_data_
                        continue
                    for band_name_, raster_data_ in band_data_.items():
                        # Default behaviour:
                        #   save input instance raster_data_ to current instance
                        self.scenes[scene_id_][res_][date_][band_name_] = \
                            raster_data_

    def __sub__(self, instance):
        """Remove the contents of this SentinelAOI instance that correspond to
            a second input SentinelAOI instance

        Args:
            scenes (SentinelAOI): SentinelAOI instance to be desequenced
        """
        for scene_id_, res_dict_ in instance.scenes.items():
            if not isinstance(res_dict_, dict):
                # Corner case: if res_dict_ is not a dict
                del self.scenes[scene_id_]
                continue
            for res_, date_dict_ in res_dict_.items():
                if not isinstance(date_dict_, dict):
                    # Corner case: if date_dict_ is not a dict
                    del self.scenes[scene_id_][res_]
                    continue
                for date_, band_data_ in date_dict_.items():
                    if not isinstance(band_data_, dict):
                        # Corner case: if band_data_ is not a dict
                        del self.scenes[scene_id_][res_][date_]
                        continue
                    for band_name_, _ in band_data_.items():
                        # Default behaviour:
                        #   remove current data if it exists in input instance
                        del self.scenes[scene_id_][res_][date_][band_name_]

    def __repr__(self):
        return "\n".join([
            "SentinelAOI: ",
            f"{'AOI: ':>17}{self.geojson}",
            f"{'Start Date: ':>17}{self.start_date}",
            f"{'End Date: ':>17}{self.end_date}",
            f"{'Cloud Cover Max: ':>17}{self.cloud_cover}",
            f"{'Collection: ':>17}{self.collection}",
            f"{'Band Names: ':>17}{self.band_names}",
        ])

    def __str__(self):
        return self.__repr__()


class KMeansNDVI(SentinelAOI):
    """Class to contain STAC Sentinel-2 Data Structure
    """

    def __init__(
            self, geojson: str,
            start_date: str = '2020-01-01',
            end_date: str = '2020-02-01',
            cloud_cover: int = 1,
            collection: str = 'sentinel-s2-l2a',
            band_names: list = ['B04', 'B08'],
            download: bool = False,
            n_clusters: int = 5,
            n_sig: int = 10,
            quantile_range: list = [1, 99],
            verbose: bool = False,
            verbose_plot: bool = False,
            hist_bins: int = 100,
            quiet: bool = False
    ):
        """[summary]

        Args:
            geojson (str): filepath to geojson for AOI [Inherited]
            start_date (str, optional): start date for STAC query.
                Defaults to '2020-01-01'. [Inherited]
            end_date (str, optional): end date for STAC Query.
                Defaults to '2020-02-01'. [Inherited]
            cloud_cover (int, optional): Percent cloud cover maximum.
                Defaults to 1. [Inherited]
            collection (str, optional): S3 bucket collection for STAC_API_URL.
                Defaults to 'sentinel-s2-l2a'. [Inherited]
            band_names (list, optional): Sentinel-2 band names.
                Defaults to ['B04', 'B08']. [Inherited]
            download (bool, optional): Flag whether to initate a download
                (costs money). Defaults to False. [Inherited]
            n_clusters (int, optional): number of clusters to operate K-Means.
                Defaults to 5.
            n_sig (int, optional): Number of sigma to flag outliers.
                Defaults to 10.
            quantile_range (list, optional): Range of distribution to
                RobustScale. Defaults to [1, 99].
            verbose (bool, optional): Flag whether to output extra print
                statemetns to stdout. Defaults to False. [Inherited]
            verbose_plot (bool, optional): Flag whether to display extra
                matplotlib figures. Defaults to False.
            hist_bins (int, optional): number of bins in matplotlib plt.hist.
                Defaults to 100.
            quiet (bool, optional): Flag to turn off all text and visual output
        """
        super().__init__(
            geojson=geojson,
            start_date=start_date,
            end_date=end_date,
            cloud_cover=cloud_cover,
            collection=collection,
            band_names=band_names,
            download=download,
            verbose=verbose,
            quiet=quiet
        )
        self.n_clusters = n_clusters
        self.n_sig = n_sig
        self.quantile_range = quantile_range
        self.verbose = verbose
        self.verbose_plot = verbose_plot
        self.hist_bins = hist_bins
        self.quiet = quiet

        if self.quiet:
            # Force all output to be supressed
            # Useful when iterating over instances
            self.verbose = False
            self.verbose_plot = False

    def compute_ndvi_for_all(self):
        """Cycle over self.scenes and compute NDVI for each scene and date_
        """
        # Behaviour disable=self.quiet allows user to turn off tqdm via CLI
        scene_iter = tqdm(self.scenes.items(), disable=self.quiet)
        for scene_id_, res_dict_ in scene_iter:
            res_iter = tqdm(res_dict_.items(), disable=self.quiet)
            for res_, date_dict_ in res_iter:
                date_iter = tqdm(date_dict_.items(), disable=self.quiet)
                for date_, band_data_ in date_iter:
                    if not 'B04' in band_data_.keys() and \
                            not 'B08' in band_data_.keys():
                        warning_message(
                            'NDVI cannot be computed without '
                            'both Band04 and Band08'
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
                    # Behaviour below used to maintain 79 characters per line
                    date_dict_ = self.scenes[scene_id_][res_][date_]
                    date_dict_['ndvi'] = ndvi_masked_
                    date_dict_['transform'] = mask_transform_

                    self.scenes[scene_id_][res_][date_] = date_dict_

    def allocate_ndvi_timeseries(self):
        """Allocate NDVI images per scene and date inot time series
        """
        scene_iter = tqdm(self.scenes.items(), disable=self.quiet)
        for scene_id_, res_dict_ in scene_iter:
            res_iter = tqdm(res_dict_.items(), disable=self.quiet)
            for res_, date_dict_ in res_iter:
                timestamps_ = []  # Create blank list for datetime stampes
                timeseries_ = []  # Create blank list for NDVI data
                date_iter = tqdm(date_dict_.items(), disable=self.quiet)
                for date_, dict_ in date_iter:
                    if 'ndvi' not in dict_.keys():
                        warning_message(
                            f"{scene_id_} - {res_} - {date_}: " + "\n"
                            f"'ndvi' not in date_dict_[{date_}].keys()"
                        )
                        continue

                    if date_ != 'timeseries':
                        # Append this time stamp to timestamps_ list
                        timestamps_.append(datetime.fromisoformat(date_))

                        # Append this NDVI to timeseries_ list
                        timeseries_.append(dict_['ndvi'])

                # Redefine the list of arrays to an array of arrays (image cube)
                timeseries_ = np.array(timeseries_)

                # store in 'timeseries' dict inside self.scenes data structure
                # Behaviour below used to maintain 79 characters per line
                timeseries_dict = {}
                timeseries_dict['ndvi'] = timeseries_
                timeseries_dict['timestamps'] = timestamps_

                self.scenes[scene_id_][res_]['timeseries'] = timeseries_dict

    def compute_spatial_kmeans(self, n_clusters=None):
        """Cycle through all NDVI and Compute NDVI for each Scene, Resolution,
            and Date

        Args:
            n_clusters (int, optional): Allow user to override the n_clusters
                used in K-Means when hyperparameter optimizeding.
                Defaults to None.
        """
        # Allow user to override n_clusters
        n_clusters = self.n_clusters if n_clusters is None else n_clusters

        scene_iter = tqdm(self.scenes.items(), disable=self.quiet)
        for scene_id_, res_dict_ in scene_iter:
            res_iter = tqdm(res_dict_.items(), disable=self.quiet)
            for res_, date_dict_ in res_iter:
                date_iter = tqdm(date_dict_.items(), disable=self.quiet)
                for date_, band_data_ in date_iter:
                    if not 'B04' in band_data_.keys() and \
                            not 'B08' in band_data_.keys():
                        warning_message(
                            'NDVI cannot be computed without both '
                            'Band04 and Band08'
                        )
                        continue

                    # Compute K-Means Spatial Clustering per Image
                    kmeans_ = kmeans_spatial_cluster(
                        self.scenes[scene_id_][res_][date_]['ndvi'],
                        n_clusters=n_clusters,
                        quantile_range=self.quantile_range,
                        verbose=self.verbose,
                        verbose_plot=self.verbose_plot,
                        scene_id=scene_id_,
                        res=res_,
                        date=date_
                    )

                    # Store the result in the self.scenes data structure
                    # This behaviour is used to maintaint 79 characters per line
                    kdict_ = self.scenes[scene_id_][res_][date_]
                    if 'kmeans' not in kdict_.keys():
                        kdict_['kmeans'] = {}

                    kdict_['kmeans'][n_clusters] = kmeans_
                    self.scenes[scene_id_][res_][date_] = kdict_

    def compute_temporal_kmeans(self, n_clusters=None):
        """Cycle over all NDVI time series and Compute NDVI for each Scene,
            Resolution, and Date

        Args:
            n_clusters (int, optional): Allow user to override the n_clusters
                used in K-Means when hyperparameter optimizeding.
                Defaults to None.
        """
        # Allow user to override n_clusters
        n_clusters = self.n_clusters if n_clusters is None else n_clusters

        scene_iter = tqdm(self.scenes.items(), disable=self.quiet)
        for scene_id_, res_dict_ in scene_iter:
            res_iter = tqdm(res_dict_.items(), disable=self.quiet)
            for res_, date_dict_ in res_iter:
                if 'timeseries' not in date_dict_.keys():
                    continue
                if date_dict_['timeseries']['ndvi'].size == 0:
                    warning_message(
                        f'Temporal NDVI does not exist for '
                        f'{scene_id_} at {res_}'
                    )
                    continue

                # Compute K-Means Spatial Clustering per Image
                kmeans_ = kmeans_temporal_cluster(
                    date_dict_['timeseries']['ndvi'],
                    n_clusters=n_clusters,
                    quantile_range=self.quantile_range,
                    verbose=self.verbose,
                    verbose_plot=self.verbose_plot,
                    scene_id=scene_id_,
                    res=res_
                )

                # Store the result in the self.scenes data structure
                # This behaviour is used to maintaint 79 characters per line
                kdict_ = self.scenes[scene_id_][res_]['timeseries']
                if 'kmeans' not in kdict_.keys():
                    kdict_['kmeans'] = {}

                kdict_['kmeans'][n_clusters] = kmeans_
                self.scenes[scene_id_][res_]['timeseries'] = kdict_
