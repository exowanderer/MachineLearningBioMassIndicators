"""Class definitions for mlbmi module"""
import boto3
import geopandas as gpd
import joblib
import numpy as np
import os
import rasterio
import sys


from dataclasses import dataclass, field
from datetime import datetime, timedelta
from rasterio.features import bounds
from satsearch import Search
from tqdm import tqdm
from typing import List

# from mlbmi.utils.base_utils import cog_download_bands

from .utils import (
    compute_ndvi,
    compute_gci,
    compute_rci,
    compute_scl_mask,
    download_cog_subscene,
    download_tile_band,
    geom_to_bounding_box,
    get_prefix_filepath,
    kmeans_spatial_cluster,
    kmeans_temporal_cluster,
    pca_spatial_components,
    pca_temporal_components,
    info_message,
    warning_message,
    debug_message
)


@dataclass
class SentinelAOIParams:
    """Class for SentinelAOI Input Params"""
    geojson: str = 'doeberitzer_multipolygon.geojson'
    band_names: List[str] = field(default_factory=lambda: ['B04', 'B08'])
    collection: str = 'sentinel-s2-l2a-cogs'
    start_date: str = '2020-01-01'
    end_date: str = '2020-02-01'
    cloud_cover: float = 1
    download: bool = True
    verbose: bool = False
    quiet: bool = True


@dataclass
class KMeansBMIParams:
    """Class for KMeansBMI Input Params"""
    geojson: str = 'doeberitzer_multipolygon.geojson'
    band_names: List[str] = field(default_factory=lambda: ['B04', 'B08'])
    collection: str = 'sentinel-s2-l2a-cogs'
    start_date: str = '2020-01-01'
    end_date: str = '2020-02-01'
    cloud_cover: float = 1
    n_sig: int = 10
    download: bool = True
    n_clusters: int = 5
    quantile_range: List[int] = field(default_factory=lambda: [1, 99])
    verbose: bool = False
    verbose_plot: bool = False
    quiet: bool = True


@dataclass
class PCABMIParams:
    """Class for PCABMI Input Params"""
    geojson: str = 'doeberitzer_multipolygon.geojson'
    band_names: List[str] = field(default_factory=lambda: ['B04', 'B08'])
    collection: str = 'sentinel-s2-l2a-cogs'
    start_date: str = '2020-01-01'
    end_date: str = '2020-02-01'
    cloud_cover: float = 1
    n_sig: int = 10
    download: bool = True
    n_components: int = 5
    quantile_range: List[int] = field(default_factory=lambda: [1, 99])
    verbose: bool = False
    verbose_plot: bool = False
    quiet: bool = True


class SentinelAOI:

    def __init__(
            self, geojson: str,
            start_date: str = '2020-01-01',
            end_date: str = '2020-02-01',
            days_back: int = None,
            cloud_cover: int = 1,
            collection: str = 'sentinel-s2-l2a-cogs',
            band_names: list = ['B04', 'B08'],
            no_scl: bool = False,
            download: bool = False,
            verbose: bool = False,
            quiet=False):
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
                Defaults to 'sentinel-s2-l2a-cogs'.
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
        self.days_back = days_back
        self.cloud_cover = cloud_cover
        self.collection = collection
        self.band_names = band_names
        self.no_scl = no_scl
        self.download = download
        self.verbose = verbose
        self.quiet = quiet

        # If start or end dates are not given, then assume "the last week"
        if self.days_back is not None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            end_date = end_date.strftime("%Y-%m-%d")
            start_date = start_date.strftime("%Y-%m-%d")

        if self.quiet:
            # Force all output to be supressed
            # Useful when iterating over instances
            self.verbose = False

        if not self.no_scl:
            self.band_names.append('SCL')

    def search_earth_aws(self):
        """Organize input parameters and call search query to AWS STAC API"""

        # Check if s3_client is properly configured
        assert(self.s3_client is not None), \
            'Please assign and allocate an s3_client'

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

        if self.verbose:
            info_message(
                sys._getframe().f_code.co_name,
                'Search Query Parameters\n',
                url=self.url_earth_search,
                intersects=bounding_box['features'][0]['geometry'],
                datetime=eo_datetime,
                query=eo_query,
                collections=self.collection
            )

        # Use Sat-Search to idenitify and load all meta data from search field
        self.search = Search(
            url=self.url_earth_search,
            intersects=bounding_box['features'][0]['geometry'],
            datetime=eo_datetime,
            query=eo_query,
            collections=[self.collection]
        )

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

    def acquire_cog_images(self, band_names=['red', 'nir']):

        band_names = band_names if self.band_names is None else self.band_names

        for item_ in tqdm(self.items):
            hrefs = {band_: item_.asset(band_)["href"] for band_ in band_names}

            if self.verbose:
                print(
                    f"Latest data found that intersects geometry: {item_.date}"
                )
                for band_, href_ in hrefs.items():
                    print(f"URL {band_.upper()} band: {href_}")

            product_id_ = item_.properties['sentinel:product_id']
            scene_id_ = product_id_.split('_')[5]
            date_ = item_.date.isoformat()
            res_ = 'cog'

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

            # scenes_ = self.scenes[scene_id_][res_][date_]
            # for kgeo_, geometry_ in tqdm(enumerate(self.gdf['geometry'])):
            #     if f"geometry{kgeo_}" not in scenes_.keys():
            #         scenes_[f"geometry{kgeo_}"] = {}
            geometry_ = self.gdf['geometry'][0]
            for band_, geotiff_file in tqdm(hrefs.items()):
                # with rasterio.open(geotiff_file) as cog_fp:
                cog_fp = rasterio.open(geotiff_file)
                # cog_fp.crs = cog_fp.crs.from_epsg("4326")

                band_keys = self.scenes[scene_id_][res_][date_].keys()
                if band_ not in band_keys:
                    # Set band dict are blank dict per band
                    self.scenes[scene_id_][res_][date_][band_] = {}

                bbox = bounds(geometry_)
                subscene = download_cog_subscene(cog_fp, bbox)

                # Load the JP2 file
                self.scenes[scene_id_][res_][date_][band_]['image'] = subscene
                self.scenes[scene_id_][res_][date_][band_]['raster'] = cog_fp

                # self.scenes[scene_id_][res_][date_][band_] = scenes_

    def download_full_image_filepaths(self):
        """Cycle through geoJSON to download files (if download is True)
            and return list of files for later storage
        """

        if not hasattr(self, 'items_geojson'):
            # Call and store STAC Sentinel-2 query
            self.search_earth_aws()

        # Log all filepaths to queried scenes
        if self.download:
            # Loop over GeoJSON Features
            feat_iter = tqdm(
                self.items_geojson['features'], disable=self.quiet
            )
            for feat_ in feat_iter:
                # Loop over GeoJSON Bands
                band_iter = tqdm(self.band_names, disable=self.quiet)

                for bnd_name_ in band_iter:
                    if bnd_name_.upper() not in feat_['assets'].keys():
                        continue
                    print(feat_['assets'][bnd_name_.upper()]['href'])
                    # Download the selected bands
                    _ = download_tile_band(  # filepath_
                        feat_['assets'][bnd_name_.upper()]['href'],
                        s3_client=self.s3_client,
                        collection=self.collection
                    )

        # Loop over GeoJSON Features to Allocate all requested files
        self.filepaths = {}
        for feat_ in self.items_geojson['features']:
            # Loop over GeoJSON Bands
            for bnd_name_ in self.band_names:
                if bnd_name_ not in self.filepaths.keys():
                    # Check if this is the first file per band
                    self.filepaths[bnd_name_] = []

                if bnd_name_.upper() not in feat_['assets'].keys():
                    continue

                # Download the selected bands
                href = feat_['assets'][bnd_name_.upper()]['href']
                _, output_filepath = get_prefix_filepath(
                    href, collection=self.collection
                )
                # Storoe the file name in a per band structure
                self.filepaths[bnd_name_].append(output_filepath)

    def download_and_acquire_full_images(self):
        """Cycle through geoJSON to download files (if download is True)
            and return list of files for later storage
        """

        if not hasattr(self, 'items_geojson'):
            # Call and store STAC Sentinel-2 query
            self.search_earth_aws()

        # Log all filepaths to queried scenes
        if self.download:
            # Loop over GeoJSON Features
            feat_iter = tqdm(
                self.items_geojson['features'], disable=self.quiet
            )
            for feat_ in feat_iter:
                # Loop over GeoJSON Bands
                band_iter = tqdm(self.band_names, disable=self.quiet)
                for bnd_name_ in band_iter:
                    if bnd_name_.upper() not in feat_['assets'].keys():
                        continue

                    print(feat_['assets'][bnd_name_.upper()]['href'])

                    # Download the selected bands
                    _ = download_tile_band(  # filepath_
                        feat_['assets'][bnd_name_.upper()]['href'],
                        s3_client=self.s3_client,
                        collection=self.collection
                    )

        # Loop over GeoJSON Features to Allocate all requested files
        self.filepaths = {}
        for feat_ in self.items_geojson['features']:
            # Loop over GeoJSON Bands
            for bnd_name_ in self.band_names:
                if bnd_name_ not in self.filepaths.keys():
                    # Check if this is the first file per band
                    self.filepaths[bnd_name_] = []

                if bnd_name_.upper() not in feat_['assets'].keys():
                    continue

                # Download the selected bands
                href = feat_['assets'][bnd_name_.upper()]['href']
                _, output_filepath = get_prefix_filepath(
                    href, collection=self.collection
                )
                # Storoe the file name in a per band structure
                self.filepaths[bnd_name_].append(output_filepath)

    def load_data_into_struct(self):
        """Load all files in filepaths inot data structure self.scenes"""

        for bnd_name_, filepaths_ in self.filepaths.items():
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

                if bnd_name_ not in self.scenes[scene_id_][res_][date_].keys():
                    # Set band dict are blank dict per band
                    self.scenes[scene_id_][res_][date_][bnd_name_] = {}

                if self.verbose:
                    info_message(f"{fpath_} :: {os.path.exists(fpath_)}")

                # Load the JP2 file
                raster_ = {}  # Aid to maintain 79 characters per line
                if 'cog' in self.collection:
                    raster_['raster'] = rasterio.open(fpath_)
                else:
                    raster_['raster'] = rasterio.open(
                        fpath_,
                        driver='JP2OpenJPEG'
                    )

                # Store the raster in the self.scenes data structure
                self.scenes[scene_id_][res_][date_][bnd_name_] = raster_

    def old_load_data_into_struct(self):
        """Load all files in filepaths inot data structure self.scenes"""

        for bnd_name_, filepaths_ in self.filepaths.items():
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

                if bnd_name_ not in self.scenes[scene_id_][res_][date_].keys():
                    # Set band dict are blank dict per band
                    self.scenes[scene_id_][res_][date_][bnd_name_] = {}

                if self.verbose:
                    info_message(f"{fpath_} :: {os.path.exists(fpath_)}")

                # Load the JP2 file
                raster_ = {}  # Aid to maintain 79 characters per line
                if 'cog' in self.collection:
                    raster_['raster'] = rasterio.open(fpath_)
                else:
                    raster_['raster'] = rasterio.open(
                        fpath_,
                        driver='JP2OpenJPEG'
                    )

                # Store the raster in the self.scenes data structure
                self.scenes[scene_id_][res_][date_][bnd_name_] = raster_

    def create_scl_mask(self, mask_vals=[0, 1, 9]):  # 2, 3, 7, 8, 10
        # required_bands = ['B04', 'B08']
        # if bmi == 'gci':
        #     required_bands = ['B03', 'B08']
        # if bmi == 'rci':
        #     required_bands = ['B04', 'B08']

        if self.no_scl:
            return

        # Behaviour disable=self.quiet allows user to turn off tqdm via CLI
        scene_iter = tqdm(self.scenes.items(), disable=self.quiet)
        for scene_id_, res_dict_ in scene_iter:
            res_iter = tqdm(res_dict_.items(), disable=self.quiet)
            for res_, date_dict_ in res_iter:
                date_iter = tqdm(date_dict_.items(), disable=self.quiet)
                for date_, band_data_ in date_iter:
                    scl_ = band_data_['SCL'] if 'SCL' in band_data_ else None

                    if scl_ is None:
                        return

                    scl_mask_, mask_transform_ = compute_scl_mask(
                        scl=scl_,
                        mask_vals=mask_vals,
                        gdf=self.gdf,
                        scene_id=scene_id_,
                        res=res_,
                        date=date_,
                        bins=self.hist_bins,
                        verbose=self.verbose,
                        verbose_plot=self.verbose_plot
                    )

                    date_dict_ = self.scenes[scene_id_][res_][date_]
                    date_dict_['scl_mask'] = scl_mask_
                    date_dict_['transform'] = mask_transform_

                    self.scenes[scene_id_][res_][date_] = date_dict_

    def compute_bmi_for_all(self, bmi='ndvi', alpha=0):
        """Cycle over self.scenes and compute BMI for each scene and date_"""
        compute_bmi = compute_ndvi
        required_bands = ['B04', 'B08']
        if bmi == 'gci':
            compute_bmi = compute_gci
            required_bands = ['B03', 'B08']
        if bmi == 'rci':
            compute_bmi = compute_rci
            required_bands = ['B04', 'B08']

        # Behaviour disable=self.quiet allows user to turn off tqdm via CLI
        scene_iter = tqdm(self.scenes.items(), disable=self.quiet)
        for scene_id_, res_dict_ in scene_iter:
            res_iter = tqdm(res_dict_.items(), disable=self.quiet)
            for res_, date_dict_ in res_iter:
                date_iter = tqdm(date_dict_.items(), disable=self.quiet)
                for date_, band_data_ in date_iter:
                    # for band_ in required_bands:
                    #     debug_message(
                    #         sys._getframe().f_code.co_name,
                    #         f"{band_} in {band_data_.keys()}:"
                    #         f"{band_ in band_data_.keys()}"
                    #     )

                    has_bands = np.any([
                        band_ in band_data_.keys()
                        for band_ in required_bands
                    ])
                    if not has_bands:
                        warning_message(
                            f'{bmi.upper()} cannot be computed without '
                            + ' and '.join(required_bands)
                        )
                        continue

                    # TODO all 3 `compute_bmi` functions accept SCL,
                    # but do not know what to do with it
                    # TODO Either have each `compute_bmi` compute
                    # which flags to ignore in a mask or precompute the mask
                    # and send that instead of band_data_['SCL']

                    scl_mask_ = None
                    if not self.no_scl:
                        scl_mask_ = band_data_['scl_mask'] \
                            if 'scl_mask' in band_data_ else None

                    # Compute BMI for individual scene, res, date
                    bmi_masked_, mask_transform_ = compute_bmi(
                        band_data_[required_bands[0]],
                        band_data_[required_bands[1]],
                        scl_mask=scl_mask_,
                        gdf=self.gdf,
                        alpha=alpha,
                        n_sig=self.n_sig,
                        scene_id=scene_id_,
                        res=res_,
                        date=date_,
                        bins=self.hist_bins,
                        verbose=self.verbose,
                        verbose_plot=self.verbose_plot
                    )

                    # Store the BMI and masked transform in data struct
                    # Behaviour below used to maintain 79 chars per line
                    date_dict_ = self.scenes[scene_id_][res_][date_]
                    date_dict_[bmi] = bmi_masked_
                    date_dict_['transform'] = mask_transform_

                    self.scenes[scene_id_][res_][date_] = date_dict_

    def allocate_bmi_timeseries(self, bmi='ndvi'):
        """Allocate BMI images per scene and date inot time series"""
        scene_iter = tqdm(self.scenes.items(), disable=self.quiet)
        for scene_id_, res_dict_ in scene_iter:
            res_iter = tqdm(res_dict_.items(), disable=self.quiet)
            for res_, date_dict_ in res_iter:
                timestamps_ = []  # Create blank list for datetime stampes
                timeseries_ = []  # Create blank list for BMI data
                date_iter = tqdm(date_dict_.items(), disable=self.quiet)
                for date_, dict_ in date_iter:
                    if bmi not in dict_.keys():
                        warning_message(
                            f"{scene_id_} - {res_} - {date_}: " + "\n"
                            f"{bmi} not in date_dict_[{date_}].keys()"
                        )
                        continue

                    if date_ != 'timeseries':
                        # Append this time stamp to timestamps_ list
                        timestamps_.append(datetime.fromisoformat(date_))

                        # Append this BMI to timeseries_ list
                        timeseries_.append(dict_[bmi])

                # Redefine the list of arrays to array of arrays (image cube)
                timeseries_ = np.array(timeseries_)

                # store in 'timeseries' dict inside self.scenes data structure
                # Behaviour below used to maintain 79 characters per line
                timeseries_dict = {}
                timeseries_dict[bmi] = timeseries_
                timeseries_dict['timestamps'] = timestamps_

                self.scenes[scene_id_][res_]['timeseries'] = timeseries_dict
    """
    def compute_ndvi_for_all(self, alpha=0):
        '''Cycle over self.scenes and compute NDVI for each scene and date_'''
        # Behaviour disable=self.quiet allows user to turn off tqdm via CLI
        scene_iter = tqdm(self.scenes.items(), disable=self.quiet)
        for scene_id_, res_dict_ in scene_iter:
            res_iter = tqdm(res_dict_.items(), disable=self.quiet)
            for res_, date_dict_ in res_iter:
                date_iter = tqdm(date_dict_.items(), disable=self.quiet)
                for date_, band_data_ in date_iter:
                    if 'B04' not in band_data_.keys() and \
                            'B08' not in band_data_.keys():
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
                        alpha=alpha,
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

    def compute_gci_for_all(self, alpha=0):
        '''Cycle over self.scenes and compute GCI for each scene and date_'''
        # Behaviour disable=self.quiet allows user to turn off tqdm via CLI
        scene_iter = tqdm(self.scenes.items(), disable=self.quiet)
        for scene_id_, res_dict_ in scene_iter:
            res_iter = tqdm(res_dict_.items(), disable=self.quiet)
            for res_, date_dict_ in res_iter:
                date_iter = tqdm(date_dict_.items(), disable=self.quiet)
                for date_, band_data_ in date_iter:
                    if 'B03' not in band_data_.keys() and \
                            'B08' not in band_data_.keys():
                        warning_message(
                            'GCI cannot be computed without '
                            'both Band03 and Band08'
                        )
                        continue

                    # Compute GCI for individual scene, res, date
                    gci_masked_, mask_transform_ = compute_gci(
                        band_data_['B03'],
                        band_data_['B08'],
                        gdf=self.gdf,
                        alpha=alpha,
                        n_sig=self.n_sig,
                        scene_id=scene_id_,
                        res=res_,
                        date=date_,
                        bins=self.hist_bins,
                        verbose=self.verbose,
                        verbose_plot=self.verbose_plot
                    )

                    # Store the GCI and masked transform in data struct
                    # Behaviour below used to maintain 79 characters per line
                    date_dict_ = self.scenes[scene_id_][res_][date_]
                    date_dict_['gci'] = gci_masked_
                    date_dict_['transform'] = mask_transform_

                    self.scenes[scene_id_][res_][date_] = date_dict_

    def allocate_gci_timeseries(self):
        '''Allocate GCI images per scene and date inot time series'''
        scene_iter = tqdm(self.scenes.items(), disable=self.quiet)
        for scene_id_, res_dict_ in scene_iter:
            res_iter = tqdm(res_dict_.items(), disable=self.quiet)
            for res_, date_dict_ in res_iter:
                timestamps_ = []  # Create blank list for datetime stampes
                timeseries_ = []  # Create blank list for GCI data
                date_iter = tqdm(date_dict_.items(), disable=self.quiet)
                for date_, dict_ in date_iter:
                    if 'gci' not in dict_.keys():
                        warning_message(
                            f"{scene_id_} - {res_} - {date_}: " + "\n"
                            f"'gci' not in date_dict_[{date_}].keys()"
                        )
                        continue

                    if date_ != 'timeseries':
                        # Append this time stamp to timestamps_ list
                        timestamps_.append(datetime.fromisoformat(date_))

                        # Append this GCI to timeseries_ list
                        timeseries_.append(dict_['gci'])

                # Redefine the list of arrays to array of arrays (image cube)
                timeseries_ = np.array(timeseries_)

                # store in 'timeseries' dict inside self.scenes data structure
                # Behaviour below used to maintain 79 characters per line
                timeseries_dict = {}
                timeseries_dict['gci'] = timeseries_
                timeseries_dict['timestamps'] = timestamps_

                self.scenes[scene_id_][res_]['timeseries'] = timeseries_dict

    def compute_rci_for_all(self, alpha=0):
        '''Cycle over self.scenes and compute RCI for each scene and date_'''
        # Behaviour disable=self.quiet allows user to turn off tqdm via CLI
        scene_iter = tqdm(self.scenes.items(), disable=self.quiet)
        for scene_id_, res_dict_ in scene_iter:
            res_iter = tqdm(res_dict_.items(), disable=self.quiet)
            for res_, date_dict_ in res_iter:
                date_iter = tqdm(date_dict_.items(), disable=self.quiet)
                for date_, band_data_ in date_iter:
                    if 'B04' not in band_data_.keys() and \
                            'B08' not in band_data_.keys():
                        warning_message(
                            'RCI cannot be computed without '
                            'both Band04 and Band08'
                        )
                        continue

                    # Compute RCI for individual scene, res, date
                    rci_masked_, mask_transform_ = compute_rci(
                        band_data_['B04'],
                        band_data_['B08'],
                        gdf=self.gdf,
                        alpha=alpha,
                        n_sig=self.n_sig,
                        scene_id=scene_id_,
                        res=res_,
                        date=date_,
                        bins=self.hist_bins,
                        verbose=self.verbose,
                        verbose_plot=self.verbose_plot
                    )

                    # Store the RCI and masked transform in data struct
                    # Behaviour below used to maintain 79 characters per line
                    date_dict_ = self.scenes[scene_id_][res_][date_]
                    date_dict_['rci'] = rci_masked_
                    date_dict_['transform'] = mask_transform_

                    self.scenes[scene_id_][res_][date_] = date_dict_

    def allocate_rci_timeseries(self):
        '''Allocate RCI images per scene and date inot time series'''
        scene_iter = tqdm(self.scenes.items(), disable=self.quiet)
        for scene_id_, res_dict_ in scene_iter:
            res_iter = tqdm(res_dict_.items(), disable=self.quiet)
            for res_, date_dict_ in res_iter:
                timestamps_ = []  # Create blank list for datetime stampes
                timeseries_ = []  # Create blank list for RCI data
                date_iter = tqdm(date_dict_.items(), disable=self.quiet)
                for date_, dict_ in date_iter:
                    if 'rci' not in dict_.keys():
                        warning_message(
                            f"{scene_id_} - {res_} - {date_}: " + "\n"
                            f"'rci' not in date_dict_[{date_}].keys()"
                        )
                        continue

                    if date_ != 'timeseries':
                        # Append this time stamp to timestamps_ list
                        timestamps_.append(datetime.fromisoformat(date_))

                        # Append this RCI to timeseries_ list
                        timeseries_.append(dict_['rci'])

                # Redefine the list of arrays to array of arrays (image cube)
                timeseries_ = np.array(timeseries_)

                # store in 'timeseries' dict inside self.scenes data structure
                # Behaviour below used to maintain 79 characters per line
                timeseries_dict = {}
                timeseries_dict['rci'] = timeseries_
                timeseries_dict['timestamps'] = timestamps_

                self.scenes[scene_id_][res_]['timeseries'] = timeseries_dict
    """

    def save_results(self, save_filename):
        info_message(f'Saving Results to JobLib file: {save_filename}')
        save_dict_ = {}
        for key, val in self.__dict__.items():
            if not hasattr(val, '__call__'):
                save_dict_[key] = val

        joblib.dump(save_dict_, save_filename)

    def load_results(self, load_filename):
        load_dict_ = joblib.load(load_filename)
        for key, val in load_dict_.items():
            if not hasattr(val, '__call__'):
                self.__dict__[key] = val

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
                    for bnd_name_, raster_data_ in band_data_.items():
                        # Default behaviour:
                        #   save input raster_data_ to current instance
                        self.scenes[scene_id_][res_][date_][bnd_name_] = \
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
                    for bnd_name_, _ in band_data_.items():
                        # Default behaviour:
                        #   remove current data if it exists in input instance
                        del self.scenes[scene_id_][res_][date_][bnd_name_]

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


class PCABMI(SentinelAOI):
    """PCABMI sub-class inherits SentinelAOI class and operates PCA """

    def __init__(
            self, geojson: str,
            start_date: str = '2020-01-01',
            end_date: str = '2020-02-01',
            cloud_cover: int = 1,
            collection: str = 'sentinel-s2-l2a-cogs',
            band_names: list = ['B04', 'B08'],
            no_scl: bool = False,
            download: bool = False,
            n_components: int = 5,
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
                Defaults to 'sentinel-s2-l2a-cogs'. [Inherited]
            band_names (list, optional): Sentinel-2 band names.
                Defaults to ['B04', 'B08']. [Inherited]
            download (bool, optional): Flag whether to initate a download
                (costs money). Defaults to False. [Inherited]
            n_components (int, optional): number of comonents to operate PCA.
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
            quiet=quiet,
            no_scl=no_scl
        )
        self.n_components = n_components
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

    def compute_spatial_pca(self, bmi='ndvi', n_components=None):
        """Cycle through all BMI and Compute BMI for each Scene, Resolution,
            and Date

        Args:
            n_components (int, optional): Allow user to override the n_components
                used in K-Means when hyperparameter optimizeding.
                Defaults to None.
        """
        # Allow user to override n_components
        n_components = self.n_components \
            if n_components is None else n_components

        required_bands = ['B04', 'B08']
        if bmi == 'gci':
            required_bands = ['B03', 'B08']
        if bmi == 'rci':
            required_bands = ['B04', 'B08']

        scene_iter = tqdm(self.scenes.items(), disable=self.quiet)
        for scene_id_, res_dict_ in scene_iter:
            res_iter = tqdm(res_dict_.items(), disable=self.quiet)
            for res_, date_dict_ in res_iter:
                date_iter = tqdm(date_dict_.items(), disable=self.quiet)
                for date_, band_data_ in date_iter:
                    if bmi not in self.scenes[scene_id_][res_][date_].keys():
                        warning_message(
                            f'{bmi.upper()} does not exist in scenes for '
                            f'scene_id_:{scene_id_}; res_:{res_}; date_:{date_}'
                        )
                        continue

                    has_bands = np.any([
                        band_ not in band_data_.keys()
                        for band_ in required_bands
                    ])

                    if has_bands:
                        warning_message(
                            f'{bmi.upper()} cannot be computed without '
                            + ' and '.join(required_bands)
                        )
                        continue

                    # Compute K-Means Spatial Clustering per Image
                    pca_ = pca_spatial_components(
                        self.scenes[scene_id_][res_][date_][bmi],
                        n_components=n_components,
                        quantile_range=self.quantile_range,
                        verbose=self.verbose,
                        verbose_plot=self.verbose_plot,
                        scene_id=scene_id_,
                        res=res_,
                        date=date_
                    )

                    # Store the result in the self.scenes data structure
                    # This behaviour is used to maintain 79 characters per line
                    kdict_ = self.scenes[scene_id_][res_][date_]
                    if 'pca' not in kdict_.keys():
                        kdict_['pca'] = {}

                    kdict_['pca'][n_components] = pca_
                    self.scenes[scene_id_][res_][date_] = kdict_

    def compute_temporal_pca(self, bmi='ndvi', n_components=None):
        """Cycle over all BMI time series and Compute BMI for each Scene,
            Resolution, and Date

        Args:
            n_components (int, optional): Allow user to override the n_components
                used in K-Means when hyperparameter optimizeding.
                Defaults to None.
        """
        # Allow user to override n_components
        n_components = self.n_components if n_components is None else n_components

        scene_iter = tqdm(self.scenes.items(), disable=self.quiet)
        for scene_id_, res_dict_ in scene_iter:
            res_iter = tqdm(res_dict_.items(), disable=self.quiet)
            for res_, date_dict_ in res_iter:
                if 'timeseries' not in date_dict_.keys():
                    continue

                if bmi not in date_dict_['timeseries'].keys():
                    warning_message(
                        f'Temporal {bmi} does not exist for '
                        f'{scene_id_} at {res_}'
                    )
                    continue

                if date_dict_['timeseries'][bmi].size == 0:
                    warning_message(
                        f'Temporal {bmi} is empty for {scene_id_} at {res_}'
                    )
                    continue

                # Compute K-Means Spatial Clustering per Image
                pca_ = pca_temporal_components(
                    date_dict_['timeseries'][bmi],
                    n_components=n_components,
                    quantile_range=self.quantile_range,
                    verbose=self.verbose,
                    verbose_plot=self.verbose_plot,
                    scene_id=scene_id_,
                    res=res_
                )

                # Store the result in the self.scenes data structure
                # This behaviour is used to maintaint 79 characters per line
                kdict_ = self.scenes[scene_id_][res_]['timeseries']
                if 'pca' not in kdict_.keys():
                    kdict_['pca'] = {}

                kdict_['pca'][n_components] = pca_
                self.scenes[scene_id_][res_]['timeseries'] = kdict_


class KMeansBMI(SentinelAOI):
    """KMeansBMI sub-class inherits SentinelAOI class and operates KMeans """

    def __init__(
            self, geojson: str,
            start_date: str = '2020-01-01',
            end_date: str = '2020-02-01',
            cloud_cover: int = 1,
            collection: str = 'sentinel-s2-l2a-cogs',
            band_names: list = ['B04', 'B08'],
            no_scl: bool = False,
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
                Defaults to 'sentinel-s2-l2a-cogs'. [Inherited]
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
            quiet=quiet,
            no_scl=no_scl
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

    def compute_spatial_kmeans(self, bmi='ndvi', n_clusters=None):
        """Cycle through all BMI and Compute BMI for each Scene, Resolution,
            and Date

        Args:
            n_clusters (int, optional): Allow user to override the n_clusters
                used in K-Means when hyperparameter optimizeding.
                Defaults to None.
        """
        required_bands = ['B04', 'B08']
        if bmi == 'gci':
            required_bands = ['B03', 'B08']
        if bmi == 'rci':
            required_bands = ['B04', 'B08']

        # Allow user to override n_clusters
        n_clusters = self.n_clusters if n_clusters is None else n_clusters

        scene_iter = tqdm(self.scenes.items(), disable=self.quiet)
        for scene_id_, res_dict_ in scene_iter:
            res_iter = tqdm(res_dict_.items(), disable=self.quiet)
            for res_, date_dict_ in res_iter:
                date_iter = tqdm(date_dict_.items(), disable=self.quiet)
                for date_, band_data_ in date_iter:

                    has_bands = np.any([
                        band_ in band_data_.keys()
                        for band_ in required_bands
                    ])

                    if not has_bands:
                        warning_message(
                            f'{bmi.upper()} cannot be computed without '
                            + ' and '.join(required_bands)
                        )
                        continue

                    # Compute K-Means Spatial Clustering per Image
                    kmeans_ = kmeans_spatial_cluster(
                        self.scenes[scene_id_][res_][date_][bmi],
                        n_clusters=n_clusters,
                        quantile_range=self.quantile_range,
                        verbose=self.verbose,
                        verbose_plot=self.verbose_plot,
                        scene_id=scene_id_,
                        res=res_,
                        date=date_
                    )

                    # Store the result in the self.scenes data structure
                    # This behaviour is used to maintain 79 characters per line
                    kdict_ = self.scenes[scene_id_][res_][date_]
                    if 'kmeans' not in kdict_.keys():
                        kdict_['kmeans'] = {}

                    kdict_['kmeans'][n_clusters] = kmeans_
                    self.scenes[scene_id_][res_][date_] = kdict_

    def compute_temporal_kmeans(self, bmi='ndvi', n_clusters=None):
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

                if bmi in date_dict_['timeseries'].keys():
                    warning_message(
                        f'Temporal {bmi} does not exist for '
                        f'{scene_id_} at {res_}'
                    )
                    continue

                if date_dict_['timeseries'][bmi].size == 0:
                    warning_message(
                        f'Temporal {bmi} is empty for {scene_id_} at {res_}'
                    )
                    continue

                # Compute K-Means Spatial Clustering per Image
                kmeans_ = kmeans_temporal_cluster(
                    date_dict_['timeseries'][bmi],
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
