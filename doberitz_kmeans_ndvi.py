# # from logging import warn
# from logging import debug, warning
# import boto3
# import geopandas as gpd
# import json
# import numpy as np
# import os
# import rasterio

from argparse import ArgumentParser
# from datetime import datetime
from dotenv import load_dotenv
# from matplotlib import pyplot as plt
# from tqdm import tqdm

# from sklearn.cluster import KMeans, MiniBatchKMeans
# from sklearn.preprocessing import RobustScaler, MinMaxScaler
# from sklearn.model_selection import train_test_split
# from statsmodels.robust import scale

# # from fiona.crs import from_epsg
# # from rasterio import plot
# # from rasterio.merge import merge
# from rasterio.mask import mask
# # from shapely.geometry import box
# from satsearch import Search
# # from satsearch.search import SatSearchError

# from kmeans_ndvi import *

# TODO: change from utils to .utils when modularizing
from kmeans_ndvi import (
    KMeansNDVI,
    info_message,
    warning_message,
    debug_message
)


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
    args.add_argument('--env_filename', type=str, default='.env')
    args.add_argument('--n_clusters', type=int, default=5)
    args.add_argument('--quantile_range', nargs='+', default=(1, 99))
    args.add_argument('--verbose', action='store_true')
    args.add_argument('--verbose_plot', action='store_true')
    clargs = args.parse_args()

    load_dotenv(clargs.env_filename)

    info_message("Generate JP2 KMeansNDVI Instance")
    jp2_data = KMeansNDVI(
        geojson=clargs.geojson,
        start_date=clargs.start_date,
        end_date=clargs.end_date,
        cloud_cover=clargs.cloud_cover,
        collection=clargs.collection,
        band_names=[band_name_.upper() for band_name_ in clargs.band_names],
        download=clargs.download,
        n_clusters=clargs.n_clusters,
        n_sig=clargs.n_sig,
        quantile_range=clargs.quantile_range,
        verbose=clargs.verbose,
        verbose_plot=clargs.verbose_plot,
    )

    info_message("Downloading and acquiring images")
    jp2_data.download_and_acquire_images()

    info_message("Loading JP2 files into data structure")
    jp2_data.load_data_into_struct()

    info_message("Computing NDVI for all scenes")
    jp2_data.compute_ndvi_for_all()

    info_message("Allocating NDVI time series")
    jp2_data.allocate_ndvi_timeseries()

    info_message("Computing spatial K-Means for each scene NDVI")
    jp2_data.compute_spatial_kmeans()

    info_message("Computing temporal K-Means for each scene NDVIs over time")
    jp2_data.compute_temporal_kmeans()
