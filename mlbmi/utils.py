"""Functional forms for utilities for mlbmi module"""
import os
import json
import numpy as np

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from statsmodels.robust import scale
# from fiona.crs import from_epsg
# from rasterio import plot
# from rasterio.merge import merge
from rasterio.mask import mask

# from shapely.geometry import box
from icecream import ic

# import the wget module
from wget import download


ic.configureOutput(includeContext=True)


def info_message(message, end='\n', *args, **kwargs):
    ic.configureOutput(prefix='INFO | ')
    ic(message)


def warning_message(message, end='\n', *args, **kwargs):
    ic.configureOutput(prefix='WARNING | ')
    ic(message)


def debug_message(message, end='\n', *args, **kwargs):
    ic.configureOutput(prefix='DEBUG | ')
    ic(message)


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
        dict: GeoJSON input with only the bounding box
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


def get_prefix_filepath(href, collection='sentinel-s2-l2a-cogs'):
    """Generate prefix and filepath for boto3 s3 download request.

    Args:
        href (str): s3 url for single scene from satsearch service
        collection (str, optional): AWS bucket from which to grab jp2 files.
            Defaults to 'sentinel-s2-l2a-cogs'.

    Returns:
        tuple (str, str): `prefix` and `output_filepath` for boto3
    """

    # Isolate the boto3 S3 bucket prefix from the href url
    if 'cogs' in href:
        prefix = href.split(collection + '/')[1]
        prefix = f'tiles/{prefix}'

        date_ = prefix.split('/')[6].split('_')[2]
        year_ = date_[:4]
        month_ = date_[4:6]
        day_ = date_[6:]

        # Isolate the output file path from the prefix
        dir_resolution = 'R10m'
        dir_sector = ''.join(prefix.split('/')[1:4])
        dir_date = f'{year_}-{month_}-{day_}'
    else:
        prefix = href.replace(f's3://{collection}/', '')

        # Isolate the output file path from the prefix
        dir_resolution = prefix.split('/')[8]
        dir_sector = ''.join(prefix.split('/')[1:4])
        dir_date = '-'.join(prefix.split('/')[4:7])

    # Isolate the jp2 filename from the prefix
    filename = prefix[prefix.rfind('/') + 1:]

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


def download_tile_band(href, collection='sentinel-s2-l2a-cogs', s3_client=None):
    """Download a specific S3 file URL

    Args:
        href (str): S3 file URL
        collection (str, optional): Earth-AWS collection.
            Defaults to 'sentinel-s2-l2a-cogs'.
    """
    assert(s3_client is not None), 'assign s3_client in SentinelAOI instance'

    # Use the href to form the boto3 S3 prefix and output file path
    prefix, output_filepath = get_prefix_filepath(
        href,
        collection=collection
    )

    # Check if file already exists to skip double downloading
    if not os.path.exists(output_filepath):
        if 'cogs' in collection:
            download(url=href, out=output_filepath)
        else:
            # Download it to current directory
            s3_client.download_file(
                collection,
                prefix,
                output_filepath,
                {'RequestPayer': 'requester'}
            )

    return output_filepath


def get_coords_from_geometry(gdf):
    """Function to parse features from GeoDataFrame in such a manner
        that rasterio wants them

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


def compute_ndvi(
        band04, band08, gdf, n_sig=10, verbose=False, verbose_plot=False,
        scene_id=None, res=None, date=None, bins=100):
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
    gdf_crs = gdf.to_crs(
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

    # Identify outliers as points outside nsig x std_ndvi from median
    outliers = abs(ndvi_masked - med_ndvi) > n_sig * std_ndvi

    # Set outliers to median value
    ndvi_masked[outliers] = med_ndvi

    if verbose_plot:
        # Show NDVI image and plot the histogram over its values
        sanity_check_ndvi_statistics(
            ndvi_masked, scene_id, res, date, bins=bins
        )

    return ndvi_masked, mask_transform


def kmeans_spatial_cluster(
        image, n_clusters=5, quantile_range=(1, 99),
        verbose=False, verbose_plot=False,
        scene_id=None, res=None, date=None):
    """Compute kmeans clustering spatially over image as grey scale levels

    Args:
        image (np.array): input greayscale image
        n_clusters (int, optional): number of grey scales. Defaults to 5.
        quantile_range (tuple, optional): RobustScaler outlier rejecton
            threshold. Defaults to (1, 99).

    Returns:
        sklearn.cluster._kmeans.KMeans: trained kmeans clustering object
    """
    # Scale the input image data after reformatting 2D image as 1D vector and
    #   rejecting the outliers using the RobustScaler algorithm
    sclr = RobustScaler(quantile_range=quantile_range)
    pixel_scaled = sclr.fit_transform(image.reshape(-1, 1))

    # Configure kmeans clustering instance
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        max_iter=300,
        tol=0.0001,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm='auto',
    )

    # Compute the K-Means clusters and store in object
    kmeans.fit(pixel_scaled)

    if verbose_plot:
        # Show NDVI cluster images
        sanity_check_spatial_kmeans(
            kmeans, image, quantile_range=quantile_range,
            scene_id=scene_id, res=res, date=date
        )

    return kmeans


def kmeans_temporal_cluster(
        image_stack, n_clusters=5, quantile_range=(1, 99),
        verbose=False, verbose_plot=False,
        scene_id=None, res=None):
    """Compute kmeans clustering spatially over image as grey scale levels

    Args:
        image (np.array): input greayscale image
        n_clusters (int, optional): number of grey scales. Defaults to 5.
        quantile_range (tuple, optional): RobustScaler outlier rejecton
            threshold. Defaults to (1, 99).

    Returns:
        sklearn.cluster._kmeans.KMeans: trained kmeans clustering object
    """

    # Preprocess image data into a sequence of (nonzero) pixels over time
    samples_ = image_stack.reshape(image_stack.shape[0], -1).T
    where_zero = samples_.sum(axis=1) == 0
    samples_ = samples_[~where_zero]

    # Scale the input image data after reformatting 2D image as 1D vector and
    #   rejecting the outliers using the RobustScaler algorithm
    sclr = RobustScaler(quantile_range=quantile_range)
    samples_scaled = sclr.fit_transform(samples_)

    # Configure kmeans clustering instance
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        max_iter=300,
        tol=0.0001,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm='auto',
    )

    # Compute the K-Means clusters and store in object
    kmeans.fit(samples_scaled)

    if verbose_plot:
        # Show NDVI cluster images
        sanity_check_temporal_kmeans(
            kmeans, image_stack, quantile_range=quantile_range,
            scene_id=scene_id, res=res
        )

    return kmeans


def sanity_check_ndvi_statistics(
        image, scene_id, res, date, bins=100, plot_now=False):
    """Plot imshow and hist over image

    Args:
        image (np.arra): iamge with which to visual
        scene_id (str): Sentinel-2A L2A scene ID
        res (str): Sentinel-2A L2A resolution
        date (str): Sentinel-2A L2A acquistion datetime
        bins (int, optional): Number of bins for histogram. Defaults to 100.
    """

    # Sanity Check with imshow
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))
    ax1.imshow(image, interpolation='None')
    ax1.set_title(
        f"NDVI Image: {scene_id} - {res} - {date}",
        fontsize=20
    )

    # Remove all unnecessary markers from figure
    ax1.grid(False)  # remove grid for images
    ax1.xaxis.set_ticks([])  # remove xticks
    ax1.yaxis.set_ticks([])  # remove xticks

    # Sanity Check with visual histogram
    ax2.hist(image.ravel()[(image.ravel() != 0)], bins=bins)
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    plt.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=.90,
        wspace=1e-2
    )

    ax2.set_title(
        f"NDVI Hist: {scene_id} - {res} - {date}",
        fontsize=20
    )

    if plot_now:
        plt.show()


def sanity_check_spatial_kmeans(kmeans, image, quantile_range=(1, 99),
                                scene_id=None, res=None, date=None,
                                plot_now=False):
    """Plot imshow of clustering solution as sanity check

    Args:
        kmeans (sklearn.cluster._kmeans.kmeans): object storing kmeans solution
        image (np.array): image with which kmeans was trains
        quantile_range (tuple, optional): RobustScaler outlier rejecton
            threshold. Defaults to (1, 99).
    """
    # Preprocess image data
    sclr = RobustScaler(quantile_range=quantile_range)
    pixel_scaled = sclr.fit_transform(image.reshape(-1, 1))

    # Predict each cluster value per pixel
    cluster_pred = kmeans.predict(pixel_scaled)

    base_fig_size = 5  # Each sub figure will be base_fig_size x base_fig_size
    fig, axs = plt.subplots(
        ncols=kmeans.n_clusters + 1,
        figsize=(base_fig_size * (kmeans.n_clusters + 1), base_fig_size)
    )

    # Plot the entire cluster_pred image
    axs[0].imshow(cluster_pred.reshape(image.shape), interpolation='None')

    # Cycle through and plot each cluster_pred image per 'class'
    for k in range(kmeans.n_clusters):
        axs[k + 1].imshow(
            (cluster_pred == k).reshape(image.shape),
            interpolation='None'
        )

    # Remove all unnecessary markers from figure
    [ax.grid(False) for ax in axs.ravel()]  # remove grid for images
    [ax.xaxis.set_ticks([]) for ax in axs.ravel()]  # remove xticks
    [ax.yaxis.set_ticks([]) for ax in axs.ravel()]  # remove xticks

    # Adjust figure to maximize use of gui box
    plt.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=0.9,
        wspace=1e-2
    )

    # Set title for entire figure
    fig.suptitle(
        f"Spatial K-Means Reconstruction: {scene_id} - {res} - {date}",
        fontsize=20
    )

    if plot_now:
        # User can override default behaviour and plot on-the-fly
        plt.show()


def sanity_check_temporal_kmeans(
        kmeans, image_stack, quantile_range=(1, 99),
        scene_id=None, res=None, plot_now=False):
    """Plot imshow of clustering solution as sanity check

    Args:
        kmeans (sklearn.cluster._kmeans.kmeans): object storing kmeans solution
        image_stack (np.array): image_stack with which kmeans was trains
        quantile_range (tuple, optional): RobustScaler outlier rejecton
            threshold. Defaults to (1, 99).
    """
    samples_ = image_stack.reshape(image_stack.shape[0], -1).T
    where_zero = samples_.sum(axis=1) == 0
    samples_notzero = samples_[~where_zero]

    # Scale the input image data after reformatting 2D image as 1D vector and
    #   rejecting the outliers using the RobustScaler algorithm
    sclr = RobustScaler(quantile_range=quantile_range)
    samples_scaled = sclr.fit_transform(samples_notzero)

    # Predict each cluster value per pixel
    cluster_pred = kmeans.predict(samples_scaled)

    # Embedd above image in a zero array to re-constitute zeros
    #   for the out of mask shape
    cluster_image = np.zeros(samples_.shape[0])

    # Add one to each Class to represent the "out of mask" is class zero
    cluster_image[~where_zero] = cluster_pred + 1

    # Reshape 1D array into 2D image of the original image shape
    img_shape = image_stack.shape[1:]
    cluster_image = cluster_image.reshape(img_shape)

    base_fig_size = 5  # Each sub figure will be base_fig_size x base_fig_size
    fig, axs = plt.subplots(
        ncols=kmeans.n_clusters + 2,
        figsize=(base_fig_size * (kmeans.n_clusters + 1), base_fig_size)
    )

    # Plot the entire cluster_pred image
    axs[0].imshow(cluster_image, interpolation='None')

    # Plot the pixels outside the mask, which were not clustered
    axs[1].imshow(cluster_image == 0, interpolation='None')

    # Cycle through and plot each cluster_pred image per 'class'
    for k in range(kmeans.n_clusters):
        axs[k + 2].imshow((cluster_image == (k + 1)), interpolation='None')

    # Remove all unnecessary markers from figure
    [ax.grid(False) for ax in axs.ravel()]  # remove grid for images
    [ax.xaxis.set_ticks([]) for ax in axs.ravel()]  # remove xticks
    [ax.yaxis.set_ticks([]) for ax in axs.ravel()]  # remove xticks

    # Adjust figure to maximize use of gui box
    plt.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=0.9,
        wspace=1e-2
    )

    # Set title for entire figure
    fig.suptitle(
        f"Temporal K-Means Reconstruction: {scene_id} - {res}",
        fontsize=20
    )

    if plot_now:
        # User can override default behaviour and plot on-the-fly
        plt.show()
