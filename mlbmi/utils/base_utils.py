"""Functional forms for utilities for mlbmi module"""
import os
import json
import numpy as np
import rasterio
import sys

from datetime import datetime, timedelta
from icecream import ic
from matplotlib import pyplot as plt
from pyproj import Transformer
from rasterio.features import bounds
from rasterio.mask import mask
from rasterio.windows import Window
from satsearch import Search
from statsmodels.robust import scale
from wget import download
# from fiona.crs import from_epsg
# from rasterio import plot
# from rasterio.merge import merge

ic.configureOutput(includeContext=True)


def info_message(*args, end='\n', **kwargs):
    ic.configureOutput(prefix='INFO | ')

    for arg_ in args:
        ic(arg_)

    for arg_, val_ in kwargs.items():
        ic(f"{arg_}: {val_}")


def warning_message(*args, end='\n', **kwargs):
    ic.configureOutput(prefix='WARNING | ')
    for arg_ in args:
        ic(arg_)

    for arg_, val_ in kwargs.items():
        ic(f"{arg_}: {val_}")


def debug_message(*args, end='\n', **kwargs):
    ic.configureOutput(prefix='DEBUG | ')
    for arg_ in args:
        ic(arg_)

    for arg_, val_ in kwargs.items():
        ic(f"{arg_}: {val_}")


def sanity_check_image_statistics(
        image, scene_id, res, date, image_name=None, bins=100, plot_now=False):
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
        f"{image_name} Image: {scene_id} - {res} - {date}",
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
        bottom=0.10,
        top=.90,
        wspace=1e-2
    )

    ax2.set_title(
        f"{image_name} Hist: {scene_id} - {res} - {date}",
        fontsize=20
    )

    if plot_now:
        plt.show()


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


def download_cog_subscene(cog_fp, bbox):
    coord_transformer = Transformer.from_crs("epsg:4326", cog_fp.crs)

    # calculate pixels to be streamed in cog
    coord_upper_left = coord_transformer.transform(bbox[3], bbox[0])
    coord_lower_right = coord_transformer.transform(bbox[1], bbox[2])
    pix_upper_left = cog_fp.index(coord_upper_left[0], coord_upper_left[1])
    pix_lower_right = cog_fp.index(coord_lower_right[0], coord_lower_right[1])

    for pixel in pix_upper_left + pix_lower_right:
        # If the pixel value is below 0, that means that
        # the bounds are not inside of our available dataset.
        if pixel < 0:
            print("Provided geometry extends available datafile.")
            print("Provide a smaller area of interest to get a result.")
            sys.exit()

    # make http range request only for bytes in window
    window = Window.from_slices(
        (pix_upper_left[0], pix_lower_right[0]),
        (pix_upper_left[1], pix_lower_right[1])
    )

    return cog_fp.read(1, window=window)


def cog_download_and_plot_bands(search, geometry):
    # debug_message(geojson)
    # # file_path = "path/to/your/file.geojson"
    # with open(geojson, "r") as fp:
    #     file_content = json.load(fp)
    # debug_message(file_content)
    # geometry = file_content["features"][0]["geometry"]
    # debug_message(geometry)
    # debug_message(start_date, end_date)
    # # only request images with cloudcover less than 20%
    # query = {
    #     "eo:cloud_cover": {
    #         "lt": cloud_cover
    #     }
    # }
    # debug_message(query)
    # search = Search(
    #     url='https://earth-search.aws.element84.com/v0',
    #     intersects=geometry,
    #     datetime=f"{start_date}/{end_date}",
    #     collections=[collection],
    #     query=query
    # )
    debug_message(search)
    # Grab latest red && nir
    items = search.items()
    debug_message(items)
    latest_data = items.dates()[-1]
    debug_message(items[0].asset('scl'))
    # scl = items[0].asset('scl')["href"]
    red = items[0].asset('red')["href"]
    nir = items[0].asset('nir')["href"]
    print(f"Latest data found that intersects geometry: {latest_data}")
    print(f"Url red band: {red}")
    print(f"Url nir band: {nir}")

    for geotiff_file in [red, nir]:  # , scl]:
        with rasterio.open(geotiff_file) as geo_fp:
            bbox = bounds(geometry)
            coord_transformer = Transformer.from_crs(
                "epsg:4326", geo_fp.crs)
            # calculate pixels to be streamed in cog
            coord_upper_left = coord_transformer.transform(
                bbox[3], bbox[0])
            coord_lower_right = coord_transformer.transform(
                bbox[1], bbox[2])
            pixel_upper_left = geo_fp.index(
                coord_upper_left[0],
                coord_upper_left[1]
            )
            pixel_lower_right = geo_fp.index(
                coord_lower_right[0],
                coord_lower_right[1]
            )

            for pixel in pixel_upper_left + pixel_lower_right:
                # If the pixel value is below 0, that means that
                # the bounds are not inside of our available dataset.
                if pixel < 0:
                    print("Provided geometry extends available datafile.")
                    print("Provide a smaller area of interest to get a result.")
                    sys.exit()

            # make http range request only for bytes in window
            window = Window.from_slices(
                (
                    pixel_upper_left[0],
                    pixel_lower_right[0]
                ),
                (
                    pixel_upper_left[1],
                    pixel_lower_right[1]
                )
            )
            subset = geo_fp.read(1, window=window)

            # vizualize
            import matplotlib.pyplot as plt
            plt.imshow(subset, cmap="seismic")
            plt.colorbar()
            plt.show()


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
        band04, band08, gdf, scl_mask=None, alpha=0, n_sig=10, verbose=False,
        verbose_plot=False, scene_id=None, res=None, date=None, bins=100):
    """Compute the NDVI image from band08 and band04 values

    Args:
        band04 (dict): Sentinel-2-L2A Band04 raster data
        band08 (dict): Sentinel-2-L2A Band08 raster data
        gdf (gpd.GeoDataFrame): GeoDataFrame with geometry information
        alpha (float): For alpha > 0, NDVI becomes WDRVI
        n_sig (int, optional): Number is sigma to quality as an outlier.
            Defaults to 10.
        verbose (bool): Toggle to print extra info statements. Default False.
        verbose_plot (bool): Toggle to plot extra figures. Default False.
        scene_id (str): Scene ID for verbose_plot figures. Default None.
        res (str): Resolution for verbose_plot figures. Default None.
        date (str): Date for verbose_plot figures. Default None.
        bins (int): Number of bins for verbose_plot histograms. Default 100.

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
        alpha + band08_masked[0] - band04_masked[0],
        alpha + band08_masked[0] + band04_masked[0]
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
    debug_message(scl_mask.shape, ndvi_masked.shape)
    # If the SCL mas exists, then use it to remove 'bad pixels'
    # if scl_mask is not None:
    #     ndvi_masked[scl_mask] = 0

    if verbose_plot:
        # Show NDVI image and plot the histogram over its values
        sanity_check_image_statistics(
            ndvi_masked, scene_id, res, date, image_name='NDVI', bins=bins
        )

    return ndvi_masked, mask_transform


def compute_gci(
        band03, band08, gdf, scl_mask=None, alpha=0, n_sig=10, verbose=False,
        verbose_plot=False, scene_id=None, res=None, date=None, bins=100):
    """Compute the Green Chlorophyll Index image from band08 and band03 values

    Args:
        band03 (dict): Sentinel-2-L2A Band03 raster data
        band08 (dict): Sentinel-2-L2A Band08 raster data
        gdf (gpd.GeoDataFrame): GeoDataFrame with geometry information
        alpha (float): For alpha > 0, GCI becomes GCI-WDRVI
        n_sig (int, optional): Number is sigma to quality as an outlier.
            Defaults to 10.
        verbose (bool): Toggle to print extra info statements. Default False.
        verbose_plot (bool): Toggle to plot extra figures. Default False.
        scene_id (str): Scene ID for verbose_plot figures. Default None.
        res (str): Resolution for verbose_plot figures. Default None.
        date (str): Date for verbose_plot figures. Default None.
        bins (int): Number of bins for verbose_plot histograms. Default 100.

    Returns:
        tuple (np.array, affine.Affine): GCI image and its related transform
    """
    # Convert from MAD to STD because Using the MAD is
    #   more agnostic to outliers than STD
    mad2std = 1.4826

    # By definition, the CRS is identical across bands
    gdf_crs = gdf.to_crs(
        crs=band03['raster'].crs.data
    )

    # Compute the AOI coordinates from the raster crs data
    coords = get_coords_from_geometry(gdf_crs)

    # Mask Band03 data with AOI coords
    band03_masked, _ = mask(
        dataset=band03['raster'],
        shapes=coords,
        crop=True
    )

    # Mask Band08 data with AOI coords
    band08_masked, mask_transform = mask(
        dataset=band08['raster'],
        shapes=coords,
        crop=True
    )

    # Create GCI from masked Band04 and Band08
    gci_masked = np.true_divide(
        alpha + band08_masked[0], alpha + band03_masked[0]
    )
    gci_masked = gci_masked - 1

    # FIll in missing data (outside mask) as zeros
    gci_masked[np.isnan(gci_masked)] = 0

    # median replacement from n_sigma outlier rejection
    med_gci = np.median(gci_masked.ravel())
    std_gci = scale.mad(gci_masked.ravel()) * mad2std

    # Identify outliers as points outside nsig x std_gci from median
    outliers = abs(gci_masked - med_gci) > n_sig * std_gci

    # Set outliers to median value
    gci_masked[outliers] = med_gci

    # If the SCL mas exists, then use it to remove 'bad pixels'
    if scl_mask is not None:
        gci_masked[scl_mask] = 0

    if verbose_plot:
        # Show GCI image and plot the histogram over its values
        sanity_check_image_statistics(
            gci_masked, scene_id, res, date, image_name='GCI', bins=bins
        )

    return gci_masked, mask_transform


def compute_rci(
        band04, band08, gdf, scl_mask=None, alpha=0, n_sig=10, verbose=False,
        verbose_plot=False, scene_id=None, res=None, date=None, bins=100):
    """Compute the Red Chlorophyll Index image from band08 and band04 values

    Args:
        band04 (dict): Sentinel-2-L2A Band04 raster data
        band08 (dict): Sentinel-2-L2A Band08 raster data
        gdf (gpd.GeoDataFrame): GeoDataFrame with geometry information
        alpha (float): For alpha > 0, RCI becomes RCI-WDRVI
        n_sig (int, optional): Number is sigma to quality as an outlier.
            Defaults to 10.
        verbose (bool): Toggle to print extra info statements. Default False.
        verbose_plot (bool): Toggle to plot extra figures. Default False.
        scene_id (str): Scene ID for verbose_plot figures. Default None.
        res (str): Resolution for verbose_plot figures. Default None.
        date (str): Date for verbose_plot figures. Default None.
        bins (int): Number of bins for verbose_plot histograms. Default 100.

    Returns:
        tuple (np.array, affine.Affine): RCI image and its related transform
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

    # Create RCI from masked Band04 and Band08
    rci_masked = np.true_divide(
        alpha + band08_masked[0], alpha + band04_masked[0]
    )
    rci_masked = rci_masked - 1

    # FIll in missing data (outside mask) as zeros
    rci_masked[np.isnan(rci_masked)] = 0

    # median replacement from n_sigma outlier rejection
    med_rci = np.median(rci_masked.ravel())
    std_rci = scale.mad(rci_masked.ravel()) * mad2std

    # Identify outliers as points outside nsig x std_rci from median
    outliers = abs(rci_masked - med_rci) > n_sig * std_rci

    # Set outliers to median value
    rci_masked[outliers] = med_rci

    # If the SCL mas exists, then use it to remove 'bad pixels'
    if scl_mask is not None:
        rci_masked[scl_mask] = 0

    if verbose_plot:
        # Show RCI image and plot the histogram over its values
        sanity_check_image_statistics(
            rci_masked, scene_id, res, date, image_name='RCI', bins=bins
        )

    return rci_masked, mask_transform


def compute_scl_mask(
        scl, mask_vals, gdf, verbose=False,
        verbose_plot=False, scene_id=None, res=None, date=None, bins=100):
    """Compute the Red Chlorophyll Index image from band08 and band04 values

    Args:
        scl (dict): Sentinel-2-L2A Band04 raster data
        mask_vals (dict): Sentinel-2-L2A Band08 raster data
        gdf (gpd.GeoDataFrame): GeoDataFrame with geometry information
        alpha (float): For alpha > 0, RCI becomes RCI-WDRVI
        n_sig (int, optional): Number is sigma to quality as an outlier.
            Defaults to 10.
        verbose (bool): Toggle to print extra info statements. Default False.
        verbose_plot (bool): Toggle to plot extra figures. Default False.
        scene_id (str): Scene ID for verbose_plot figures. Default None.
        res (str): Resolution for verbose_plot figures. Default None.
        date (str): Date for verbose_plot figures. Default None.
        bins (int): Number of bins for verbose_plot histograms. Default 100.

    Returns:
        tuple (np.array, affine.Affine): RCI image and its related transform
    """

    # Convert from MAD to STD because Using the MAD is
    #   more agnostic to outliers than STD
    # mad2std = 1.4826

    # By definition, the CRS is identical across bands
    gdf_crs = gdf.to_crs(
        crs=scl['raster'].crs.data
    )

    # Compute the AOI coordinates from the raster crs data
    coords = get_coords_from_geometry(gdf_crs)

    # Mask Band04 data with AOI coords
    scl_masked_, mask_transform = mask(
        dataset=scl['raster'],
        shapes=coords,
        crop=True
    )
    scl_masked_ = scl_masked_[0]

    scl_mask = np.ones_like(scl_masked_, dtype=bool)
    for val in mask_vals:
        scl_mask[scl_masked_ == val] = False

    # FIll in missing data (outside mask) as zeros
    scl_mask[np.isnan(scl_mask)] = 0

    # median replacement from n_sigma outlier rejection

    if verbose_plot:
        # Show RCI image and plot the histogram over its values
        sanity_check_image_statistics(
            scl_mask, scene_id, res, date, image_name='SCL', bins=bins
        )

    return scl_mask, mask_transform
