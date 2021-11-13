"""Run File for Operating KMeans BMI with CLI"""
from argparse import ArgumentParser
from dotenv import load_dotenv
from matplotlib import pyplot as plt

from mlbmi import (
    KMeansBMI,
    info_message,
    warning_message,
    debug_message
)


if __name__ == "__main__":
    EXAMPLE_USAGE = """
    Use case (bash):

    python doeberitzer_kmeans_bmi.py \
        --band_names b03 b04 b08\
        --start_date 2020-01-01 \
        --end_date 2020-02-01 \
        --cloud_cover 1 \
        --download\
        --verbose\
        --verbose_plot


    To access the (paid for) JP2 files instead. Use

    python doeberitzer_kmeans_bmi.py \
        --band_names b03 b04 b08\
        --start_date 2020-01-01 \
        --end_date 2020-02-01 \
        --cloud_cover 1 \
        --download\
        --verbose\
        --verbose_plot\
        --collection sentinel-s2-l2a
    """

    args = ArgumentParser(prog="Doeberitzer K-Means BMI")
    args.add_argument(
        "--geojson",
        type=str,
        default="doeberitzer_multipolygon.geojson"
    )
    args.add_argument("--scene_id", type=str)
    args.add_argument("--band_names", nargs="+", default=["B03", "B04", "B08"])
    args.add_argument("--ndvi", action="store_true")
    args.add_argument("--gci", action="store_true")
    args.add_argument("--rci", action="store_true")
    args.add_argument("--alpha", type=float, default=0)
    args.add_argument("--collection", type=str, default="sentinel-s2-l2a-cogs")
    args.add_argument("--start_date", type=str, default="2020-01-01")
    args.add_argument("--end_date", type=str, default="2020-02-01")
    args.add_argument("--cloud_cover", type=int, default=1)
    args.add_argument("--n_sig", type=float, default=10)
    args.add_argument("--download", action="store_true")
    args.add_argument("--env_filename", type=str, default=".env")
    args.add_argument("--n_clusters", type=int, default=5)
    args.add_argument("--quantile_range", nargs="+", default=(1, 99))
    args.add_argument("--verbose", action="store_true")
    args.add_argument("--verbose_plot", action="store_true")
    args.add_argument("--quiet", action="store_true")
    clargs = args.parse_args()

    load_dotenv(clargs.env_filename)

    info_message("Generate JP2 KMeansBMI Instance")
    jp2_data = KMeansBMI(
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
        quiet=clargs.quiet,
    )

    info_message(jp2_data)

    info_message("Downloading and acquiring images")
    jp2_data.download_and_acquire_images()

    info_message("Loading JP2 files into data structure")
    jp2_data.load_data_into_struct()

    if clargs.ndvi:
        info_message("Computing NDVI for all scenes")
        jp2_data.compute_bmi_for_all(bmi='ndvi')

        info_message("Allocating NDVI time series")
        jp2_data.allocate_bmi_timeseries(bmi='ndvi')

        info_message("Computing spatial K-Means for each scene NDVI")
        jp2_data.compute_spatial_kmeans(bmi='ndvi')

        info_message(
            "Computing temporal K-Means for each scene NDVIs over time"
        )
        jp2_data.compute_temporal_kmeans(bmi='ndvi')

    if clargs.gci:
        info_message("Computing GCI for all scenes")
        jp2_data.compute_bmi_for_all(bmi='gci')

        info_message("Allocating GCI time series")
        jp2_data.allocate_bmi_timeseries(bmi='gci')

        info_message("Computing spatial K-Means for each scene GCI")
        jp2_data.compute_spatial_kmeans(bmi='gci')

        info_message("Computing temporal K-Means for each scene GCIs over time")
        jp2_data.compute_temporal_kmeans(bmi='gci')

    if clargs.rci:
        info_message("Computing RCI for all scenes")
        jp2_data.compute_bmi_for_all(bmi='rci')

        info_message("Allocating RCI time series")
        jp2_data.allocate_bmi_timeseries(bmi='rci')

        info_message("Computing spatial K-Means for each scene RCI")
        jp2_data.compute_spatial_kmeans(bmi='rci')

        info_message("Computing temporal K-Means for each scene RCIs over time")
        jp2_data.compute_temporal_kmeans(bmi='rci')

    if clargs.verbose_plot:
        plt.show()
