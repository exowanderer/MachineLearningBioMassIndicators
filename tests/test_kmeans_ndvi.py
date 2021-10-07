"""List of tests for PyTests"""
import os
import joblib
import pytest

from context import (
    SentinelAOI,
    SentinelAOIParams,
    KMeansNDVI,
    KMeansNDVIParams,
    debug_message
)


def test_SentinelAOI():
    """Test that the input system works correctly for SentinelAOI"""

    param_keys = [
        'geojson', 'band_names', 'collection', 'start_date',
        'end_date', 'cloud_cover', 'download', 'verbose', 'quiet'
    ]
    sentinel_params = SentinelAOIParams()
    sentinel_params.geojson = f"../{sentinel_params.geojson}"
    instance = SentinelAOI(**sentinel_params.__dict__)
    for key in param_keys:
        val1 = instance.__dict__[key]
        val2 = instance.__dict__[key]
        assert(val1 == val2), f"{key} not matched"


def test_KMeansNDVI():
    """Test that the input system works correctly for KMeansNDVI"""
    param_keys = [
        'geojson', 'band_names', 'collection', 'start_date',
        'end_date', 'cloud_cover', 'n_sig', 'download', 'n_clusters',
        'quantile_range', 'verbose', 'verbose_plot', 'quiet'
    ]
    kmean_ndvi_params = KMeansNDVIParams()
    kmean_ndvi_params.geojson = f"../{kmean_ndvi_params.geojson}"
    instance = KMeansNDVI(**kmean_ndvi_params.__dict__)
    for key in param_keys:
        val1 = instance.__dict__[key]
        val2 = instance.__dict__[key]
        assert(val1 == val2), f"{key} not matched"


def test_download_and_acquire_images():
    """Test Earth on AWS Database and KMeansNDVI filepaths match"""
    expected_filepaths = {
        'B04': [
            'sentinel-s2-l2a/33UUU/R10m/2020-1-24/B04.jp2',
            'sentinel-s2-l2a/32UQD/R10m/2020-1-24/B04.jp2',
            'sentinel-s2-l2a/32UQD/R10m/2020-1-17/B04.jp2',
            'sentinel-s2-l2a/33UUU/R10m/2020-1-17/B04.jp2'
        ],
        'B08': [
            'sentinel-s2-l2a/33UUU/R10m/2020-1-24/B08.jp2',
            'sentinel-s2-l2a/32UQD/R10m/2020-1-24/B08.jp2',
            'sentinel-s2-l2a/32UQD/R10m/2020-1-17/B08.jp2',
            'sentinel-s2-l2a/33UUU/R10m/2020-1-17/B08.jp2'
        ]
    }

    kmean_ndvi_params = KMeansNDVIParams()
    kmean_ndvi_params.band_names = ['B04', 'B08']
    kmean_ndvi_params.start_date = '2020-01-01'
    kmean_ndvi_params.end_date = '2020-02-01'

    assert(os.path.exists(kmean_ndvi_params.geojson)),\
        f"GeoJSON file {kmean_ndvi_params.geojson} does not exist"

    instance = KMeansNDVI(**kmean_ndvi_params.__dict__)
    instance.download_and_acquire_images()
    instance_filepaths = instance.filepaths
    for band_name_, filepaths_ in expected_filepaths.items():
        assert(band_name_ in instance_filepaths.keys()),\
            f"{band_name_} not in instance.filepaths.keys()"

        assert(len(instance_filepaths[band_name_]) == len(filepaths_)),\
            "length of instance.filepaths does not match expected length"

        for instfpath, expfpaths in zip(instance_filepaths, expected_filepaths):
            assert(instfpath == expfpaths),\
                "found mismatched file path from instance to expected"


def test_load_data_into_struct():
    """Test KMeansNDVI data structure matches expectations"""
    kmean_ndvi_params = KMeansNDVIParams()
    kmean_ndvi_params.band_names = ['B04', 'B08']
    kmean_ndvi_params.start_date = '2020-01-01'
    kmean_ndvi_params.end_date = '2020-02-01'
    kmean_ndvi_params.download = True

    assert(os.path.exists(kmean_ndvi_params.geojson)),\
        f"GeoJSON file {kmean_ndvi_params.geojson} does not exist"

    instance = KMeansNDVI(**kmean_ndvi_params.__dict__)
    instance.download_and_acquire_images()
    instance.load_data_into_struct()

    expected_data_struct = joblib.load(
        'tests/empty_jp2_data_structure_for_test.joblib.save'
    )

    for scene_id_, res_data_ in instance.scenes.items():
        assert(scene_id_ in expected_data_struct.keys())
        for res_, date_data_ in res_data_.items():
            val = expected_data_struct[scene_id_]
            assert(res_ in val.keys())
            for date_, band_data_ in date_data_.items():
                val = expected_data_struct[scene_id_][res_]
                assert(date_ in val.keys())
                for band_name_, raster_data_ in band_data_.items():
                    val = expected_data_struct[scene_id_][res_][date_]
                    assert(band_name_ in val.keys())


"""
def test_compute_ndvi_for_all():
    pass


def test_allocate_ndvi_timeseries():
    pass


def test_compute_spatial_kmeans():
    pass


def test_compute_temporal_kmeans():
    pass


def test_sanity_check_ndvi_statistics():
    pass


def test_sanity_check_spatial_kmeans():
    pass


def test_sanity_check_temporal_kmeans():
    pass
"""
