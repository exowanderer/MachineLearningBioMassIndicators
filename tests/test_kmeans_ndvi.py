"""List of tests for PyTests"""
import pytest
from context import SentinelAOI, SentinelAOIParams, KMeansNDVI, KMeansNDVIParams


def test_SentinelAOI():
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
    pass


def test_load_data_into_struct():
    pass


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
