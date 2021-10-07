"""List of tests for PyTests"""
import pytest
from context import SentinelAOI, SentinelAOIParams, KMeansNDVI


def test_SentinelAOI():
    instance = SentinelAOI(**SentinelAOIParams().__dict__)


def test_SentinelAOIParams():
    pass


def test_KMeansNDVI():
    pass


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
