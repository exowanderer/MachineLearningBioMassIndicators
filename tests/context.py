"""Context File for imports for testing procedures"""
import os
import sys

# pylint: disable=unused-import
from mlbmi import (
    SentinelAOI,
    SentinelAOIParams,
    KMeansNDVI,
    KMeansNDVIParams,
    debug_message
)

# Path hacks to make the code available for testing
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../mlbmi"
        )
    )
)
