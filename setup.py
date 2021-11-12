import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="Machine Learning Biomass Indicators",
    version="0.0.1",
    author="Jonathan Fraine",
    author_email="jdfraine@gmail.com",
    description=(
        "A demonstration of how to download, organize, and operate machine"
        " learning models on Sentinel-2 observations for biomass indicators"
    ),
    license="MIT",
    keywords="machine learning remote sensing biomass indicators",
    # url="http://packages.python.org/an_example_pypi_project",
    url="http://github.com/exowanderer/mlbmi",
    packages=['mlbmi', 'mlbmi.utils', 'tests'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
    ],
)
