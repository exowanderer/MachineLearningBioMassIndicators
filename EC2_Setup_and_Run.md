# Create Downloads directory
mkdir ~/Downloads

# Change to Downloads directory
cd ~/Downloads/

# wget miniconda3 from source
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install MiniConda3
bash Miniconda3-latest-Linux-x86_64.sh

# Return to home directory
cd

# Add Conda-Forge channel
conda config --add channels conda-forge

# Set priority for Conda-Forge channel
conda config --set channel_priority strict

# Clone KMeans NDVI Repo
git clone https://github.com/exowanderer/kmeans_doeberitzer_heide_sentinel_l2a

# Change directory to github repo
cd kmeans_doeberitzer_heide_sentinel_l2a/

# Create kmeans_ndvi conda environment
conda env create --file kmeans_ndvi_environment.yml

# Active kmeans_ndvi conda environment
conda activate kmeans_ndvi

# Configure AWS with AWS_ACCESS_KEY_ID and AWS_SECRET_KEY
aws configure

# Create environment variable for STAC_API_URL
# This can be added the ~/.bashrc as well
export STAC_API_URL='https://earth-search.aws.element84.com/v0'

# Activate run script for base parameters without plotting
python doeberitzer_kmeans_ndvi.py --band_names b04 b08 --start_date 2020-01-01 --end_date 2020-02-01 --cloud_cover 1 --download --verbose

# Run PyTest operations
python -m pytest

# Open Jupyter Lab to run Jupyter notebook on local machine
jupyter lab

# If using on EC2, follow the instructions here to access the notebook
# https://medium.com/@alexjsanchez/python-3-notebooks-on-aws-ec2-in-15-mostly-easy-steps-2ec5e662c6c6
jupyter lab --no-browser
