# Create Downloads directory
```bash
mkdir ~/Downloads
```

# Change to Downloads directory
```bash
cd ~/Downloads/
```

# wget miniconda3 from source
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

# Install MiniConda3
```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

# Return to home directory
```bash
cd
```

# Add Conda-Forge channel
```bash
conda config --add channels conda-forge
```

# Set priority for Conda-Forge channel
```bash
conda config --set channel_priority strict
```

# Clone KMeans NDVI Repo
```bash
git clone https://github.com/exowanderer/kmeans_doeberitzer_heide_sentinel_l2a
```

# Change directory to github repo
```bash
cd kmeans_doeberitzer_heide_sentinel_l2a/
```

# Create kmeans_ndvi conda environment
```bash
conda env create --file kmeans_ndvi_environment.yml
```

# Active kmeans_ndvi conda environment
```bash
conda activate kmeans_ndvi
```

# Configure AWS with AWS_ACCESS_KEY_ID and AWS_SECRET_KEY
```bash
aws configure
```

# Create environment variable for STAC_API_URL

This can be added the ~/.bashrc as well
```bash
export STAC_API_URL='https://earth-search.aws.element84.com/v0'
```

# Activate run script for base parameters without plotting
```bash
python doeberitzer_kmeans_ndvi.py --band_names b04 b08 --start_date 2020-01-01 --end_date 2020-02-01 --cloud_cover 1 --download --verbose
```

# Run PyTest operations
```bash
python -m pytest
```

# Open Jupyter Lab to run Jupyter notebook on local machine
```bash
jupyter lab
```

# If using on EC2, follow the instructions here to access the notebook

# https://medium.com/@alexjsanchez/python-3-notebooks-on-aws-ec2-in-15-mostly-easy-steps-2ec5e662c6c6
```bash
jupyter lab --no-browser
```
