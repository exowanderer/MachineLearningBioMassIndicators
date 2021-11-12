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

# Activate newly created Conda system

```bash
source ~/.bashrc
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
git clone https://github.com/exowanderer/MachineLearningBioMassIndicators
```

# Change directory to github repo

```bash
cd MachineLearningBioMassIndicators/
```

# Create mlbmi conda environment

```bash
conda env create --file mlbmi_environment.yml
```

# Active mlbmi conda environment

```bash
conda activate mlbmi
```

# Configure AWS with AWS_ACCESS_KEY_ID and AWS_SECRET_KEY

Only necessary is the user wants to request the JP2 files

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
python doeberitzer_mlbmi.py --band_names b04 b08 --start_date 2020-01-01 --end_date 2020-02-01 --cloud_cover 1 --download --verbose
```

# Run PyTest operations

```bash
python -m pytest
```

# Open Jupyter Lab to run Jupyter notebook on local machine

```bash
jupyter lab
```

# If using on EC2, follow the instructions [here](https://medium.com/@alexjsanchez/python-3-notebooks-on-aws-ec2-in-15-mostly-easy-steps-2ec5e662c6c6) to access the notebook

```bash
jupyter lab --no-browser
```
