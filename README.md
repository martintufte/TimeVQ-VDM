# TimeVQ-VDM
This is an official Github repository for the PyTorch implementation of TimeVQ-VDM

TimeVQ-VDM is a time series generation model that utilizes vector quantization for data compression into the discrete latent space (stage1) and a variational diffusion model the prior learning (stage2).

## Install / Environment setup
The following command creates the conda environment from the `environment.yml`. The installed environment is named `timevqvdm`.
```
$ conda env create -f environment.yml
```
You can activate the environment by running
```
$ conda activate timevqvdm
```

## Usage

### Configuration
- `configs/config.yaml`: configuration for dataset, data loading, optimizer, and models (_i.e.,_ encoder, decoder, vector-quantizer, and VDM)
- `config/sconfig_cas.yaml`: configuration for running CAS, Classification Accuracy Score (= TSTR, Training on Synthetic and Test on Real).
