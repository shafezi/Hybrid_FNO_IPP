# FNO OceanNet Training Information

## Overview
This repository contains two training scripts for Fourier Neural Operator (FNO) models for ocean sea surface height (SSH) prediction:

1. **Single-step model** (`OceanNet_train_single_step_GMS.py`) - Predicts one timestep ahead
2. **Two-step model** (`OceanNet_train_2step_GSM.py`) - Predicts two timesteps ahead simultaneously

## Dataset Details

### Training Data
- **Years**: 1993-2019 (27 years of data)
- **Data files**: `EnKF_surface_YYYY_5dmean_EC.nc` for each year (YYYY = 1993-2019)
- **Grid size**: ~320×600 spatial grid points
- **Temporal resolution**: 5-day mean SSH fields
- **Total timesteps per year**: 365 days (73 timesteps per year)
- **Training timesteps per year**: 361 (365 - lead time of 4)

### Validation Data
- **Year**: 2020
- **File**: `EnKF_surface_2020_5dmean_EC.nc`

### Normalization Files
- **Mean field**: `EnKF_surface_1993to2020_avg.nc` - Climatological mean SSH
- **Standard deviation**: `EnKF_surface_5day_std.nc` - Standard deviation of SSH

## Model Architecture

### FNO2d (Fourier Neural Operator 2D)
- **Fourier modes**: 128 modes in both dimensions
- **Width**: 20 channels
- **Layers**: 4 spectral convolution layers with residual connections
- **Activation**: GELU
- **Normalization**: InstanceNorm2d
- **Input**: SSH field + spatial grid coordinates (3 channels total)
- **Output**: Predicted SSH field (1 channel)

## Training Configuration

### Single-Step Model (`OceanNet_train_single_step_GMS.py`)
- **Lead time**: 4 timesteps (20 days ahead)
- **Batch size**: 30
- **Epochs**: 180
- **Learning rate**: 0.001
- **Optimizer**: Adam with weight decay 1e-4
- **Loss function**: Spectral regularizer + MSE
  - MSE weight: 0.1 (1 - lambda_reg)
  - Spectral loss weight: 0.9 (lambda_reg)
  - Wavenumber cutoff (zonal): 90
  - Wavenumber cutoff (meridional): 30
- **Numerical integrator**: PEC (Predictor-Evaluator-Corrector) with delta_t=1.0
- **Model output**: `./Models/FNO_single_trialPECstep.pt`

### Two-Step Model (`OceanNet_train_2step_GSM.py`)
- **Lead time**: 4 timesteps per step (predicts 2 steps ahead = 8 timesteps total = 40 days)
- **Batch size**: 30
- **Epochs**: 180
- **Learning rate**: 0.001
- **Optimizer**: Adam with weight decay 1e-4
- **Loss function**: Spectral regularizer + MSE for both timesteps
  - MSE weight: 0.1 (1 - lambda_reg)
  - Spectral loss weight: 0.9 (lambda_reg)
  - Wavenumber cutoff (zonal): 90
  - Wavenumber cutoff (meridional): 30
- **Numerical integrator**: PEC (Predictor-Evaluator-Corrector) with delta_t=1.0
- **Pretrained model**: Attempts to load `./Models/FNO_single_PECstep.pt` if available
- **Model output**: `./Models/FNO_double_trialPECstep.pt`

## Loss Function Details

### Spectral Loss
The loss function combines:
1. **MSE Loss**: Mean squared error between predicted and target SSH fields
2. **Spectral Loss**: Penalizes differences in high-frequency Fourier modes
   - Computes FFT along zonal (x) and meridional (y) directions
   - Compares Fourier coefficients above cutoff wavenumbers
   - Encourages model to capture fine-scale features

### Formula
```
Total Loss = (1-λ) × MSE + λ × [0.25×Spectral_x + 0.25×Spectral_y + ...]
where λ = 0.9
```

## Training Process

### Single Epoch
1. Loop through all training years (1993-2019)
2. Load data for each year
3. Normalize using climatological mean and std
4. Shuffle data randomly
5. Process in batches:
   - Forward pass through FNO model
   - Apply PEC numerical integrator
   - Compute loss
   - Backward pass and optimizer step

### Validation
- After each epoch:
  - Select 20 random validation samples from 2020 data
  - Run autoregressive prediction for 30 timesteps
  - Compute validation loss
  - Print epoch number and validation loss

## Data Loaders

### `data_loader_SSH.py` (Single-step)
- Loads SSH data from NetCDF files
- Creates input-label pairs with specified lead time
- Returns: input tensor, label tensor, ocean grid size

### `data_loader_SSH_two_step.py` (Two-step)
- Loads SSH data from NetCDF files
- Creates input-label pairs for two timesteps ahead
- Returns: input tensor, label tensor (step 1), label tensor (step 2), ocean grid size

## Running the Training

### Single-Step Model
```bash
python OceanNet_train_single_step_GMS.py
```

### Two-Step Model
```bash
python OceanNet_train_2step_GSM.py
```

**Note**: The two-step model requires a pretrained single-step model (`./Models/FNO_single_PECstep.pt`). If not found, it will start from scratch.

## Requirements
- PyTorch
- NumPy
- xarray
- netCDF4 (optional, falls back to xarray if not available)
- CUDA-capable GPU (recommended)

## Output
Trained models are saved in `./Models/` directory:
- Single-step: `FNO_single_trialPECstep.pt`
- Two-step: `FNO_double_trialPECstep.pt`

## Important Notes
- Training on full dataset (27 years) with batch_size=30 requires significant GPU memory (~320×600 grid)
- Each epoch processes ~9,747 training samples (27 years × 361 timesteps)
- Training time: Several hours to days depending on GPU
- Models use InstanceNorm2d for normalization
- NaN values in data are replaced with 0.0
- Ocean grid mask is used to compute loss only over ocean points

