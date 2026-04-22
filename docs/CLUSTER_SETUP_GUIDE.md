# Great Lakes Cluster Setup Guide

## Step 1: Connect to Great Lakes

```bash
ssh your_uniqname@greatlakes.arc-ts.umich.edu
```

## Step 2: Create Project Directory

```bash
mkdir -p ~/oceanet_training
cd ~/oceanet_training
```

## Step 3: Upload Files

From your local machine, upload all required files (see `FILES_TO_UPLOAD.md`):

```bash
# Upload Python scripts and batch scripts
scp OceanNet_train_single_step_GMS.py data_loader_SSH.py data_loader_SSH_two_step.py OceanNet_train_2step_GSM.py train_*.sh your_uniqname@greatlakes.arc-ts.umich.edu:~/oceanet_training/

# Upload data directory (use rsync for better progress tracking)
rsync -avz --progress extracted_data/ your_uniqname@greatlakes.arc-ts.umich.edu:~/oceanet_training/extracted_data/
```

## Step 4: Install Required Packages (Optional)

The batch scripts will use the system Python from the loaded modules. If you need to install packages beforehand, you can do:

```bash
cd ~/oceanet_training

# Load modules
module load python3.9-anaconda/2021.11
module load cuda/11.8.0

# Install packages with --user flag (no environment needed)
pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install --user numpy xarray netcdf4

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

**Note**: Alternatively, you can uncomment the pip install lines in the batch scripts to install packages automatically when the job runs.

## Step 5: Edit Batch Scripts

Edit the batch scripts to set your account and email:

```bash
# Edit train_single_step.sh
nano train_single_step.sh
# Change:
#   --account=your_account_name  →  --account=your_actual_account
#   --mail-user=your_email@umich.edu  →  --mail-user=your_actual_email@umich.edu

# Edit train_two_step.sh
nano train_two_step.sh
# Make the same changes
```

## Step 6: Make Batch Scripts Executable

```bash
chmod +x train_single_step.sh train_two_step.sh
```

## Step 7: Test Script (Optional)

Before submitting the full training, test that everything works:

```bash
# Test data loading
python -c "import xarray as xr; ds = xr.open_dataset('extracted_data/EnKF_surface_1993_5dmean_EC.nc'); print('Data loaded successfully'); print(f'Shape: {ds.SSH.shape}')"

# Test imports
python -c "from data_loader_SSH import load_test_data; print('Imports successful')"
```

## Step 8: Submit Jobs

### Submit Single-Step Training First

```bash
sbatch train_single_step.sh
```

This will return a job ID. Monitor the job:

```bash
# Check job status
squeue -u your_uniqname

# View output log
tail -f oceanet_single_<job_id>.log

# View error log
tail -f oceanet_single_<job_id>.err
```

### Submit Two-Step Training After Single-Step Completes

**Important**: Wait for single-step training to complete and save the model before submitting two-step training.

```bash
# Check if model was saved
ls -lh Models/FNO_single_trialPECstep.pt

# If model exists, submit two-step training
sbatch train_two_step.sh
```

## Step 9: Monitor Training

```bash
# Check job queue
squeue -u your_uniqname

# View recent output
tail -n 100 oceanet_single_<job_id>.log

# Check GPU usage (if on compute node)
nvidia-smi

# Cancel job if needed
scancel <job_id>
```

## Step 10: Retrieve Results

After training completes, download the trained models:

```bash
# From your local machine
scp your_uniqname@greatlakes.arc-ts.umich.edu:~/oceanet_training/Models/*.pt ./
scp your_uniqname@greatlakes.arc-ts.umich.edu:~/oceanet_training/oceanet_*.log ./
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Increase `--mem` in batch script or reduce `batch_size` in Python script
2. **CUDA Out of Memory**: Reduce `batch_size` in training script
3. **Module Not Found**: Ensure Python environment is activated in batch script
4. **File Not Found**: Check that data files are in `extracted_data/` directory
5. **Permission Denied**: Make sure batch scripts are executable (`chmod +x`)

### Checking Available Resources

```bash
# Check available partitions
sinfo

# Check your account limits
sacctmgr show user $USER

# Check GPU availability
sinfo -o "%P %G" | grep gpu
```

### Requesting More Resources

If you need more resources, edit the batch script:
- More memory: `--mem=64G`
- More CPUs: `--cpus-per-task=16`
- More time: `--time=72:00:00` (3 days)
- Multiple GPUs: `--gres=gpu:2` (requires code changes)

## Additional Resources

- Great Lakes User Guide: https://arc-ts.umich.edu/greatlakes/user-guide/
- SLURM Documentation: https://slurm.schedmd.com/documentation.html
- Contact: hpc-support@umich.edu

