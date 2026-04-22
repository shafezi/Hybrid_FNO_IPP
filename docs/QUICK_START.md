# Quick Start Guide - Great Lakes Cluster

## Files to Upload

**Python Scripts:**
- `OceanNet_train_single_step_GMS.py`
- `OceanNet_train_2step_GSM.py`
- `data_loader_SSH.py`
- `data_loader_SSH_two_step.py`

**Batch Scripts:**
- `train_single_step.sh`
- `train_two_step.sh`

**Data:**
- Entire `extracted_data/` directory (all .nc files)

## Quick Setup (5 Steps)

### 1. Upload Files
```bash
# From local machine
rsync -avz --progress \
  OceanNet_train_*.py data_loader_*.py train_*.sh \
  your_uniqname@greatlakes.arc-ts.umich.edu:~/oceanet_training/

rsync -avz --progress extracted_data/ \
  your_uniqname@greatlakes.arc-ts.umich.edu:~/oceanet_training/extracted_data/
```

### 2. SSH to Great Lakes
```bash
ssh your_uniqname@greatlakes.arc-ts.umich.edu
cd ~/oceanet_training
```

### 3. Install Packages (Optional - can also be done in batch script)
```bash
module load python3.9-anaconda/2021.11 cuda/11.8.0
pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install --user numpy xarray netcdf4
# Or uncomment pip install lines in batch scripts to install automatically
```

### 4. Edit Batch Scripts
```bash
# Edit both train_single_step.sh and train_two_step.sh
nano train_single_step.sh
# Change: --account=your_account_name
# Change: --mail-user=your_email@umich.edu
chmod +x train_*.sh
```

### 5. Submit Jobs
```bash
# Submit single-step first
sbatch train_single_step.sh

# Wait for completion, then submit two-step
sbatch train_two_step.sh
```

## Monitor Jobs
```bash
squeue -u your_uniqname              # Check status
tail -f oceanet_single_<job_id>.log  # View output
scancel <job_id>                      # Cancel if needed
```

## Expected Runtime
- Single-step: ~24-48 hours (180 epochs)
- Two-step: ~24-48 hours (180 epochs)

## Output Files
- Models: `Models/FNO_single_trialPECstep.pt`
- Models: `Models/FNO_double_trialPECstep.pt`
- Logs: `oceanet_single_<job_id>.log`
- Logs: `oceanet_twostep_<job_id>.log`

