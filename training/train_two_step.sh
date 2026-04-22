#!/bin/bash
#SBATCH --job-name=OceanNet_TwoStep
#SBATCH --account=your_account_name
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@umich.edu
#SBATCH --output=oceanet_twostep_%j.log
#SBATCH --error=oceanet_twostep_%j.err

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Load necessary modules
module purge
module load python3.9-anaconda/2021.11
module load cuda/11.8.0

# Install/upgrade required packages if needed (using --user to avoid permissions issues)
pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install --user numpy xarray netcdf4

# Verify Python and packages
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" || echo "Warning: PyTorch not found or CUDA not available"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Print GPU information
nvidia-smi

# Navigate to project directory (adjust path as needed)
cd $SLURM_SUBMIT_DIR

# Run the training script
echo "Starting two-step model training..."
python OceanNet_train_2step_GSM.py

echo "Training completed at: $(date)"