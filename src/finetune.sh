#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
# Replace ACCOUNT with your account name before submitting.
#
#SBATCH --account=glab        # Replace ACCOUNT with your group account name
#SBATCH --job-name=train    # The job name
#SBATCH --gres=gpu:1
#SBATCH -c 2                    # The number of cpu cores to use (up to 32 cores per server)
#SBATCH --time=0-15:00            # The time the job will take to run in D-HH:MM
#SBATCH --mem-per-cpu=64gb         # The memory the job will use per cpu core
 
module load anaconda
module load cuda11.1/toolkit
#Command to execute Python program
pip install tft-torch
 
#first train boreal 2
CHECKPOINT_PATH="path/to/pretrained/ckpt"
python train_tft.py NDT.pickle --checkpoint $CHECKPOINT_PATH

# End of script
