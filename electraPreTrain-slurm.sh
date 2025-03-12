#!/bin/bash
#SBATCH -J bkNLP            # job name
#SBATCH -o log_slurm.o%j    # output and error file name (%j expands to jobID)

#SBATCH --ntasks=1           # gpu's per task
#SBATCH --nodes=1            # number of nodes you want to run on
#SBATCH --gres=gpu:1     # request 1 L40 GPU
#SBATCH -p gpu-l40          # queue (partition)
#SBATCH -t 96:00:00         # run time (hh:mm:ss)

# (optional) Print some debugging information
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated  = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated  = $SLURM_NTASKS"
echo "GPU Allocated              = $SLURM_JOB_GPUS"
echo ""
nvidia-smi

# Activate conda environment properly
. ~/.bashrc
conda activate NLP_25
pwd

echo "Starting python script"
# Run the python script
srun python3 trainModel.py
