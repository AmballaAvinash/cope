#!/bin/bash
#SBATCH --mail-type=BEGIN
#SBATCH -c 2  # Number of Cores per Task
#SBATCH --mem=100G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 20:00:00  # Job time limit
#SBATCH -o logs/slurm/train.log  # %j = job ID
#SBATCH --constraint=[a100]
export PYTHONPATH="${PYTHONPATH}=$(pwd):$PYTHONPATH"


module load conda/latest

conda activate 674_IVC

export PROJECT_ROOT=.

export HF_HOME=/work/pi_mccallum_umass_edu/aamballa_umass_edu/continuous_decoding/hfcache

 
python glue_qnli.py || true
python glue_cola.py || true


