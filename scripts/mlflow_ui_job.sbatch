#!/bin/bash

#SBATCH --job-name="dss-mlflow-ui"
#SBATCH --output="/beegfs/ws/0/s4610340-sleep_states/.logs/mlflow_ui_job/slurm-%j.out"
#SBATCH --error="/beegfs/ws/0/s4610340-sleep_states/.logs/mlflow_ui_job/slurm-%j.out"
#SBATCH --account="p_scads"
#SBATCH --time=8:00:00
#SBATCH --partition=romeo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=5000

module load modenv/hiera GCC/11.3.0 Python/3.10.4 CUDA/11.8.0

source /beegfs/ws/0/s4610340-sleep_states/kaggle-detect_sleep_states/.venv/bin/activate

mlflow ui --backend-store-uri /beegfs/ws/0/s4610340-sleep_states/kaggle-detect_sleep_states/ml-runs