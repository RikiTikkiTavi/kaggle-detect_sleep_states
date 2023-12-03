#!/bin/bash

#SBATCH --job-name="dss-notebook"
#SBATCH --output="/beegfs/ws/0/s4610340-sleep_states/.logs/notebook_job_cpu/slurm-%j.out"
#SBATCH --error="/beegfs/ws/0/s4610340-sleep_states/.logs/notebook_job_cpu/slurm-%j.out"
#SBATCH --account="p_scads"
#SBATCH --time=8:00:00
#SBATCH --partition=alpha
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=3000

module load GCC/11.3.0 Python/3.10.4 CUDA/12.0.0

source /beegfs/ws/0/s4610340-sleep_states/kaggle-detect_sleep_states/.venv/bin/activate

export XDG_RUNTIME_DIR=""

cd /beegfs/ws/0/s4610340-sleep_states/kaggle-detect_sleep_states/notebooks && jupyter notebook --no-browser --port=8888