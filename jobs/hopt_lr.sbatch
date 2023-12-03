#!/bin/bash

#SBATCH --job-name="dss-hopt-lr"
#SBATCH --output="/beegfs/ws/0/s4610340-sleep_states/.logs/dss-hopt-slurm-%j.out"
#SBATCH --error="/beegfs/ws/0/s4610340-sleep_states/.logs/dss-hopt-slurm-%j.out"
#SBATCH --account="p_scads"
#SBATCH --time=16:00:00
#SBATCH --partition=alpha
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=3000

module load GCC/11.3.0 Python/3.10.4 CUDA/12.0.0

source /beegfs/ws/0/s4610340-sleep_states/kaggle-detect_sleep_states/.venv/bin/activate

python -m detect_sleep_states.train --multirun exp_name="CNNSpectrogram-UNet-hopt_lr" run_name="base-focal_loss-lr" loss="focal" num_workers=6 loss.params.gamma=0.3 loss.params.pos_weight="[1,3,3]" optimizer.lr=0.0001,0.0003,0.0004,0.00045,0.0005,0.00055,0.0006,0.0007,0.0009