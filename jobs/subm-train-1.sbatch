#!/bin/bash

#SBATCH --job-name="dss-opt-u++"
#SBATCH --output="/beegfs/ws/0/s4610340-sleep_states/.logs/dss-hopt-slurm-%j.out"
#SBATCH --error="/beegfs/ws/0/s4610340-sleep_states/.logs/dss-hopt-slurm-%j.out"
#SBATCH --account="p_scads"
#SBATCH --time=8:00:00
#SBATCH --partition=alpha
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=10000

module load GCC/11.3.0 Python/3.10.4 CUDA/12.0.0

source /beegfs/ws/0/s4610340-sleep_states/kaggle-detect_sleep_states/.venv/bin/activate

python -m detect_sleep_states.train n_chunks_visualize=1 features="[anglez, enmo, hour_cos, hour_sin]" feature_extractor.params.base_filters=32 exp_name="CNNSpectrogram-UNet++" run_name="base-focal-u++" loss="focal" num_workers=6 loss.params.gamma=0.8 loss.params.pos_weight="[1,3,3]"