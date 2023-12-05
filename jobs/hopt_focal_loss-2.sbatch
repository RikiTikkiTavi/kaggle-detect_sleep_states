#!/bin/bash

#SBATCH --job-name="dss-hopt-focal-hp"
#SBATCH --output="/beegfs/ws/0/s4610340-sleep_states/.logs/dss-hopt-slurm-%j.out"
#SBATCH --error="/beegfs/ws/0/s4610340-sleep_states/.logs/dss-hopt-slurm-%j.out"
#SBATCH --account="p_scads"
#SBATCH --time=8:00:00
#SBATCH --partition=alpha
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=3000

module load GCC/11.3.0 Python/3.10.4 CUDA/12.0.0

source /beegfs/ws/0/s4610340-sleep_states/kaggle-detect_sleep_states/.venv/bin/activate

python -m detect_sleep_states.train --multirun n_chunks_visualize=1 features="[anglez, enmo, hour_cos, hour_sin]" feature_extractor.params.base_filters=32 hydra.launcher.n_jobs=4 exp_name="CNNSpectrogram-UNet-focal-hopt" run_name="base-focal_hopt" loss="focal" num_workers=6 loss.params.gamma=1.5,2,2.5,3,4 loss.params.pos_weight="[1,3,3]"