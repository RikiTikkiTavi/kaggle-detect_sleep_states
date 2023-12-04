#!/bin/bash

#SBATCH --job-name="dss-hopt-pp"
#SBATCH --output="/beegfs/ws/0/s4610340-sleep_states/.logs/dss-hopt-slurm-%j.out"
#SBATCH --error="/beegfs/ws/0/s4610340-sleep_states/.logs/dss-hopt-slurm-%j.out"
#SBATCH --account="p_scads"
#SBATCH --time=10:00:00
#SBATCH --partition=alpha
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=20000

python -m detect_sleep_states.inference\
    dir=alpha\
    model.params.encoder_name=resnet34\
    model.params.encoder_weights=null\
    num_workers=2\
    exp_name=test\
    batch_size=32\
    duration=5760\
    downsample_rate=3\
    pp.score_th=0.05\
    pp.distance=40\
    feature_extractor.params.base_filters=32\
    feature_extractor.params.kernel_sizes=[""]\
    'checkpoint_path=/beegfs/ws/0/s4610340-sleep_states/ml-runs/355750298837538791/c1695ef28cd8425fa02d226f1a845e62/artifacts/model/state_dict.pth'\
    phase=test

module load GCC/11.3.0 Python/3.10.4 CUDA/12.0.0

source /beegfs/ws/0/s4610340-sleep_states/kaggle-detect_sleep_states/.venv/bin/activate

python -m detect_sleep_states.train --multirun hydra.launcher.n_jobs=1 n_chunks_visualize=1 exp_name="CNNSpectrogram-UNet-hopt_pp" run_name="base-focal-pp" loss="focal" num_workers=6 pp.score_th=0.09,0.11,0.25 pp.distance=40,60,80