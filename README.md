```
module load modenv/hiera GCC/11.3.0 Python/3.10.4 CUDA/11.8.0
```
```
singularity shell --nv -B /beegfs/.global0/ws/s4610340-sleep_states/.tmp:/tmp -B /beegfs/.global0/ws/s4610340-sleep_states:/mnt/s4610340-sleep_states ./detect_sleep_states.sif
```
```
cd /mnt/s4610340-sleep_states/kaggle-detect_sleep_states/src
python -m detect_sleep_states.train
```