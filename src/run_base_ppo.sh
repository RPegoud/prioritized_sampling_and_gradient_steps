#!/bin/bash
#SBATCH --job-name=parallel_ppo
#SBATCH --output=parallel_ppo%j.out
#SBATCH --error=parallel_ppo%j.err
#SBATCH --constraint=v100-16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu-dev
#SBATCH --hint=nomultithread
#SBATCH --account=xyk@v100

set -x
conda activate igs
wandb offline
python main.py --total-timesteps 500000 --log-results --trainer base_ppo --env-name Breakout-MinAtar