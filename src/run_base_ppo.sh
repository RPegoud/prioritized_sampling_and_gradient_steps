#!/bin/bash
#SBATCH --job-name=ppo_base
#SBATCH --output=ppo_base%j.out
#SBATCH --error=ppo_base%j.err
#SBATCH --constraint=v100-16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --time=01:00:00
#SBATCH --qos=qos_gpu-dev
#SBATCH --hint=nomultithread
#SBATCH --account=xyk@v100

set -x # activer l’echo des commandes
cd $WORK/prioritized_sampling_and_gradient_steps/src/
srun python -u base_ppo.py --total-timesteps 50000 --log-results

# srun --job-name=ppo_base --output=ppo_base%j.out --error=ppo_base%j.err --constraint=v100-16g --nodes=1 --ntasks=1 --gres=gpu:4 --cpus-per-task=10 --time=01:00:00 --qos=qos_gpu-dev --hint=nomultithread --account=xyk@v100 --pty bash
# python -u python base_ppo.py --total-timesteps 50000 --log-results