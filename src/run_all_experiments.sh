#!/bin/bash

trainers=(
    "base_ppo"
    "parallel_ppo_1"
    "parallel_ppo_1a"
    "parallel_ppo_1b"
    "parallel_ppo_1c"
    "parallel_ppo_1d"
)

envs=(
    "Breakout-MinAtar"
    "Freeway-MinAtar"
    "SpaceInvaders-MinAtar"
    "Asterix-MinAtar"
)

alphas=("0.2" "0.4" "0.6")

for trainer in "${trainers[@]}"
do
    for env in "${envs[@]}"
    do
        if [[ "$trainer" == "parallel_ppo_1c" || "$trainer" == "parallel_ppo_1d" ]]
        then
            for alpha in "${alphas[@]}"
            do
                create_and_submit "$trainer" "$env" "$alpha"
            done
        else
            create_and_submit "$trainer" "$env" "0.2"
        fi
    done
done

function create_and_submit {
    trainer=$1
    env=$2
    alpha=$3
    cat <<EOF > "${trainer}_${env}_${alpha}.slurm"
#!/bin/bash
#SBATCH --job-name=${trainer}_${env}_${alpha}
#SBATCH --output=${trainer}_${env}_${alpha}%j.out
#SBATCH --error=${trainer}_${env}_${alpha}%j.err
#SBATCH --constraint=v100-16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --time=04:00:00
#SBATCH --qos=qos_gpu-dev
#SBATCH --hint=nomultithread
#SBATCH --account=xyk@v100

set -x
conda activate igs
wandb offline
python main.py --total-timesteps 10000000 --log-results --trainer "$trainer" --env-name "$env" --alpha "$alpha"
EOF
    sbatch "${trainer}_${env}_${alpha}.slurm"
}
