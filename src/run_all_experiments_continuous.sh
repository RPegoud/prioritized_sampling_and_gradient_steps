#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <environment-name>"
    exit 1
fi

env=$1

trainers=(
     "base_ppo_continuous"
     "parallel_ppo_1_continuous"
    # "parallel_ppo_1a_continuous"
    # "parallel_ppo_1b_continuous"
    # "parallel_ppo_1c_continuous"
    # "parallel_ppo_1d_continuous"
)

alphas=("0.2" "0.4" "0.6")

function create_and_submit {
    trainer=$1
    env=$2
    alpha=$3

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${trainer}_${env}_${alpha}
#SBATCH --output=${trainer}_${env}_${alpha}%j.out
#SBATCH --error=${trainer}_${env}_${alpha}%j.err
#SBATCH --constraint=v100-16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --hint=nomultithread
#SBATCH --account=xyk@v100

module purge
conda deactivate

set -x
module load python/3.10.4
conda activate igs
wandb offline
cd \$WORK/prioritized_sampling_and_gradient_steps/src
python main_continuous.py --total-timesteps 5000000 --log-results --trainer "$trainer" --env-name "$env" --alpha "$alpha"
EOF

    echo "Submitting ${trainer} run on ${env}, alpha: ${alpha}"
}

for trainer in "${trainers[@]}"; do
    if [[ "$trainer" == "parallel_ppo_1c_continuous" || "$trainer" == "parallel_ppo_1d_continuous" ]]; then
        for alpha in "${alphas[@]}"; do
            create_and_submit "$trainer" "$env" "$alpha"
        done
    else
        create_and_submit "$trainer" "$env" "0.2"
    fi
done
