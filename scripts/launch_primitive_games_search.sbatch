#! /bin/bash
# launch_primitive_games_search.sbatch
#
#SBATCH --job-name=jlee/HRL_ODE
#SBATCH -p cpu
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH -t 0-8:00
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G   
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR

##The arguments are in order of ${logdir} ${env_name} ${max_num_episode} ${lr} ${save_frequency} ${crop} ${random_policy} ${policy}

echo experiments/procgen_composite/procgen/primitive_train.py --logdir ${1} --env-name ${2} --max-num-episode ${3} --lr ${4} --save-frequency ${5} ${6} ${7} ${8}

python experiments/procgen_composite/procgen/primitive_train.py --logdir ${1} --env-name ${2} --max-num-episode ${3} --lr ${4} --save-frequency ${5} ${6} ${7} ${8}