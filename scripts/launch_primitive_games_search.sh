#! /bin/bash

echo "Launch a general search for primitive games"

args=("$@")
logdir='procgen_logs/040124_move_primitive/'
max_num_episode=800000
save_frequency=100
set | grep '^[a-z].*='
mkdir -p $logdir
for env_name in bossfight_move_100 bossfight_move_200 bossfight_move_400 ; do
for lr in 0.5 0.1 1. 2. 5. 10. ; do
for crop in '--crop' ''; do 
for policy in '--policy' ''; do
for random_policy in '--random-policy' ''; do

sbatch experiments/procgen_composite/scripts/launch_primitive_games_search.sbatch ${logdir} ${env_name} ${max_num_episode} ${lr} ${save_frequency} ${crop} ${random_policy} ${policy}

done
done
done
done
done

