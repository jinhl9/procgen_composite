#! /bin/bash

echo "Launch a general search for primitive games"

args=("$@")
logdir='procgen_logs/040124_compositional/'
max_num_episode=2000000
save_frequency=100
set | grep '^[a-z].*='
mkdir -p $logdir
for env_name in bossfight_compositional_100 bossfight_compositional_200 bossfight_compositional_400 ; do
for lr in 0.5 0.1 1. 2. 5. 10. ; do
for crop in '--crop' ''; do 
for policy in '--policy' ''; do
for random_policy in '--random-policy' ''; do
for V_move_init in 0.9 0.5 0.1; do

sbatch experiments/procgen_composite/scripts/launch_compositional_game_search.sbatch ${logdir} ${env_name} ${max_num_episode} ${lr} ${save_frequency} ${crop} ${random_policy} ${policy} ${V_move_init}

done
done
done
done
done
done

