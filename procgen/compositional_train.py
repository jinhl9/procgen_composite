import argparse
from datetime import datetime
import os
import json
import numpy as np
import joblib as jl
from typing import Union
from procgen import ProcgenGym3Env


def sigmoid(x):
    if x < 0:
        return np.exp(x) / (1 + np.exp(x))
    else:
        return 1. / (1 + np.exp(-x))


def train(env, max_num_episode: int, lr: float, V_move_init: float,
          V_attack_init: float, crop: bool, random_policy: bool, policy: bool,
          save_frequency: int, W_move: Union[np.ndarray,
                                             None], W_attack: Union[np.ndarray,
                                                                    None]):

    episode_count = 0
    successful_episode = 0
    cumul_reward = 0
    reward_history = []
    if crop:
        N_features = 35 * 64 * 3
    else:
        N_features = 64 * 64 * 3

    game_type = env.options['env_name'].split('_')[1]

    if game_type == 'compositional':
        s = np.random.rand()
        if s > 0.5:
            action1 = np.array([1])
        else:
            action1 = np.array([7])
        action2 = np.array([9])
        success_reward_val = 1

    T = int(env.options['env_name'].split('_')[-1])
    if W_move == None:
        W_move = np.random.normal(loc=0, scale=0.001, size=(N_features))
    if W_attack == None:
        W_attack = np.random.normal(loc=0, scale=0.001, size=(N_features))

    V_move = V_move_init
    V_attack = V_attack_init
    W_move_saved = np.zeros(
        (int(max_num_episode // save_frequency) + 1, N_features))
    W_attack_saved = np.zeros(
        (int(max_num_episode // save_frequency) + 1, N_features))
    V_saved = np.zeros((int(max_num_episode // save_frequency) + 1, 2))
    while episode_count <= max_num_episode:
        if episode_count % save_frequency == 0:
            W_move_saved[int(episode_count // save_frequency)] = W_move
            W_attack_saved[int(episode_count // save_frequency)] = W_attack
            V_saved[int(episode_count // save_frequency)] = np.array(
                [V_move, V_attack])
        reward, ob, first = env.observe()
        if first[0]:
            if episode_count != 0:
                cumul_reward += reward[0] == success_reward_val
                reward_history.append(cumul_reward.item())
            if reward[
                    0] == success_reward_val and episode_count != 0:  ## If the last episode was a success, do a weight update:
                w_move_temp = W_move.copy()
                w_attack_temp = W_attack.copy()
                u = np.mean(y * X, axis=1).T
                W_move += lr * u * V_move
                W_attack += lr * u * V_attack

                V_move += V_move + np.dot(y, np.sum(w_move_temp * X.T,
                                                    axis=1)) / T
                V_attack += V_attack + np.dot(
                    y, np.sum(w_attack_temp * X.T, axis=1)) / T

                norm = np.linalg.norm([V_move, V_attack])
                V_move = V_move / norm
                V_attack = V_attack / norm

                successful_episode += 1
            step = 0
            episode_count += 1
            y = np.zeros([T])
            X = np.zeros([N_features, T])

        if crop:
            state = ob['rgb'][:, :35, :, :].flatten()
        else:
            state = ob['rgb'].flatten()

        if random_policy:
            action = 2 * (np.random.rand() > 0.5) - 1
        elif policy:
            move_val = np.dot(W_move, state) / T
            attack_val = np.dot(W_attack, state) / T
            s = sigmoid(move_val * V_move + attack_val * V_attack)
            action = 2 * ((np.random.rand() < s) - 1 / 2)
        else:
            move_val = np.dot(W_move, state) / T
            attack_val = np.dot(W_attack, state) / T
            action = np.sign(move_val * V_move + attack_val * V_attack)
        if action == -1:
            env.act(action1)
        elif action == +1:
            env.act(action2)
        y[step] = action
        X[:, step] = state
        step += 1

    return cumul_reward, reward_history, W_move_saved, W_attack_saved, V_saved


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Run a training for primitive game")

    parser.add_argument("--logdir",
                        type=str,
                        help="Path to log the experiment")

    parser.add_argument("--env-name",
                        type=str,
                        choices=[
                            "bossfight_compositional_100",
                            "bossfight_compositional_200",
                            "bossfight_compositional_400"
                        ],
                        help="game environment type")
    parser.add_argument("--max-num-episode",
                        type=int,
                        help='number of training episodes')

    parser.add_argument("--lr", type=float, help='learning_rate')

    parser.add_argument("--V-move-init", type=float, help='Initial V_move')

    parser.add_argument("--V-attack-init", type=float, help='Initial V_attack')
    parser.add_argument("--save-frequency",
                        type=int,
                        help='Frequency of saving weights')
    parser.add_argument("--crop",
                        action="store_true",
                        default=False,
                        help='Crop input environment rgb pixels or not')
    parser.add_argument("--random-policy",
                        action="store_true",
                        default=False,
                        help='Use random policy while training or not')
    parser.add_argument("--policy",
                        action="store_true",
                        default=False,
                        help='Use sigmoidal policy or not')

    args = parser.parse_args()

    env = ProcgenGym3Env(num=1,
                         env_name=args.env_name,
                         use_backgrounds=False,
                         restrict_themes=True)
    cumul_reward, reward_history, W_move_saved, W_attack_saved, V_saved = train(
        env=env,
        max_num_episode=args.max_num_episode,
        V_attack_init=args.V_attack_init,
        V_move_init=args.V_move_init,
        lr=args.lr,
        crop=args.crop,
        random_policy=args.random_policy,
        policy=args.policy,
        save_frequency=args.save_frequency,
        W_move=None,
        W_attack=None)

    args_dict = vars(parser.parse_args())

    log_time = datetime.now().strftime("%Y%m%d%H%M%S.%f")
    log_folder = os.path.join(args.logdir, log_time)
    if not os.path.isdir(log_folder):
        os.makedirs(log_folder)

    with open(os.path.join(log_folder, 'args_train.json'), 'w') as f:
        json.dump(args_dict, f)

    result_dict = {
        'cumul_reward': cumul_reward,
        'reward_history': reward_history,
        'W_move_saved': W_move_saved,
        'W_attack_saved': W_attack_saved,
        'V_saved': V_saved
    }
    jl.dump(result_dict, os.path.join(log_folder, 'result_train.jl'))
