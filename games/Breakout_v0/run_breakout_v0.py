import gym
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Set log level: only output error.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only use #0 GPU.
import time

from games.Breakout_v0.hyperparameters import MAX_EPISODES
from games.Breakout_v0.hyperparameters import REPLY_START_SIZE
from games.Breakout_v0.hyperparameters import UPDATE_FREQUENCY


def run_Breakout(env, RL, model):
    total_steps = 0  # total steps after training begins.
    steps_total = []  # sum of steps until one episode.
    episodes = []  # episode's index.
    steps_episode = []  # steps for every single episode.

    for i_episode in range(MAX_EPISODES):
        print('episode:' + str(i_episode))
        # initial observation
        observation = env.reset()
        # counter for one episode
        episode_steps = 0

        if 'sarsa' in model:
            action = RL.choose_action(observation, total_steps)

        while True:
            env.render()
            # RL choose action based on observation
            if 'sarsa' in model:
                pass
            else:
                action = RL.choose_action(observation, total_steps)
            # RL take action and get next observation and reward
            observation_, reward, done, info = env.step(action)

            if 'sarsa' in model:
                action_ = RL.choose_action(observation_, total_steps)
                RL.store_transition(observation, action, reward, observation_, action_)
            else:
                RL.store_transition(observation, action, reward, observation_)

            # 'step % UPDATE_FREQUENCY == 0' is for frame-skipping technique.
            if (total_steps > REPLY_START_SIZE) and (total_steps % UPDATE_FREQUENCY == 0):
                RL.learn()

            observation = observation_
            episode_steps += 1

            # break while loop when end of this episode
            if done:
                print('episode ', i_episode, ' finished')
                total_steps += episode_steps
                steps_episode.append(episode_steps)
                steps_total.append(total_steps)
                episodes.append(i_episode)
                break

            if 'sarsa' in model:
                action = action_

    return [np.vstack((episodes, steps_total)), np.vstack((episodes, steps_episode))]


def main(model):
    env = gym.make('Breakout-v0')
    # build network.
    n_actions = env.action_space.n
    n_features = env.observation_space.high.size
    if model == 'double_dqn':
        from brains.double_dqn import DeepQNetwork
        from games.Breakout_v0.network_double_dqn import build_network
        inputs, outputs, weights = build_network(n_features, n_actions, lr=0.01)
        # get the DeepQNetwork Agent
        RL = DeepQNetwork(
            n_actions=n_actions,
            n_features=n_features,
            eval_net_input=inputs[0],
            target_net_input=inputs[1],
            q_target=inputs[2],
            q_eval_net_out=outputs[0],
            loss=outputs[1],
            train_op=outputs[2],
            q_target_net_out=outputs[3],
            e_params=weights[0],
            t_params=weights[1],
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=100,
            memory_size=2000,
            e_greedy_increment=0.001,
            output_graph=True,
        )
    elif model == 'dueling_dqn':
        from brains.dueling_dqn import DeepQNetwork
        from games.Breakout_v0.network_dueling_dqn import build_network
        inputs, outputs, weights = build_network(n_features, n_actions, lr=0.01)
        # get the DeepQNetwork Agent
        RL = DeepQNetwork(
            n_actions=n_actions,
            n_features=n_features,
            eval_net_input=inputs[0],
            target_net_input=inputs[1],
            q_target=inputs[2],
            q_eval_net_out=outputs[0],
            loss=outputs[1],
            train_op=outputs[2],
            q_target_net_out=outputs[3],
            e_params=weights[0],
            t_params=weights[1],
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=100,
            memory_size=2000,
            e_greedy_increment=0.001,
            output_graph=True,
        )
    else:  # sarsa
        from brains.sarsa import DeepQNetwork
        from games.Breakout_v0.network_sarsa import build_network
        inputs, outputs, weights = build_network(n_features, n_actions, lr=0.01)
        # get the DeepQNetwork Agent
        RL = DeepQNetwork(
            n_actions=n_actions,
            n_features=n_features,
            eval_net_input=inputs[0],
            target_net_input=inputs[1],
            q_target=inputs[2],
            q_eval_net_out=outputs[0],
            loss=outputs[1],
            train_op=outputs[2],
            q_target_net_out=outputs[3],
            e_params=weights[0],
            t_params=weights[1],
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=100,
            memory_size=2000,
            e_greedy_increment=0.001,
            output_graph=True,
        )

    # Calculate running time
    start_time = time.time()

    run_Breakout(env, RL, model)

    end_time = time.time()
    running_time = (end_time - start_time) / 60

    fo = open("./logs/running_time.txt", "w")
    fo.write(str(running_time) + "minutes")
    fo.close()


if __name__ == '__main__':
    # # change different models here:
    # double_dqn, dueling_dqn, sarsa...
    main(model='sarsa')
