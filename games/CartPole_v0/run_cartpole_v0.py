import gym
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Set log level: only output error.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only use #0 GPU.
import time
import tensorflow as tf

from games.CartPole_v0.hyperparameters import MAX_EPISODES
from games.CartPole_v0.hyperparameters import REPLY_START_SIZE
from games.CartPole_v0.hyperparameters import UPDATE_FREQUENCY
from games.CartPole_v0.hyperparameters import WEIGHTS_SAVER_ITER
from games.CartPole_v0.hyperparameters import SAVED_NETWORK_PATH


def store_parameters(sess):
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(SAVED_NETWORK_PATH)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
        path_ = checkpoint.model_checkpoint_path
        step = int((path_.split('-'))[-1])
    else:
        # Re-train the network from zero.
        print("Could not find old network weights")
        step = 0
    return saver, step


def run_cartpole(env, RL, model, saver, load_step):
    total_steps = 0  # total steps after training begins.
    steps_total = []  # sum of steps until one episode.
    episodes = []  # episode's index.
    steps_episode = []  # steps for every single episode.

    for i_episode in range(MAX_EPISODES):
        print('episode:' + str(i_episode))
        observation = env.reset()
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

            if (total_steps > REPLY_START_SIZE) and (total_steps % WEIGHTS_SAVER_ITER == 0):
                saver.save(RL.sess, SAVED_NETWORK_PATH + '-dqn-' + str(total_steps + load_step))
                print('-----save weights-----')

            print(total_steps)

            # swap observation
            observation = observation_
            episode_steps += 1
            total_steps += 1

            if done:
                print('episode ', i_episode, ' finished')
                steps_episode.append(episode_steps)
                steps_total.append(total_steps)
                episodes.append(i_episode)
                break

            if 'sarsa' in model:
                action = action_

    return [np.vstack((episodes, steps_total)), np.vstack((episodes, steps_episode))]


def main(model):
    env = gym.make('CartPole-v0')
    # build network.
    n_actions = env.action_space.n
    n_features = env.observation_space.high.size
    if model == 'dqn_2013':
        from brains.dqn_2013 import DeepQNetwork
        from games.CartPole_v0.network_dqn_2013 import build_network
        inputs, outputs = build_network(n_features, n_actions, lr=0.01)
        # get the DeepQNetwork Agent
        RL = DeepQNetwork(
            n_actions=n_actions,
            n_features=n_features,
            eval_net_input=inputs[0],
            q_target=inputs[1],
            q_eval_net_out=outputs[0],
            loss=outputs[1],
            train_op=outputs[2],
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=100,
            memory_size=2000,
            e_greedy_increment=0.001,
            output_graph=True,
        )
        saver, load_step = store_parameters(RL.sess)
    else:  # dqn_2015
        from brains.dqn_2015 import DeepQNetwork
        from games.CartPole_v0.network_dqn_2015 import build_network
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
        saver, load_step = store_parameters(RL.sess)

    # Calculate running time
    start_time = time.time()

    run_cartpole(env, RL, model, saver, load_step)

    end_time = time.time()
    running_time = (end_time - start_time) / 60

    fo = open("./logs/running_time.txt", "w")
    fo.write(str(running_time) + "minutes")
    fo.close()


if __name__ == '__main__':
    # # change different models here:
    # dqn_2013, dqn_2015,...
    main(model='dqn_2013')
