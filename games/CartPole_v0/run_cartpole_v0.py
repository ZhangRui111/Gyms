import errno
import gym
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Set log level: only output error.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only use #0 GPU.
import time
import tensorflow as tf

from games.CartPole_v0.hyperparameters import Hyperparameters
from utils.write_to_file import write_to_file_running_time
from utils.write_to_file import write_to_file_running_steps


def restore_parameters(sess, model):
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(Hp.SAVED_NETWORK_PATH + model + '/')
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

    for i_episode in range(Hp.MAX_EPISODES):
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

            if total_steps > RL.memory_size:
                if total_steps > Hp.REPLY_START_SIZE:
                    if total_steps % Hp.UPDATE_FREQUENCY == 0:
                        RL.learn()

                    if total_steps % Hp.WEIGHTS_SAVER_ITER == 0:
                        saver.save(RL.sess, Hp.SAVED_NETWORK_PATH + model + '/' + '-' + model + '-' +
                                   str(total_steps + load_step))
                        print('-----save weights-----')

                    if total_steps % Hp.OUTPUT_SAVER_ITER == 0:
                        filename1 = Hp.LOGS_DATA_PATH + model + '/steps_total.txt'
                        write_to_file_running_steps(filename1, str(np.vstack((episodes, steps_total))))
                        filename2 = Hp.LOGS_DATA_PATH + model + '/steps_episode.txt'
                        write_to_file_running_steps(filename2, str(np.vstack((episodes, steps_episode))))
                        print('-----save outputs-----')

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
            replay_start_size=Hp.REPLY_START_SIZE,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=100,
            memory_size=2000,
            e_greedy_increment=0.001,
            output_graph=True,
        )
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
            replay_start_size=Hp.REPLY_START_SIZE,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=100,
            memory_size=2000,
            e_greedy_increment=0.001,
            output_graph=True,
        )

    saver, load_step = restore_parameters(RL.sess, model)
    # Calculate running time
    start_time = time.time()

    run_cartpole(env, RL, model, saver, load_step)

    end_time = time.time()
    running_time = (end_time - start_time) / 60

    filename = Hp.LOGS_DATA_PATH + model + "/running_time.txt"
    write_to_file_running_time(filename, str(running_time))


if __name__ == '__main__':
    Hp = Hyperparameters()
    # # change different models here:
    # dqn_2013, dqn_2015,...
    main(model='dqn_2015')
