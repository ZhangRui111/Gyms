import errno
import gym
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Set log level: only output error.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only use #0 GPU.
import time
import tensorflow as tf
import matplotlib as mlp
mlp.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from games.MountainCar_v0.hyperparameters import MAX_EPISODES
from games.MountainCar_v0.hyperparameters import REPLY_START_SIZE
from games.MountainCar_v0.hyperparameters import UPDATE_FREQUENCY
from games.MountainCar_v0.hyperparameters import WEIGHTS_SAVER_ITER
from games.MountainCar_v0.hyperparameters import OUTPUT_SAVER_ITER
from games.MountainCar_v0.hyperparameters import SAVED_NETWORK_PATH
from games.MountainCar_v0.hyperparameters import LOGS_DATA_PATH


def plot_results(his_natural, his_prio):
    # compare based on first success
    plt.plot(his_natural[0, :], his_natural[1, :] - his_natural[1, 0], c='b', label='natural DQN')
    plt.plot(his_prio[0, :], his_prio[1, :] - his_prio[1, 0], c='r', label='DQN with prioritized replay')
    plt.legend(loc='best')
    plt.ylabel('total training time')
    plt.xlabel('episode')
    plt.grid()
    plt.show()


def restore_parameters(sess, model):
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(SAVED_NETWORK_PATH + model + '/')
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


def run_mountaincar(env, RL, model, saver, load_step):
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

            if done:
                reward = 10

            if 'sarsa' in model:
                action_ = RL.choose_action(observation_, total_steps)
                RL.store_transition(observation, action, reward, observation_, action_)
            else:
                RL.store_transition(observation, action, reward, observation_)

            if total_steps > RL.memory_size:
                if total_steps > REPLY_START_SIZE:
                    if total_steps % UPDATE_FREQUENCY == 0:
                        RL.learn()

                    if total_steps % WEIGHTS_SAVER_ITER == 0:
                        saver.save(RL.sess, SAVED_NETWORK_PATH + model + '/' + '-' + model + '-' +
                                   str(total_steps + load_step))
                        print('-----save weights-----')

                    if total_steps % OUTPUT_SAVER_ITER == 0:
                        filename1 = LOGS_DATA_PATH + model + '/steps_total.txt'
                        if not os.path.exists(os.path.dirname(filename1)):
                            try:
                                os.makedirs(os.path.dirname(filename1))
                            except OSError as exc:  # Guard against race condition
                                if exc.errno != errno.EEXIST:
                                    raise
                        fp1 = open(filename1, "w")
                        fp1.write(str(np.vstack((episodes, steps_total))))
                        fp1.close()
                        filename2 = LOGS_DATA_PATH + model + '/steps_episode.txt'
                        if not os.path.exists(os.path.dirname(filename2)):
                            try:
                                os.makedirs(os.path.dirname(filename2))
                            except OSError as exc:  # Guard against race condition
                                if exc.errno != errno.EEXIST:
                                    raise
                        fp2 = open(filename2, "w")
                        fp2.write(str(np.vstack((episodes, steps_episode))))
                        fp2.close()
                        print('-----save outputs-----')

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
    env = gym.make('MountainCar-v0')
    # env.unwrapped can give us more information.
    env = env.unwrapped
    # build network.
    n_actions = env.action_space.n
    n_features = env.observation_space.high.size

    if model == 'pri_dqn':
        from brains.pri_dqn import DeepQNetwork
        from games.MountainCar_v0.network_pri_dqn import build_network
        inputs, outputs, weights = build_network(n_features, n_actions, lr=0.01)
        # get the DeepQNetwork Agent
        RL = DeepQNetwork(
            n_actions=3,
            n_features=2,
            eval_net_input=inputs[0],
            target_net_input=inputs[1],
            q_target=inputs[2],
            ISWeights=inputs[3],
            q_eval_net_out=outputs[0],
            loss=outputs[1],
            train_op=outputs[2],
            q_target_net_out=outputs[3],
            abs_errors=outputs[4],
            e_params=weights[0],
            t_params=weights[1],
            e_greedy_increment=0.00005,
            output_graph=True,
        )
    else:  # double_dqn
        from brains.double_dqn import DeepQNetwork
        from games.MountainCar_v0.network_double_dqn import build_network
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
            learning_rate=0.005,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=500,
            memory_size=10000,
            e_greedy_increment=0.00005,
            output_graph=True,
        )

    saver, load_step = restore_parameters(RL.sess)

    # Calculate running time
    start_time = time.time()

    his_prio = run_mountaincar(env, RL, model, saver, load_step)
    print(his_prio)  # his_prio can be plotted by plot_results()

    end_time = time.time()
    running_time = (end_time - start_time) / 60

    filename = LOGS_DATA_PATH + model + "/running_time.txt"
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    fo = open(filename, "w")
    fo.write(str(running_time) + "minutes")
    fo.close()


if __name__ == '__main__':
    # # change different models here:
    # pri_dqn, double_dqn...
    main(model='double_dqn')
