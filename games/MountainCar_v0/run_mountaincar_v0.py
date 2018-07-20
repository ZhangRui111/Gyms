import gym
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Set log level: only output error.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only use #0 GPU.
import time
import matplotlib as mlp
mlp.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from games.MountainCar_v0.hyperparameters import MAX_EPISODES
from games.MountainCar_v0.hyperparameters import REPLY_START_SIZE
from games.MountainCar_v0.hyperparameters import UPDATE_FREQUENCY


def plot_results(his_natural, his_prio):
    # compare based on first success
    plt.plot(his_natural[0, :], his_natural[1, :] - his_natural[1, 0], c='b', label='natural DQN')
    plt.plot(his_prio[0, :], his_prio[1, :] - his_prio[1, 0], c='r', label='DQN with prioritized replay')
    plt.legend(loc='best')
    plt.ylabel('total training time')
    plt.xlabel('episode')
    plt.grid()
    plt.show()


def run_mountaincar(env, RL):
    total_steps = 0
    steps = []
    episodes = []
    for i_episode in range(MAX_EPISODES):
        observation = env.reset()
        while True:
            # env.render()

            action = RL.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            if done: reward = 10

            RL.store_transition(observation, action, reward, observation_)

            if total_steps > RL.memory_size:
                if (total_steps > REPLY_START_SIZE) and (total_steps % UPDATE_FREQUENCY == 0):
                    RL.learn()

            if done:
                print('episode ', i_episode, ' finished')
                steps.append(total_steps)
                episodes.append(i_episode)
                break

            observation = observation_
            total_steps += 1

    return np.vstack((episodes, steps))


def main(model):
    env = gym.make('MountainCar-v0')
    # env.unwrapped can give us more information.
    env = env.unwrapped
    # build network.
    n_actions = env.action_space.n
    n_features = env.observation_space.high.size

    # if model == 'pri_dqn':
    #     from brains.dqn_2013 import DeepQNetwork
    #     from games.CartPole_v0.network_dqn_2013 import build_network
    #     inputs, outputs = build_network(n_features, n_actions, lr=0.01)
    #     # get the DeepQNetwork Agent
    #     RL = DeepQNetwork(
    #         n_actions=n_actions,
    #         n_features=n_features,
    #         eval_net_input=inputs[0],
    #         q_target=inputs[1],
    #         q_eval_net_out=outputs[0],
    #         loss=outputs[1],
    #         train_op=outputs[2],
    #         learning_rate=0.01,
    #         reward_decay=0.9,
    #         e_greedy=0.9,
    #         replace_target_iter=100,
    #         memory_size=2000,
    #         e_greedy_increment=0.001,
    #         output_graph=True,
    #     )
    # else:  # other model
    #     pass

    from brains.pri_dqn import DeepQNetwork
    from games.MountainCar_v0.network_pri_dqn import build_network
    inputs, outputs, weights = build_network(n_features, n_actions, lr=0.01)
    # get the DeepQNetwork Agent
    RL_prio = DeepQNetwork(
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

    # Calculate running time
    start_time = time.time()

    print(run_mountaincar(env, RL_prio))

    end_time = time.time()
    running_time = (end_time - start_time) / 60

    fo = open("./logs/running_time.txt", "w")
    fo.write(str(running_time) + "minutes")
    fo.close()


if __name__ == '__main__':
    # # change different models here:
    # pri_dqn, ...
    main(model='pri_dqn')
