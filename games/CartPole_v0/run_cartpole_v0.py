import gym
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Set log level: only output error.
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only use #0 GPU.
import time

from brains.dqn_2015 import DeepQNetwork
from games.CartPole_v0.network import build_network
from games.CartPole_v0.hyperparameters import MAX_EPISODES
from games.CartPole_v0.hyperparameters import REPLY_START_SIZE
from games.CartPole_v0.hyperparameters import UPDATE_FREQUENCY


def run_cartpole(env, RL):
    step = 0
    count_list = []
    for episode in range(MAX_EPISODES):
        print('episode:' + str(episode))
        # initial observation
        observation = env.reset()
        # counter for one episode
        count = 0

        while True:
            env.render()
            # RL choose action based on observation
            action = RL.choose_action(observation, step)
            # RL take action and get next observation and reward
            observation_, reward, done, info = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            # 'step % UPDATE_FREQUENCY == 0' is for frame-skipping technique.
            if (step > REPLY_START_SIZE) and (step % UPDATE_FREQUENCY == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

            # counter for one episode.
            count += 1

        print('episode:' + str(episode) + ' | ' + str(count))
        count_list.append(count)

    fo = open("./logs/counts.txt", "w")
    fo.write(str(count_list))
    fo.close()


def main():
    env = gym.make('CartPole-v0')
    # build network.
    n_actions = env.action_space.n
    n_features = len(env.observation_space.high)
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
        replace_target_iter=200,
        memory_size=2000,
        e_greedy_increment=0.01,
        output_graph=True,
    )
    # Calculate running time
    start_time = time.time()

    run_cartpole(env, RL)

    end_time = time.time()
    running_time = (end_time - start_time) / 60

    fo = open("./logs/running_time.txt", "w")
    fo.write(str(running_time) + "minutes")
    fo.close()


if __name__ == '__main__':
    main()
