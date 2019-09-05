import keras
import gym
from actor_critic import DiscreteAgent, DiscreteReplayAgent

#from utils import plotLearning
import numpy as np
import matplotlib.pyplot as plt
import argparse

#so tf doesn't take up entire gpu memory
import tensorflow as tf
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#set precision
keras.backend.set_floatx('float32')

def train(num_episodes=100):
    '''
    training loop
    '''
    agent = DiscreteAgent(alpha=0.00001, beta=0.00005)

    env = gym.make('LunarLander-v2')
    score_history = []

    for i in range(num_episodes):
        done = False
        score = 0
        observation = env.reset()

        while not done:
            #env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)

            agent.learn(observation, action, reward, observation_, done)
            observation = observation_
            score += reward

        #env.close()
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        print('episode', i, 'score %.2f, average score %.2f' % \
            (score, avg_score))

    plt.plot(score_history)
    plt.title('Score History')
    plt.show()
    agent.policy.save('lander.h5')

def train_with_replay(num_episodes=100):
    '''
    training loop
    '''
    agent = DiscreteReplayAgent(alpha=0.00001, beta=0.00005)

    env = gym.make('LunarLander-v2')
    score_history = []

    for i in range(num_episodes):
        done = False
        score = 0
        observation = env.reset()

        while not done:
            #env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)

            #add to memory and run training
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()

            observation = observation_
            score += reward

        #env.close()
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        print('episode', i, 'score %.2f, average score %.2f' % \
            (score, avg_score))

    plt.plot(score_history)
    plt.title('Score History')
    plt.show()
    agent.policy.save('lander.h5')

def run_agent(model_name, num_episodes=3):
    '''
    run a trained agent
    needs a full path to a saved model
    '''
    agent = DiscreteReplayAgent(alpha=0.00001, beta=0.00005)
    agent.load_policy(model_name)

    env = gym.make('LunarLander-v2')

    for i in range(num_episodes):
        done = False
        score = 0
        observation = env.reset()

        while not done:
            env.render()
            action = agent.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            #transfer observation
            observation = observation_

            #add to score
            score += reward

        print(score)
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a trained agent or run training loop')
    parser.add_argument('-t','--train', type=bool, nargs='?', const=True, help='use this if you want to train')
    args = parser.parse_args()

    if (args.train == None) or (args.train==False):
        print('run trained agent')
        run_agent('lander.h5')
    else:
        print('run training')
        train_with_replay()
