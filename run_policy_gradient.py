import gym
import matplotlib.pyplot as plt
import numpy as np
from policy_gradient import Agent

import keras
import keras.backend as K

def custom_loss(y_true, y_pred):
    '''custom loss function'''
    out = K.clip(y_pred*y_true, 1e-8, 1-1e-8)
    log_lik = K.log(out)

    return K.sum(-log_lik*advantages)

def train():
    '''
    reinforce algorithm
    run an episode, then train with collected transitions
    '''
    agent = Agent(ALPHA=0.005, input_dims=8, GAMMA=0.99, n_actions=4,
                layer1_size=64, layer2_size=64)
    env = gym.make('LunarLander-v2')
    score_history = []

    n_episodes = 380

    for i in range(n_episodes):
        done = False
        score = 0
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)

            agent.store_transition(observation, action, reward)

            observation = observation_
            score += reward

        score_history.append(score)
        agent.learn()
        print('episode',i,'score %.1f' % score,
            'average_score %.1f' % np.mean(score_history[-100:]))

    plt.plot(score_history)
    plt.show()


    agent.save_model()

def run_agent(num_episodes=3):
    '''
    run a trained agent
    needs a full path to a saved model
    '''
    agent = Agent(ALPHA=0.005, input_dims=8, GAMMA=0.99, n_actions=4,
                layer1_size=64, layer2_size=64)
    agent.policy = keras.models.load_model('reinforce.h5')

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


if __name__=='__main__':
    run_agent()
    #train()
