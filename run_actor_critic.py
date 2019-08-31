import gym
from actor_critic import Agent

#from utils import plotLearning
import numpy as np

if __name__ == '__main__':
    agent = Agent(alpha=0.00001, beta=0.00005)

    env = gym.make('LunarLander-v2')
    score_history = []
    num_episodes = 10

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
