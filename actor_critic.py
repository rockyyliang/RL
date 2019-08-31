'''
from Phil Tabor's actor critic tutorial
'''
from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam

import numpy as np

class Agent(object):
    def __init__(self, alpha, beta, gamma=0.99, n_actions=4,
                layer1_size=1024, layer2_size=512, input_dims=8):
        '''
        AC agent that contains all networks
        '''
        #agent parameters
        #for bellman
        self.gamma = gamma
        #actor learning rate
        self.alpha = alpha
        #critic learning rate
        self.beta = beta
        self.input_dims = input_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = n_actions

        #build network and define vector for action space
        self.actor, self.critic, self.policy = self._build_actor_critic_network()
        self.action_space = [i for i in range(self.n_actions)]

    def _build_actor_critic_network(self):
        '''
        builds agent networks
        actor net approximates policy
        critic net calcs value of action 'V'

        actor and critic nets share middle layers, actor outputs probability,
        while critic outputs value. for simple envs, you can actually define
        2 totally separate networks (phil's words)
        '''
        input = Input(shape=(self.input_dims,))

        #used in the loss function, aka advantage
        delta = Input(shape=[1])

        #shared layers
        dense1 = Dense(self.fc1_dims, activation='relu')(input)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)

        #prob output uses softmax activation
        probs = Dense(self.n_actions, activation='softmax')(dense2)

        #critic net output
        values = Dense(1, activation='linear')(dense2)


        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            print(out)
            log_lik = y_true*K.log(out)

            return K.sum(-log_lik*delta)

        #actor, used for training
        actor = Model(input=[input, delta], output=[probs])
        actor.compile(optimizer=Adam(lr=self.alpha), loss=custom_loss)

        #critic network
        critic = Model(input=[input], output=[values])
        critic.compile(optimizer=Adam(lr=self.beta), loss='mean_squared_error')

        #prediction only actor network, no compilation needed (no training)
        policy = Model(input=[input], output=[probs])

        return actor, critic, policy

    def choose_action(self, observation):
        #add new axis to obs vector
        state = observation[np.newaxis, :]

        #get probs
        probabilities = self.policy.predict(state)[0]
        #print(probabilities)
        #sample probability generated actor
        action = np.random.choice(self.action_space, p=probabilities)

        return action

    def learn(self, state, action, reward, state_, done):
        state = state[np.newaxis, :]
        state_ = state_[np.newaxis, :]

        critic_value_ = self.critic.predict(state_)
        critic_value = self.critic.predict(state)

        #compute target and advantage
        target = reward + self.gamma*critic_value_*(1-int(done))
        delta = target - critic_value

        actions = np.zeros([1, self.n_actions])
        actions[0, action] = 1.0

        self.actor.fit([state, delta], actions, verbose=0)
        self.critic.fit(state, target, verbose=0)
