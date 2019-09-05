from keras.layers import Dense, Input
from keras.models import Model, load_model
from keras.optimizers import Adam
import keras.backend as K
import numpy as np

#so tf doesn't take up entire gpu memory
import tensorflow as tf
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#set precision
import keras
keras.backend.set_floatx('float32')

class Agent():
    def __init__(self, ALPHA, GAMMA=0.99, n_actions=4,
                layer1_size=16, layer2_size=16, input_dims=8,
                fname='reinforce.h5'):
        self.gamma = GAMMA
        self.lr = ALPHA
        #discounted sum of rewards
        self.G = 0
        self.input_dims = input_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = n_actions
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

        self.policy, self.predict = self._build_policy_network()

        self.action_space = [i for i in range(n_actions)]

        self.model_file = fname

    def _build_policy_network(self):
        input = Input(shape=(self.input_dims,))
        advantages = Input(shape=[1])
        dense1 = Dense(self.fc1_dims, activation='relu')(input)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)

        #probability output
        probs = Dense(self.n_actions, activation='softmax')(dense2)

        def custom_loss(y_true, y_pred):
            '''custom loss function'''
            out = K.clip(y_pred*y_true, 1e-8, 1-1e-8)
            log_lik = K.log(out)

            return K.sum(-log_lik*advantages)

        policy = Model(input=[input, advantages], output=[probs])
        policy.compile(optimizer=Adam(lr=self.lr), loss=custom_loss)

        predict = Model(input=[input], output=[probs])
        return policy, predict

    def choose_action(self, observation):
        state = observation.reshape(1,*observation.shape)

        #get probability and sample it to get action
        probability = self.predict.predict(state)[0]
        action = np.random.choice(self.action_space, p=probability)
        return action

    def store_transition(self, observation, action, reward):
        self.action_memory.append(action)
        self.state_memory.append(observation)
        self.reward_memory.append(reward)

    def learn(self):
        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        reward_memory = np.array(self.reward_memory)

        #get one hot encoding
        actions = np.zeros([len(action_memory), self.n_actions])
        actions[np.arange(len(action_memory)), action_memory] = 1

        G = np.zeros_like(reward_memory)
        for t in range(len(reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k]*discount
                discount *= self.gamma
            G[t] = G_sum

        mean = np.mean(G)
        std = np.std(G) if np.std(G)>0 else 1
        self.G = (G-mean)/std

        cost = self.policy.train_on_batch([state_memory, self.G], actions)

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

    def save_model(self):
        self.predict.save(self.model_file)

    def load_model(self):
        self.predict = load_model(self.model_file)
