from collections import deque
import random
import numpy as np

from keras import Sequential
from keras.layers import Dense, Dropout

# Deep Q-learning Agent
from keras.optimizers import Adam


class DQNAgent:
    def __init__(self, state_size, action_size, weights=None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=4000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.9  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99 # decay epsilon to go towards higher exploitation rate after network has learned a lot
        self.learning_rate = 0.0005
        self.model = self._build_model(weights)
    def _build_model(self, weights):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        # input_dim is just the size of the input nodes before this layer.
        model.add(Dense(80, input_dim=self.state_size, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(80, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.15))
        # TODO Changed for softmax!
        model.add(Dense(self.action_size, activation='softmax'))
        # L2 loss (MSE) (predicted - actual)**2 -> predicted = Q(s,a), actual = target = R(s,a) = y*Max(Q(s`,a`))
        # learning rate used to update gradient from loss function -> not same as how learning rate is used in
        # Q learning algorithm?
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        if weights:
            model.load_weights(weights)
        return model
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def predict(self, state):
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward

            # if done only take reward as target. Max value of new Q values are not necessary
            if not done:
              #   target (is reward after action taken + max Q value. For max value, Q values are predicted for next state). Used to calculate loss between current value that
              # is predicted in the modelfit() with old state
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])
            #     same q values that model.fit(state) will give
            target_f = self.model.predict(state)
            #   only value that changes out of all the q values for all actions is the Q value for the action with the highest Q value.
            # This is updated with the newly predicted Q value for it (reward + Max Q out of new state options)
            target_f[0][action] = target
            # loss will be calculated between all 4 q values for all 4 states predicted with model.fit(state) and all
            # 4 values for all 4 states in target_f. Only one state in target_f values was updated so only loss of that
            # one will be calculated. See model.compile()
            # Network predicts Q values for one state (for all actions in that state) -> need to convergce to predicted
            # Q value to be same as actual value (Reward + MAXQ)
            self.model.fit(state, target_f, epochs=1, verbose=0)
        #     TODO changed for constant epsilon decay rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay