import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicygi
from rl.memory import SequentialMemory

ENV_Name='CartPole-v0'
env = gym.make(ENV_Name)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(10, ) + env.observation_space.shape))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=10)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
dqn.compiled(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env,nb_steps=20000,visualize=True, verbose=2)

his = dqn.test(env, nb_episodes=100, visualize=True).history['episode_reward']
print("Average reward over", len(his), "episodes: ", np.sum(his)/len(his))
