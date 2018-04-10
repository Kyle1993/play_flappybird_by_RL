from game.my_flappy_bird_env import flappy_bird_env
import numpy as np
import random
from itertools import count
from DQN_model import Agent
from config import DQN_Config
import pickle

env = flappy_bird_env()

c,h,w = env.state_shape()
action_dim = env.action_shape()[0]
config = DQN_Config()

agent = Agent(c,action_dim,config.memory_size,config.batch_size,config.lr,config.gamma,config.gpu,config.epsilon_decay)
agent.load(1100)

for episode in count(1):
    print('')
    print(episode)
    episode_reward = 0

    state,reward,_ = env.init()

    episode_reward += reward
    for step in count(1):
        action = agent.select_action(state,isTrain=False,decay=False)
        state_,reward,done = env.step(action)
        episode_reward += reward
        state = state_
        if done:
            print(episode_reward)
            break

