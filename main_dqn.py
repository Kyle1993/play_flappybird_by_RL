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
train_record = {}

for episode in count(1):
    print('')
    print(episode)
    print(agent.epsilon)
    episode_reward = 0

    state,reward,_ = env.init()

    episode_reward += reward
    for step in count(1):
        action = agent.select_action(state)
        state_,reward,done = env.step(action)
        episode_reward += reward
        agent.memory.append(state,action,state_,reward,not done)
        agent.train()
        state = state_
        if done:
            print(episode_reward)
            train_record[episode] = episode_reward
            break

    if episode % 100 == 0:
        with open('record.pkl','wb') as f:
            pickle.dump(train_record,f)
        agent.save(episode)
