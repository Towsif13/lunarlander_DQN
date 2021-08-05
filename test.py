from agent import Agent
import gym
import random
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

env = gym.make('LunarLander-v2')
#env = gym.make('CartPole-v0')
env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)


agent = Agent(state_size=8, action_size=4, seed=0)

# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

for i in range(100):
    state = env.reset()
    for j in range(200):
        action = agent.act(state)
        env.render()
        state, reward, done, _ = env.step(action)
        if done:
            break

env.close()
