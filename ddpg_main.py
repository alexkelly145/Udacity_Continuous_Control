#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import dependencies
from unityagents import UnityEnvironment
import numpy as np
from collections import deque, defaultdict,namedtuple
import pandas as pd
import matplotlib.pyplot as plt
import torch


# In[ ]:


import os
os.chdir('C:\\Users\\Admin\\Desktop\\Udacity\\deep-reinforcement-learning\\p2_continuous-control')


# In[ ]:


# Initialise environment
env = UnityEnvironment(file_name='Reacher.exe')


# In[ ]:


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# In[ ]:


from ddpg_agent import Agent

# Initialise Agent and define hyperparameters
agent = Agent(state_size = 37, action_size = brain.vector_action_space_size, seed = 0, buffer_size = 20000, 
              batch_size = 64, learning_rate_actor = 1e-3, learning_rate_critic = 1e-3, weight_decay = 0)


# In[ ]:


from replay import ExperienceReplay

def ddpg(n_episodes=20000, eps_start=1.0, eps_end=0.01, eps_decay=0.99):
    
    scores = []  # Save scores in a list                      
    scores_window = deque(maxlen=100)  # Sliding window list with max length of 100 
    eps = eps_start                    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        # Keeps track of time step
        t = 0
        while True:
            # choosing next action
            action = agent.select_action(state, eps)
            action = action.astype(int)
            # running next action through env to get next_state, reward, done
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]                   
            done = env_info.local_done[0]
            # Model weights are updated every X time steps
            agent.step(state, action, reward, next_state, done, t, 0.9)
            # S' -> S
            state = next_state
            score += reward
            # Adds 1 to time step
            t+=1
            # Break while loop if done
            if done:
                break 
        scores_window.append(score)       
        scores.append(score) 
        # Decaying epsilon value
        eps = max(eps_end, eps_decay*eps) 
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=30:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break
    return scores

scores = ddpg()

