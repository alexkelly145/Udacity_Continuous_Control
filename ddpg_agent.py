# Import dependencies
import numpy as np
import random
import copy
from collections import namedtuple, deque, defaultdict

import torch
import torch.nn.functional as F
import torch.optim as optim

# Importing classes from other py files
from model import Actor, Critic
from replay import ExperienceReplay, PrioritizedExperienceReplay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():

    def __init__(self, state_size, action_size,buffer_size, batch_size, seed, 
                 learning_rate_actor, learning_rate_critic, e, beta, a):
    
        # Hyperparameters
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = seed
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=learning_rate_critic)

        # Noise process
        self.noise = OUNoise(action_size, seed)
        self.decay = 0.99
        
        # Replay memory
        self.replay = ExperienceReplay(buffer_size, batch_size) 
        self.prioritizedreplay = PrioritizedExperienceReplay(buffer_size, batch_size, a) 
        self.e = e
        self.beta = beta
            
    
    def step(self, states, actions, rewards, next_states, dones, time_step, update_value, gamma):
        
        # Experience is added to the replay buffer        
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.replay.add(state, action, reward, next_state, done)
        
        # When enough experiences are collected in the replay buffer and every X time 
        # steps the memory is sampled and the networks are trained
        
        if (time_step % update_value == 0) and len(self.replay) > self.batch_size:
            for _ in range(10):
                experiences = self.replay.sample()
                self.learn(experiences, gamma)

    
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        # Foward pass
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            #adding noise to the action with a decay to allow the agent to explore-exploit
            action += self.decay*self.noise.sample().reshape((-1, 4))
            self.decay *= self.decay
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, 1e-3)
        self.soft_update(self.actor_local, self.actor_target, 1e-3) 
        
    def learn_per(self, experiences, gamma, idx, weights):
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        
        # Calculating TD error and priority
        errors = abs(Q_targets - Q_expected)
        errors = errors + self.e
        priorities = errors.cpu().detach().numpy().flatten()
        
        # Updating priorities in the replay buffer with the TD error
        self.replay.update_priorities(priorities, idx)

        critic_loss = torch.mean((errors * torch.from_numpy(weights).to(device))**2)
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, 1e-3)
        self.soft_update(self.actor_local, self.actor_target, 1e-3) 
        
   
    def soft_update(self, local_model, target_model, tau):
       
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        
    def update_target(self, local_model, target_model):
       
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

class OUNoise():
    # Ornstein-Uhlenbeck process

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        # Initialize parameters 
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        # Reset the internal state
        self.state = copy.copy(self.mu)

    def sample(self):
        # Update internal state and return it as a noise sample
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
    






