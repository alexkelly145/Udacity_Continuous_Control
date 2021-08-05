import numpy as np
import random

from collections import namedtuple, deque, defaultdict

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ExperienceReplay():
    def __init__(self, buffer_size, batch_size):
        
        # Hyperparameters
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        # Init replay buffer and priorities as deque lists
        self.buffer = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])      
               
    def add(self, state, action, reward, next_state, done):
        # Add an experience to the replay buffer
        self.buffer.append(self.experience(state, action, reward, next_state, done))
            
    def sample(self):
        # Randomly select experiences from the buffer
        idx = np.random.choice(len(self.buffer), self.batch_size)
        experiences = [self.buffer[i] for i in idx]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.buffer)
    
    

class PrioritizedExperienceReplay():
    def __init__(self, buffer_size, batch_size, a):
        
        # Hyperparameters
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.a = a
              
        # Init replay buffer and priorities as deque lists
        self.buffer = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.priorities = deque(np.zeros(buffer_size, np.float32), maxlen=buffer_size)
        
        # Index to keep where in the lists the experiences are being saved
        self.index = 0
        
               
    def add(self, state, action, reward, next_state, done):
       
        priorityAddArray = list(self.priorities)

        # Adding values to priority so the list starts as non zero
        priority = max(priorityAddArray) if self.buffer else 1.0

        # Add experience and priority to lists
        self.buffer.append(self.experience(state, action, reward, next_state, done))
        self.priorities.append(priority)

        # Add 1 to index every time an experience is added up to the buffer_size
        self.index = min((self.index+1), self.buffer_size)

    def sample(self, beta):
        
        prioritySampleArray = np.asarray(list(self.priorities), dtype = float)

        # If buffer is full then use the full list else a slice of the list is needed
        if len(self.buffer) == self.buffer_size:
            samplePriorities = prioritySampleArray
        else:
            samplePriorities = prioritySampleArray[-self.index:]

        # Calculate Sampling Probabilities
        sampleProbs = (samplePriorities)**self.a / (samplePriorities**self.a).sum()

        # Choose experiences based on Probabilities
        idx = np.random.choice(len(self.buffer), self.batch_size, p = sampleProbs, replace = True)
        experiences = [self.buffer[i] for i in idx]

        # Calculate importance sampling weights for experiences 
        weights = (len(self.buffer) * sampleProbs[idx])**(-beta)
        weights = weights / weights.max()

        # Separate out S, A, R, S' and dones from experiences and transform them to a torch tensor
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones), idx, weights

     
    def update_priorities(self, prior, idx):
        for i in prior:
            self.priorities.append(i)

    def __len__(self):
        return len(self.buffer)

