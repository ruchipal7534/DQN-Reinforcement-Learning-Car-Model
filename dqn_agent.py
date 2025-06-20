import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque
from constants import device
from dqn_network import DQNetwork

class DQNAgent:
    def __init__(self, state_size, action_size=9, lr=0.0001, gamma=0.99, tau=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.q_network = DQNetwork(state_size, output_dim=action_size*2)
        self.target_network = DQNetwork(state_size, output_dim=action_size*2)
        self.update_target_network(tau=1.0) 
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=1e-5)
        self.memory = deque(maxlen=50000)
        self.priority_memory = deque(maxlen=10000)
        self.batch_size = 64
        self.epsilon = 1.0
        self.epsilon_min = 0.1 
        self.epsilon_decay = 0.999 
        self.loss_history = deque(maxlen=100)
        self.training_steps = 0
        
    def update_target_network(self, tau=None):
        if tau is None:
            tau = self.tau
            
        for target_param, param in zip(self.target_network.parameters(),
                                     self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)
            
    def remember(self, state, action_idx, reward, next_state, done):
        experience = (state, action_idx, reward, next_state, done)
        self.memory.append(experience)
        
        if abs(reward) > 2.0 or done:
            self.priority_memory.append(experience)
        
    def act(self, state):
        return self.q_network.act(state, self.epsilon)
        
    def replay(self):
        if len(self.memory) < self.batch_size * 2:
            return 0
            
        regular_size = int(self.batch_size * 0.7)
        priority_size = self.batch_size - regular_size
        
        batch = []
        if len(self.memory) >= regular_size:
            batch.extend(random.sample(self.memory, regular_size))
        if len(self.priority_memory) >= priority_size:
            batch.extend(random.sample(self.priority_memory, priority_size))
        else:
            batch.extend(random.sample(self.memory, self.batch_size - len(batch)))
            
        states = np.array([e[0] for e in batch])
        actions_steer = np.array([e[1][0] for e in batch])
        actions_accel = np.array([e[1][1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        states = torch.FloatTensor(states).to(device)
        actions_steer = torch.LongTensor(actions_steer).to(device)
        actions_accel = torch.LongTensor(actions_accel).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        current_steer_q, current_accel_q = self.q_network(states)
        steer_q = current_steer_q.gather(1, actions_steer.unsqueeze(1)).squeeze(1)
        accel_q = current_accel_q.gather(1, actions_accel.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            online_next_steer_q, online_next_accel_q = self.q_network(next_states)
            best_steer_actions = online_next_steer_q.argmax(1)
            best_accel_actions = online_next_accel_q.argmax(1)
            
            target_next_steer_q, target_next_accel_q = self.target_network(next_states)
            max_next_steer_q = target_next_steer_q.gather(1, best_steer_actions.unsqueeze(1)).squeeze(1)
            max_next_accel_q = target_next_accel_q.gather(1, best_accel_actions.unsqueeze(1)).squeeze(1)
            
            target_steer = rewards + (1 - dones) * self.gamma * max_next_steer_q
            target_accel = rewards + (1 - dones) * self.gamma * max_next_accel_q
            
        loss_steer = F.smooth_l1_loss(steer_q, target_steer)
        loss_accel = F.smooth_l1_loss(accel_q, target_accel)
        loss = loss_steer + loss_accel
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.update_target_network()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        self.loss_history.append(loss.item())
        self.training_steps += 1
            
        return loss.item()
        
    def save(self, filepath):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'loss_history': list(self.loss_history),
            'training_steps': self.training_steps
        }, filepath)
        
    def load(self, filepath):
        if not os.path.exists(filepath):
            print(f"Model file {filepath} not found")
            return False
            
        checkpoint = torch.load(filepath, map_location=device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', 0.05)
        self.training_steps = checkpoint.get('training_steps', 0)
        
        print(f"Model loaded from {filepath}")
        return True