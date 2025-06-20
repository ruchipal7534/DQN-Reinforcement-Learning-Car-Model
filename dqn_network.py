import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from constants import device

class DQNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=10):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.steer_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim // 2)
        )
        
        self.accel_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim // 2)
        )
        
        self.apply(self._init_weights)
        self.to(device)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            single_input = True
        else:
            single_input = False
        
        h1 = F.relu(self.ln1(self.fc1(x)))
        h2 = F.relu(self.ln2(self.fc2(h1)))
        h2 = self.dropout(h2)
        h3 = F.relu(self.ln3(self.fc3(h2) + h1)) 
        value = self.value_head(h3)
        steer_adv = self.steer_head(h3)
        accel_adv = self.accel_head(h3)
        steer_q = value + (steer_adv - steer_adv.mean(dim=1, keepdim=True))
        accel_q = value + (accel_adv - accel_adv.mean(dim=1, keepdim=True))

        if single_input:
            steer_q = steer_q.squeeze(0)
            accel_q = accel_q.squeeze(0)
            
        return steer_q, accel_q
        
    def act(self, state, epsilon=0.0):
        if random.random() < epsilon:
            steer_idx = random.randint(0, 4)
            accel_idx = random.randint(1, 4) 
            if len(state) > 7:
                front_sensor = state[7] 
                if front_sensor < 0.3:
                    accel_idx = random.randint(0, 1) 
                    left_sensor = state[5]
                    right_sensor = state[9]
                    if left_sensor > right_sensor:
                        steer_idx = random.randint(0, 1) 
                    else:
                        steer_idx = random.randint(3, 4) 
        else:
            self.eval()  
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(device)
                steer_q, accel_q = self.forward(state_tensor)
                steer_idx = torch.argmax(steer_q).item()
                accel_idx = torch.argmax(accel_q).item()
            self.train()  
                
        steer = (steer_idx - 2) / 2.0  
        accel = (accel_idx - 2) / 2.0  
        
        if len(state) > 7:
            front_sensor = state[7]
            if front_sensor < 0.2:
                left_space = state[5]
                right_space = state[9]
                
                if left_space > right_space:
                    steer = -1.0 
                else:
                    steer = 1.0  
                accel = -0.5     
                
            elif front_sensor < 0.4:
                if state[20] > 0: 
                    steer = max(-1.0, steer - 0.3)
                else:
                    steer = min(1.0, steer + 0.3)
                accel = min(0.0, accel) 
                
        return {'steer': steer, 'accelerate': accel}, [steer_idx, accel_idx]