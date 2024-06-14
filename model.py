import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Normal

class MLPModel(nn.Module):
    def __init__(self, args):
        super(MLPModel, self).__init__()       
        self.ln1 = nn.LayerNorm(normalized_shape = 256)
        self.fc1 = nn.Linear(args.state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, args.action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))
        mean = self.mean(x) # 动作幅度-> [-1, 1]
        return mean
    
class MLPModel_value(nn.Module):
    def __init__(self, args):
        super(MLPModel_value, self).__init__()       
        self.ln1 = nn.LayerNorm(normalized_shape = 256)
        self.fc1 = nn.Linear(args.state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))
        mean = self.mean(x) # 动作幅度-> [-1, 1]
        return mean
    
class ActorCritic(nn.Module):
    def __init__(self, args):
        super(ActorCritic, self).__init__()     

        # MLP network
        self.v_net = MLPModel_value(args)
        self.mu_net = MLPModel(args)

        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))
        

    def forward(self, obs):

        val = self.v_net(obs)
        mu = self.mu_net(obs)

        std = torch.exp(self.log_std)
        
        pi = Normal(mu, std)
        entropy = pi.entropy().mean()


        return val,pi,entropy


class Agent:
    def __init__(self, actor_critic,args):
        self.ac = actor_critic
        self.max_action = args.max_action
        self.args = args
        self.USE_CUDA = args.USE_CUDA

    @torch.no_grad()
    def act(self, obs):
        obs = torch.tensor(obs,dtype=torch.float32)
        if self.USE_CUDA:
            device = 'cuda:0'
            obs = obs.to(device)
        val, pi,_ = self.ac(obs)
        self.pi = pi
        act = pi.sample()
        #act = torch.clamp(act, -self.max_action, self.max_action) # 将动作裁减到最大动作之内
        logp = pi.log_prob(act) # 动作的对数概率密度分布\
        return val, act, logp

    @torch.no_grad()
    def get_value(self, obs):
        val, _, _= self.ac(obs)
        return val