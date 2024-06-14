import torch
import model
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Normal
import random
import os
from model import ActorCritic
from model import Agent
'''
实现PPO算法的主体，包括：
一个Agent
一个ac网络，actor网络返回动作概率分布，critic网络返回价值

方法：
训练：从经验回放池中获得数据，与环境进行交互得到梯度，进行梯度更新
保存模型
加载模型
'''
class PPOAgent:
    '''
    实现一个PPOAgent：
    评价策略:将状态输入到Critic网络，得到动作的分值；
    选择动作：通过Actor采样动作，并得到

    '''
    def __init__(self, args):
        self.max_action = args.max_action
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr # Learning rate for actor network
        self.gamma = args.gamma # Discount factor
        self.lamda = args.lamda # GAE parameter
        self.epsilon = args.epsilon # PPO clip parameter
        self.K_epochs = args.K_epochs #The frequency of critic update
        self.use_adv_norm = args.use_adv_norm # Tricks: use advantage normalization
        self.entropy_coef = args.entropy_coef # Entropy
        self.USE_CUDA = args.USE_CUDA

        self.actor_critic = ActorCritic(args)
        
        
        if self.USE_CUDA:
            device = 'cuda:0'
            self.actor_critic = self.actor_critic.to(device)
        self.agent = Agent(self.actor_critic, args)
            
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.lr, eps=1e-5)

    def evaluate(self,state):
        _,action,_ = self.agent.act(state)
        return action
    
    def update(self, replay_buffer):
        states, actions, action_logprobs, rewards, next_states, dws, dones = replay_buffer.numpy2tensor()
        '''
        dws代表胜利或失败，dw意味着没有下一状态
        done代表胜利、失败或达到最大时间步，done意味着gae = 0
        '''
        if self.USE_CUDA:
            device = 'cuda:0'
            states = states.to(device)
            actions = actions.to(device)
            action_logprobs = action_logprobs.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dws = dws.to(device)
            dones = dones.to(device)
        
        advantage = []
        gae = 0
        #首先计算advantage函数，不需要反传梯度
        with torch.no_grad():
            value_state = self.agent.get_value(states)
            value_next_state = self.agent.get_value(next_states)
            deltas = rewards + self.gamma * (1.0 - dws) * value_next_state - value_state
            for delta, dones in zip(reversed(deltas.cpu().numpy().flatten()), reversed(dones.cpu().numpy().flatten())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - dones)
                advantage.insert(0, gae)
            advantage = torch.tensor(advantage, dtype=torch.float).view(-1, 1)
            if self.use_adv_norm:  # Trick 1:advantage normalization
                advantage = ((advantage - advantage.mean()) / (advantage.std() + 1e-5))
            if self.USE_CUDA:
                advantage = advantage.cuda()
            v_target = advantage + value_state
        
        for _ in range(self.K_epochs):
            for i in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                val,dist_now,dist_entropy = self.actor_critic(states[i])
                action_logprob_now = dist_now.log_prob(actions[i])
                ratios = torch.exp(action_logprob_now.sum(1, keepdim=True) - action_logprobs[i].sum(1, keepdim=True)) # shape[mini_batch_size * 1]
                surr1 = ratios * advantage[i]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantage[i]
                
                next_value_state = self.agent.get_value(states[i])
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * dist_entropy
                critic_loss = F.mse_loss(v_target[i], next_value_state)
                loss = critic_loss * 0.5
                loss += actor_loss
                # 更新网络参数
                self.optimizer.zero_grad()
                loss.backward() 
                '''for name, param in self.actor_critic.mu_net.named_parameters():  
                    if param.grad is not None:  
                        print(f"Layer: {name}, Grad: {param.grad}")  '''     
                self.optimizer.step()
        return loss
    
    def save_model(self,path,total_steps):
        torch.save(self.actor_critic.state_dict(), os.path.join(path, 'jethexa_PPO_steps:{}.pth'.format(total_steps)))












    