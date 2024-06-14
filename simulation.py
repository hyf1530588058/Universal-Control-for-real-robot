import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from PPO import PPOAgent
from env_flat import Hexapod as envir
import os
import gym


def main(args, seed):
    r = 0
    path = './result'
    env = envir()
    env = gym.wrappers.RecordVideo(env,'video')
    env.unwrapped.render_mode = "rgb_array"
  
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])
    
    args.max_episode_steps = env.max_steps
    print('state_dim={}'.format(args.state_dim))
    print('action_dim={}'.format(args.action_dim))
    print("max_action={}".format(args.max_action))
    print("max_episode_steps={}".format(args.max_episode_steps))
    
    agent = PPOAgent(args)
    agent.agent.ac.load_state_dict(torch.load(os.path.join(path,'jethexa_PPO_steps:6000000.pth')))
    agent.agent.ac.eval()
    for i in range(1):
        obs,_ = env.reset()
        
        for j in range(399):
            _,action, _ = agent.agent.act(obs)
            obs, rewards, dones, info,_ = env.step(action)
            r += rewards

            #env.render()
        print("reward", r)
        r = 0
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-jethexa")
    parser.add_argument("--max_train_steps", type=int, default=int(6e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=1e4, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    #parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=512, help="Minibatch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Tricks: policy entropy")
    parser.add_argument('--USE_CUDA', type=bool, default=False, help='CUDA')
    parser.add_argument('--use_adv_norm', type=bool, default=True, help='Tricks: use advantage normalization')
    args = parser.parse_args()
    main(args, seed=8)