import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from Replay_buffer import Replay_buffer
from PPO import PPOAgent
from env_flat import Hexapod as envir


USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

def evaluate_policy(args, env, agent):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        state,_ = env.reset()
        
        done = False
        episode_reward = 0
        while not done:
            action = agent.evaluate(state)
            action = action.cpu()

            next_state, reward,_, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        evaluate_reward += episode_reward


    return evaluate_reward / times

def main(args, seed):
    path = './result'
    env = envir()
    env_evaluate = envir()

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

    evaluate_num = 0
    evaluate_rewards = []
    total_steps = 0

    replay_buffer = Replay_buffer(args)
    agent = PPOAgent(args)

    writer = SummaryWriter(log_dir='runs/env_jethexa_seed_{}'.format(seed))

    while total_steps < args.max_train_steps:
        state,_ = env.reset()
        episode_steps = 0
        done = False
        
        while not done:
            episode_steps += 1
            _, action, action_logprob = agent.agent.act(state)
            action = action.cpu()
            action_logprob = action_logprob.cpu()
            
            next_state, reward, _, done, _ = env.step(action)
            
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False
            
            replay_buffer.push(state, action, action_logprob, reward, next_state, dw, done)
            state = next_state
            total_steps += 1
            
            if replay_buffer.idx == args.batch_size:
                loss = agent.update(replay_buffer)
                replay_buffer.idx = 0
            
            if total_steps % args.evaluate_freq == 0 :
                evaluate_num += 1
                evaluate_reward = evaluate_policy(args, env_evaluate, agent)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                print('loss:{}'.format(loss))
                writer.add_scalar('step_rewards_jethexa', evaluate_rewards[-1], global_step=total_steps)
                # Save the rewards
                if evaluate_num % args.save_freq == 0:
                    np.save('./data_train/PPO_env_jethexa_seed_{}.npy'.format(seed), np.array(evaluate_rewards))
                    
                    agent.save_model(path,total_steps)
    agent.save_model(path,total_steps)

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


