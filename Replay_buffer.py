import numpy as np
import torch


class Replay_buffer:
    '''
    实现一个Replay buffer
    '''
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.state = np.zeros((args.batch_size, args.state_dim))
        self.action = np.zeros((args.batch_size, args.action_dim))
        self.action_logprob = np.zeros((args.batch_size, args.action_dim))
        self.reward = np.zeros((args.batch_size, 1))
        self.next_state = np.zeros((args.batch_size, args.state_dim))
        self.dw = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))
        self.idx = 0

    def push(self, state, action, action_logprob, reward, next_state, dw, done):
        self.state[self.idx] = state
        self.action[self.idx] = action
        self.action_logprob[self.idx] = action_logprob
        self.reward[self.idx] = reward
        self.next_state[self.idx] = next_state
        self.dw[self.idx] = dw
        self.done[self.idx] = done
        self.idx += 1
    
    def numpy2tensor(self):
        state = torch.tensor(self.state, dtype=torch.float)
        action = torch.tensor(self.action, dtype=torch.float)
        action_logprob = torch.tensor(self.action_logprob, dtype=torch.float)
        reward = torch.tensor(self.reward, dtype=torch.float)
        next_state = torch.tensor(self.next_state, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)
        return state, action, action_logprob, reward, next_state, dw, done

        

    def size(self):
        return len(self.buffer)