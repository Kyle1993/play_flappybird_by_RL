import torch
import numpy as np
import random
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from Memory import Memory

class DQN(nn.Module):
    def __init__(self,c,action_dim,batch_size):
        super(DQN,self).__init__()
        self.batch_size = batch_size
        self.c = c
        self.conv1 = nn.Conv2d(c,64,(8,8),stride=4,padding=2)
        self.pooling1 = nn.MaxPool2d((2,2))
        self.conv2 = nn.Conv2d(64,128,(4,4),stride=2,padding=1)
        self.pooling2 = nn.MaxPool2d((2, 2),stride=1)
        self.conv3 = nn.Conv2d(128,128,(2,2),stride=2)

        self.fc1 = nn.Linear(128*4,128*6)
        self.fc2 = nn.Linear(128*6,32)
        self.fc3 = nn.Linear(32,action_dim)

    def forward(self, s):
        out = F.relu(self.conv1(s))
        out = self.pooling1(out)
        out = F.relu(self.conv2(out))
        out = self.pooling2(out)
        out = F.relu(self.conv3(out))

        out = out.view(-1,128*2*2)

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out

class Agent():
    def __init__(self,c,action_dim,memory_size,batch_size,lr,gamma,gpu,epsilon_decay):
        self.epsilon = 1
        self.batch_size = batch_size
        self.gamma = gamma
        self.gpu = gpu
        self.epsilon_decay = epsilon_decay
        self.memory = Memory(memory_size,batch_size)
        self.DQN = DQN(c,action_dim,batch_size)
        self.optimizer = optim.Adam(self.DQN.parameters(),lr=lr)

        if gpu>=0:
            self.DQN.cuda(gpu)

    def random_action(self):
        action = np.zeros((2,),dtype=np.int)
        action[random.randint(1,5)%2]=1
        return np.asarray([0.5,0.5]), action

    def select_action(self,state,isTrain=True,decay=True):
        if not isTrain:
            action = np.zeros((2,), dtype=np.int)
            state = Variable(torch.from_numpy(state).unsqueeze(0)).float()
            if self.gpu >= 0:
                state = state.cuda(self.gpu)
            q_value = self.DQN(state).cpu().data[0]
            action_id = torch.max(q_value, dim=0, keepdim=True)[1][0]
            action[action_id] = 1
            return action

        if self.memory.counter < self.batch_size:
            action = np.zeros((2,), dtype=np.int)
            action[random.randint(1, 5) % 2] = 1
            return action

        if random.random()<self.epsilon:
            action = np.zeros((2,), dtype=np.int)
            action[random.randint(1, 5) % 2] = 1
        else:
            action = np.zeros((2,),dtype=np.int)
            state = Variable(torch.from_numpy(state).unsqueeze(0)).float()
            if self.gpu>=0:
                state = state.cuda(self.gpu)
            q_value = self.DQN(state).cpu().data[0]
            action_id = torch.max(q_value,dim=0,keepdim=True)[1][0]
            action[action_id] = 1

        if decay:
            self.epsilon -= self.epsilon_decay
            self.epsilon = max(0.1, self.epsilon)

        return action


    def train(self):
        if self.memory.counter < self.batch_size:
            return

        state_batch, action_batch, next_state_batch, reward_batch, done_batch = self.memory.sample()

        state = Variable(torch.from_numpy(state_batch))
        action = Variable(torch.from_numpy(action_batch)).float()
        next_state = Variable(torch.from_numpy(next_state_batch))
        reward = Variable(torch.from_numpy(reward_batch).view(self.batch_size,1))
        done = Variable(torch.from_numpy(done_batch).view(self.batch_size,1)).float()

        if self.gpu >= 0:
            state = state.cuda(self.gpu)
            action = action.cuda(self.gpu)
            next_state = next_state.cuda(self.gpu)
            reward = reward.cuda(self.gpu)
            done = done.cuda(self.gpu)

        q_next = self.DQN(next_state).max(1)[0].view(self.batch_size,1)
        q_target = reward + self.gamma*done*q_next
        q_target.detach_()
        q_eval = torch.sum(self.DQN(state)*action,dim=1).unsqueeze(-1)

        loss = F.smooth_l1_loss(q_eval,q_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.DQN.parameters(), 0.8)
        self.optimizer.step()

    def save(self,episode):
        if self.gpu>=0:
            self.DQN.cpu()

        torch.save(self.DQN,'DQNmodel_{}.pt'.format(episode))

        if self.gpu>=0:
            self.DQN.cuda(self.gpu)

    def load(self,episode):
        self.DQN = torch.load('DQNmodel_{}.pt'.format(episode))
        if self.gpu>=0:
            self.DQN.cuda(self.gpu)



if __name__ == '__main__':

    net = DQN(3,2,128)
    x = Variable(torch.randn(128,3,80,80))
    y = net(x)
    print(y.size())