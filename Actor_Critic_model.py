import torch
import numpy as np
import random
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from Memory import Memory

class Actor(nn.Module):
    def __init__(self,c,action_dim,batch_size):
        super(Actor,self).__init__()
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
        out = F.softmax(self.fc3(out),dim=1)

        return out

class Critic(nn.Module):
    def __init__(self, c, batch_size):
        super(Critic, self).__init__()
        self.batch_size = batch_size
        self.c = c
        self.conv1 = nn.Conv2d(c, 64, (8, 8), stride=4, padding=2)
        self.pooling1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(64, 128, (4, 4), stride=2, padding=1)
        self.pooling2 = nn.MaxPool2d((2, 2), stride=1)
        self.conv3 = nn.Conv2d(128, 128, (2, 2), stride=2)

        self.fc1 = nn.Linear(128 * 4, 128 * 6)
        self.fc2 = nn.Linear(128 * 6, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, s):
        out = F.relu(self.conv1(s))
        out = self.pooling1(out)
        out = F.relu(self.conv2(out))
        out = self.pooling2(out)
        out = F.relu(self.conv3(out))

        out = out.view(-1, 128 * 2 * 2)

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out

class Agent():
    def __init__(self,c,action_dim,memory_size,batch_size,actor_lr,critic_lr,gamma,gpu):
        self.batch_size = batch_size
        self.gamma = gamma
        self.gpu = gpu
        self.memory = Memory(memory_size,batch_size)

        self.actor = Actor(c,action_dim,batch_size)
        self.critic = Critic(c,batch_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=critic_lr)

        if gpu>=0:
            self.actor.cuda(gpu)
            self.critic.cuda(gpu)

    def random_action(self):
        action = np.zeros((2,),dtype=np.int)
        action[random.randint(1,5)%2]=1
        return np.asarray([0.5,0.5]), action

    def select_action(self,state,isTrain=True):
        # in test case, only return the max prob action
        if not isTrain:
            action = np.zeros((2,), dtype=np.int)
            state = Variable(torch.from_numpy(state).unsqueeze(0),volatile=True).float()
            if self.gpu >= 0:
                state = state.cuda(self.gpu)
            prob = self.actor(state).cpu().data[0]
            action_id = torch.max(prob, dim=0, keepdim=True)[1][0]
            action[action_id] = 1
            return prob.numpy(),action

        action = np.zeros((2,),dtype=np.int)
        state = Variable(torch.from_numpy(state).unsqueeze(0)).float()
        if self.gpu>=0:
            state = state.cuda(self.gpu)
        prob = self.actor(state).cpu().data[0]
        action_id = prob.multinomial(1)[0]
        action[action_id] = 1

        return prob.numpy(),action


    def train(self):
        if self.memory.counter < self.batch_size * 3:
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

        # update critic
        v_next = self.critic(next_state)
        v_target = reward + self.gamma*done*v_next
        v_target.detach_()
        v_eval = self.critic(state)
        td_error = (v_target-v_eval).squeeze().detach()

        value_loss = F.smooth_l1_loss(v_eval,v_target)
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic.parameters(), 0.8)
        self.critic_optimizer.step()

        # update actor
        log_prob = torch.sum(self.actor(state)*action,dim=1)
        policy_loss = -torch.sum(log_prob*td_error)
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm(self.actor.parameters(), 0.8)
        self.actor_optimizer.step()


    def save(self,episode):
        if self.gpu>=0:
            self.actor.cpu()
            self.critic.cpu()

        torch.save(self.actor,'Actor_model_{}.pt'.format(episode))
        torch.save(self.critic, 'Critic_model_{}.pt'.format(episode))

        if self.gpu>=0:
            self.actor.cuda(self.gpu)
            self.critic.cuda(self.gpu)


    def load(self,episode):
        self.actor = torch.load('Actor_model_{}.pt'.format(episode))
        self.critic = torch.load('Critic_model_{}.pt'.format(episode))
        if self.gpu>=0:
            self.actor.cuda(self.gpu)
            self.critic.cuda(self.gpu)



if __name__ == '__main__':

    a = Actor(3,2,128)
    c = Critic(3,2,128)
    s = Variable(torch.randn(128,3,80,80))
    prob = a(s)
    q = c(s,prob)
    print(prob.size())
    print(q.size())