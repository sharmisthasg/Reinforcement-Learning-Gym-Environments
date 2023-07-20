import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def transition(s, a):
    x, v, angularx, angularv = s[0], s[1], s[2], s[3]
    #defining constant values
    g = 9.8 #gravity
    mc = 1 #mass of cart
    mp = 0.1 #mass of pole
    lp = 0.5 #length of pole
    F = 10 #force magnitude
    dt = 0.02

    #dynamics
    m = mc + mp
    F = F if a == 1 else -F 
    temp = (F + mp*lp*(angularv**2)*np.sin(angularx))/m
    angularacc = (g*np.sin(angularx) - np.cos(angularx)*temp)/(lp * (4/3 - mp*np.cos(angularx)**2/m))
    acc = temp - mp*lp*angularacc*np.cos(angularx)/m

    #updates
    x += v * dt
    v += acc * dt
    angularx += angularv * dt
    angularv += angularacc * dt

    s_ = (x, v, angularx, angularv)
    return s_


def dstate(s):
    x, v, angularx, angularv = s[0], s[1], s[2], s[3]
    dx = floor((x + 4.8)/(9.6/bins)) - 1
    dangularx = floor((angularx + 0.42)/(0.84/bins)) - 1
    dv = floor((v + 5)/(10/bins)) - 1
    dangularv = floor((angularv + 5)/(10/bins)) - 1
    ds = (dx, dangularx, dv, dangularv)
    return ds


def terminal(s):
    x, v, angularx, angularv = s[0], s[1], s[2], s[3]
    if s[0] <= -2.4 or s[0] >= 2.4:
        return True
    if s[2] <= -0.2095 or s[2] >= 0.2095:
        return True

def reward(s, a, s_):
    return 1

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.input = nn.Linear(4, 128)
        self.outputa = nn.Linear(128, 2)
        self.outputv = nn.Linear(128, 1)
    def forward(self, x):
        # print("poilcy x: ", x)
        x = self.input(x)
        x = F.relu(x)
        a = self.outputa(x)
        logproba = F.log_softmax(a, dim=-1)
        v = self.outputv(x)
        return logproba, v

def action(s, network):
    with torch.no_grad():
        s = s.float()
        n = network(s)
        a = np.random.choice([0,1], p = np.exp(n[0].numpy()))
        vest = n[1]
    return a, n[0][a], vest

def REINFORCE(network):
    gamma = 1
    eps = 0
    while True and eps < 5000:
        x = random.uniform(-0.05, +0.05)
        v = random.uniform(-0.05, +0.05)
        angularx = random.uniform(-0.05, +0.05)
        angularv = random.uniform(-0.05, +0.05)
        s = (x, v, angularx, angularv)
        ts = torch.tensor(s) #tuple to tensor for NN architecture
        states = []
        actions = []
        logprob = []
        rewards = []
        valueestimatelist = []
        te = 0
        while not terminal(s) and te < 200:
            a = action(ts, network)
            s_ = transition(s, a[0])
            R = reward(s, a[0], s_)
            states.append(ts.float())
            actions.append(a[0])
            logprob.append(a[1])
            rewards.append(R)
            valueestimatelist.append(a[2])
            s = s_
            ts = torch.tensor(s)
            te += 1
        G = np.zeros(len(rewards)).tolist()
        for r in range(len(rewards) - 1, -1, -1):
            G[r] = rewards[r]
            if r < len(rewards) - 1:
                G[r] += gamma*G[r+1]
        G = torch.tensor(G, dtype=torch.float)

        policyloss = []
        valuefunctionloss= []
        for i in range(len(states)):
            delta = G[i] - valueestimatelist[i]
            valuefunctionloss.append(F.mse_loss(valueestimatelist[i], G[i]))
            policyloss.append(-logprob[i] * delta)

        optimizer = optim.Adam(network.parameters(), lr = 0.01)
        loss = torch.stack(valuefunctionloss).sum() + torch.stack(policyloss).sum()
        loss.backward()
        optimizer.step()

        eps += 1 
        # if eps%100 == 0:
        print("return at episode " +str(eps), te)
        # print("return at episode " +str(eps), te, valuefunctionloss, sum(policyloss))

network = Network()
reinforce = REINFORCE(network)






