import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


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


def terminal(s):
    x, v, angularx, angularv = s[0], s[1], s[2], s[3]
    if s[0] <= -2.4 or s[0] >= 2.4:
        return True
    if s[2] <= -0.2095 or s[2] >= 0.2095:
        return True

def reward(s, a, s_):
    return 1

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.input = nn.Linear(4, 128)
        self.output = nn.Linear(128, 2)
    def forward(self, x):
        # print("poilcy x: ", x)
        x = self.input(x)
        x = F.relu(x)
        a = self.output(x)
        # print("what is a: ", a)
        # proba = F.softmax(a, dim=0)
        logproba = F.log_softmax(a, dim=0)
        # print(proba, type(proba))
        return logproba

class ValueFunction(nn.Module):
    def __init__(self):
        super(ValueFunction, self).__init__()
        self.input = nn.Linear(4, 128)
        self.output = nn.Linear(128, 1)
    def forward(self, x):
        x = self.input(x)
        x = F.relu(x)
        valuefunction = self.output(x)
        return valuefunction


def action(s, policy):
    with torch.no_grad():
        s = s.float()
        p = policy(s)
        a = np.random.choice([0,1], p = np.exp(p.numpy()))
    return a, p[a]

def REINFORCE(policy, valuefunction):
    policyoptim = optim.Adam(policy.parameters(), lr = 1e-3)
    valuefunctionoptim = optim.Adam(valuefunction.parameters(), lr = 1e-3)
    gamma = 1
    eps = 0
    cumulativet=[0]
    steps = []
    while True and eps < 3000:
        cumulativet.append(cumulativet[-1])
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
        te = 0
        while not terminal(s) and te < 200:
            cumulativet[-1]+=1
            a = action(ts, policy)
            s_ = transition(s, a[0])
            R = reward(s, a[0], s_)
            states.append(ts.float())
            actions.append(a[0])
            logprob.append(a[1])
            rewards.append(R)
            s = s_
            ts = torch.tensor(s)
            te += 1
        G = np.zeros(len(rewards)).tolist()
        for r in range(len(rewards) - 1, -1, -1):
            G[r] = rewards[r]
            if r < len(rewards) - 1:
                G[r] += gamma*G[r+1]
        G = torch.tensor(G, dtype=torch.float)
        valueestimatelist = []
        for s in states:
            valueestimatelist.append(valuefunction(s))
        valueestimatelist = torch.tensor(valueestimatelist, dtype=torch.float, requires_grad = True).squeeze()
        #derivative of mseloss (delta**2)
        valuefunctionloss = F.mse_loss(valueestimatelist, G)
        valuefunctionoptim.zero_grad()
        valuefunctionloss.backward()
        valuefunctionoptim.step()
        policyloss = []
        deltas = []
        with torch.no_grad():
            for i in range(len(states)):
                deltas.append(G[i]-valueestimatelist[i])
        log_probs = []
        for i,j in zip(states,actions):
            log_probs.append(policy(i)[j])
        for i in range(len(states)):
            policyloss.append(-deltas[i]*log_probs[i])
        policyoptim.zero_grad()
        sum(policyloss).backward()
        policyoptim.step()
        steps.append(te)
        eps += 1 
        if eps%100 == 0:
            print("return at episode " +str(eps), te)
    cumulativet = cumulativet[1:]
    return steps, cumulativet

runs = 20
tarray = [] 
cumulativetarray = [] 
for i in range(runs):
    print("RUN: ", i + 1)
    policy = Policy()
    valuefunction = ValueFunction()
    reinforce = REINFORCE(policy, valuefunction)
    tarray.append(reinforce[0])
    cumulativetarray.append(reinforce[1])
lengths = []
for i in tarray:
    lengths.append(len(i))
eps = min(lengths)

#mean b)
cumulativesteps = np.zeros(eps)
for t in cumulativetarray:
    trunct = np.array(t[:eps])
    cumulativesteps += trunct/runs

#mean c)
meansteps = np.zeros(eps) 
for t in tarray:
    trunct = np.array(t[:eps])
    meansteps += trunct/runs
#standard deviation
std = np.zeros((eps, runs)).tolist()
for i in range(runs):
    for k in range(eps):
        std[k][i] = tarray[i][k]
stdsteps = np.zeros(eps)
for k in range(eps):
    stdsteps[k] = np.std(np.array(std[k]))

print("finavg: ", np.mean(meansteps[-100:]))

figure, ax = plt.subplots(2, 1)
#PLOT b)
ax[0].plot(cumulativesteps, range(eps))
ax[0].set_xlabel("Number of Time Steps")
ax[0].set_ylabel("Number of Episodes")

# PLOT c)
ax[1].plot(range(eps), meansteps)
ax[1].fill_between(range(eps), meansteps - stdsteps, meansteps + stdsteps, alpha=0.7, color="red")
# plt.plot(range(eps), meansteps.tolist())
ax[1].set_xlabel("Number of Episodes")
ax[1].set_ylabel("Average Number of Steps")
ax[1].set_yticks(range(0, 300, 50))
ax[1].tick_params(labelright=True)
plt.show()







