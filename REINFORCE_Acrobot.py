import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

def transition(s, a):
    c1, c2, s1, s2, v1, v2 = s[0], s[1], s[2], s[3], s[4], s[5]
    theta1 = np.arctan2(s1, c1)
    theta2 = np.arctan2(s2, c2)

    # defining constant values
    m1 = 1 #mass of link 1
    m2 = 1 #mass of link 2
    l1 = 1 #length of link 1
    l2 = 1 #length of link 2
    lc1 = 0.5 #length to center of mass of link 1
    lc2 = 0.5 #length to center of mass of link 1
    I1 = 1 #moment of inertia of link 1
    I2 = 1 #moment of inertia of link 1
    g = 9.8 #gravity
    dt = 0.05


    #dynamics
    d1 = (m1 * lc1**2) + m2*((l1**2 + lc2**2 + 2*l1*lc2*c2) + I1 + I2)
    d2 = m2 * (lc2**2 + l1*lc2*c2) + I2

    phi2 = m2*lc2*g*np.cos(theta1 + theta2 - np.pi/2)
    phi1 = (-m2*l1*lc2*(v2**2)*s2) - (2*m2*l1*lc2*v2*v1*s2) + (m1*lc1 + m2*l1)*g*np.cos(theta1 - np.pi/2) + phi2

    acc2 = (1/(m2*(lc2**2) + I2 - (d2**2/d1))) * (a - 1 + (d2/d1)*phi1 - m2*l1 *lc2*v1**2*np.sin(theta2) - phi2) #a-1 because torgue is {-1, 0, 1}
    acc1 = -(1/d1)*(d2*acc2 + phi1)


    #updates
    theta1 += v1*dt
    theta2 += v2*dt
    v1 += acc1*dt
    v2 += acc2*dt
    c1 = np.cos(theta1)
    c2 = np.cos(theta2)
    s1 = np.sin(theta1)
    s2 = np.sin(theta2)

    s_ = (c1, c2, s1, s2, v1, v2)
    return s_


def terminal(s):
    c1, c2, s1, s2, v1, v2 = s[0], s[1], s[2], s[3], s[4], s[5]
    theta1 = np.arctan2(s1, c1)
    theta2 = np.arctan2(s2, c2)
    if -np.cos(theta1) - np.cos(theta2 + theta1) > 1:
        return True

def reward(s, a, s_):
    return -1

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.input = nn.Linear(6, 512)
        self.output = nn.Linear(512, 3)
    def forward(self, x):
        # print("poilcy x: ", x)
        x = self.input(x)
        x = F.relu(x)
        a = self.output(x)
        # print("what is a: ", a)
        # proba = F.softmax(a, dim=-1)
        logproba = F.log_softmax(a, dim=0)
        # print(proba, type(proba))
        return logproba

class ValueFunction(nn.Module):
    def __init__(self):
        super(ValueFunction, self).__init__()
        self.input = nn.Linear(6, 512)
        self.output = nn.Linear(512, 1)
    def forward(self, x):
        x = self.input(x)
        x = F.relu(x)
        valuefunction = self.output(x)
        return valuefunction

def action(s, policy):
    with torch.no_grad():
        s = s.float()
        p = policy(s)
        a = np.random.choice([0,1,2], p = np.exp(p.numpy()))
    return a, p[a]

def REINFORCE(policy, valuefunction):
    policyoptim = optim.Adam(policy.parameters(), lr = 3e-2)
    valuefunctionoptim = optim.Adam(valuefunction.parameters(), lr = 3e-2)
    gamma = 1
    eps = 0
    cumulativet=[0]
    steps = []
    while True and eps < 1000:
        cumulativet.append(cumulativet[-1])
        theta1 = random.uniform(-0.1, +0.1)
        theta2 = random.uniform(-0.1, +0.1)
        c1 = np.cos(theta1)
        c2 = np.cos(theta2)
        s1 = np.sin(theta1)
        s2 = np.sin(theta2)
        v1 = random.uniform(-0.1, +0.1)
        v2 = random.uniform(-0.1, +0.1)
        s = (c1, c2, s1, s2, v1, v2)
        ts = torch.tensor(s) #tuple to tensor for NN architecture
        states = []
        actions = []
        logprob = []
        rewards = []
        te = 0
        while not terminal(s) and te < 500:
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
            policyloss.append(deltas[i]*log_probs[i])
        # print("policyloss: ", policyloss)
        policyoptim.zero_grad()
        sum(policyloss).backward()
        policyoptim.step()
        steps.append(te)
        eps += 1 
        # if eps%100 == 0:
        print("return at episode " +str(eps), te)
        # print("return at episode " +str(eps), te, valuefunctionloss, sum(policyloss))
    cumulativet = cumulativet[1:]
    return steps, cumulativet

runs = 1
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
ax[1].set_yticks(range(0, 600, 100))
ax[1].tick_params(labelright=True)
plt.show()