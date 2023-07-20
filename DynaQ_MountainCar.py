import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from math import floor
from collections import defaultdict

def transition(s, a):
    x, v = s[0], s[1]
    v_ = v + 0.001*a -0.0025*np.cos(3*x)
    x_ = x + v_
    #position clipping
    if x_ < -1.2:
        x_ = -1.2
    elif x_ > 0.5:
        x_ = 0.5
    #velocity clipping
    if v_ < -0.7:
        v_ = -0.7
    elif v_ > 0.7:
        v_ = 0.7
    #inelastic collision at bounds
    if x_ == -1.2 or x_ == 0.5:
        v_ = 0
    s_ = (x_, v_)
    return s_


def dstate(s):
    x = s[0]
    v = s[1]
    dx = floor((x + 1.2)/(1.8/bins)) #discretizing position
    dv = floor((v + 0.07)/(0.14/bins)) #discretizing velocity
    ds = (dx, dv)
    return ds


def terminal(s):
    if s[0] == 0.5:
        return True

def reward(s, a, s_):
    if terminal(s):
        return 0
    return -1


def DYNA(epsilon):
    # print("Entering fn.")
    n = 5
    alpha = 0.05
    gamma = 1
    qvaluefunction = np.random.rand(bins, bins, 3)
    qvaluefunction[bins - 1, :, :] = 0
    model = defaultdict(tuple)
    # of the form:
    # for i in range(bins):
    #     for j in range(bins):
    #         for a in range(3):
    #             model[(i, j, a)] = [0, (0,0)]
    visited = defaultdict(tuple)
    eps = 0
    cumulativet=[0]
    steps = []
    while True and eps < 7000:
        """
        Generate an episode
        """
        cumulativet.append(cumulativet[-1])
        x = random.uniform(-0.6, -0.4)
        v = 0 
        s = (x, v)
        ds = dstate(s)
        t = 0
        while not terminal(s) and t < 1000:
            cumulativet[-1]+=1
            if random.uniform(0, 1) < epsilon:
                a = np.random.choice(range(3))
            else:
                # a = np.argmax(qvaluefunction[ds[0], ds[1], :])
                besti = np.argwhere(qvaluefunction[ds[0], ds[1], :] == np.max(qvaluefunction[ds[0], ds[1], :]))
                besta = [item for i in besti for item in i]
                a = np.random.choice(besta)
            #visited
            if ds not in list(visited.keys()):
                visited[ds] = [a]
            else:
                if a not in visited[ds]: visited[ds].append(a)
            s_ = transition(s, a)
            ds_ = dstate(s_)
            R = reward(s, a, s_)
            qvaluefunction[ds[0], ds[1], a] =  qvaluefunction[ds[0], ds[1], a] + alpha*(R + gamma*np.max(qvaluefunction[ds_[0], ds_[1], :]) - qvaluefunction[ds[0], ds[1], a])
            model[(ds[0], ds[1], a)] = [R, (ds_[0], ds_[1])]
            # print("visited: ", visited.keys())
            for i in range(10):
                indexs = np.random.choice(range(len(visited)))
                rs = list(visited.keys())[indexs] #already discretized
                ra = np.random.choice(visited[rs])
                rs_ = model[(rs[0], rs[1], ra)][1]
                rR = model[(rs[0], rs[1], ra)][0]
                qvaluefunction[rs[0], rs[1], ra] =  qvaluefunction[rs[0], rs[1], ra] + alpha*(rR + gamma*np.max(qvaluefunction[rs_[0], rs_[1], :]) - qvaluefunction[rs[0], rs[1], ra])
            s = s_ 
            ds = dstate(s)
            t+=1
        if eps%100 == 0:
            valuefunction = np.mean(np.max(qvaluefunction, 2))
            print("Generating episode: ",eps + 1, t, valuefunction)
            print("Terminal State: ", s_)
        steps.append(t)
        eps+=1
    cumulativet = cumulativet[1:]
    return steps, cumulativet

bins = 40
epsilon = 0.05
runs = 5
tarray = [] 
cumulativetarray = [] 
for i in range(runs):
    print("RUN: ", i + 1)
    dyna = DYNA(epsilon)
    tarray.append(dyna[0])
    cumulativetarray.append(dyna[1])
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

print("finavg: ", np.mean(meansteps[-1000:]))

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
ax[1].set_yticks(range(-100, 1100, 100))
ax[1].tick_params(labelright=True)
plt.show()








