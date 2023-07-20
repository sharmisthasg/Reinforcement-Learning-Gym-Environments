import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from math import floor
from collections import defaultdict

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
    F = F if a == 1 else -F #syntax
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
    dx = floor((x + 4.8)/(9.6/bins[0])) - 1
    dangularx = floor((angularx + 0.42)/(0.84/bins[1])) - 1
    dv = floor((v + 5)/(10/bins[2])) - 1
    dangularv = floor((angularv + 5)/(10/bins[3])) - 1
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


def DYNA(epsilon):
    # print("Entering fn.")
    n = 1
    alpha = 0.1
    gamma = 1
    qvaluefunction = np.random.rand(bins[0], bins[1], bins[2], bins[3], 2)
    qvaluefunction[bins[0] - 1, :, bins[2] - 1, :, :] = 0
    model = defaultdict(tuple)
    visited = defaultdict(tuple)
    eps = 0
    cumulativet=[0]
    steps = []
    while True and eps < 20000:
        """
        Generate an episode
        """
        cumulativet.append(cumulativet[-1])
        x = random.uniform(-0.05, +0.05)
        v = random.uniform(-0.05, +0.05)
        angularx = random.uniform(-0.05, +0.05)
        angularv = random.uniform(-0.05, +0.05)
        s = (x, v, angularx, angularv)
        ds = dstate(s)
        t = 0
        while not terminal(s) and t < 200:
            cumulativet[-1]+=1
            ##trying random policy
            # a = np.random.choice(range(2))
            if random.uniform(0, 1) < epsilon:
                a = np.random.choice(range(2))
            else:
                besti = np.argwhere(qvaluefunction[ds[0], ds[1], ds[2], ds[3], :] == np.max(qvaluefunction[ds[0], ds[1], ds[2], ds[3], :]))
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
            qvaluefunction[ds[0], ds[1], ds[2], ds[3], a] += alpha*(R + gamma*np.max(qvaluefunction[ds_[0], ds_[1], ds_[2], ds_[3], :]) - qvaluefunction[ds[0], ds[1], ds[2], ds[3], a])
            # model[(ds[0], ds[1], ds[2], ds[3], a)] = [R, (ds_[0], ds_[1], ds_[2], ds_[3])]
            # for i in range(n):
            #     indexs = np.random.choice(range(len(visited)))
            #     rs = list(visited.keys())[indexs] #already discretized
            #     ra = np.random.choice(visited[rs])
            #     rs_ = model[(rs[0], rs[1], rs[2], rs[3], ra)][1]
            #     rR = model[(rs[0], rs[1], rs[2], rs[3], ra)][0]
            #     qvaluefunction[rs[0], rs[1], rs[2], rs[3], ra] += alpha*(rR + gamma*np.max(qvaluefunction[rs_[0], rs_[1], rs_[2], rs_[3], :]) - qvaluefunction[rs[0], rs[1], rs[2], rs[3], ra])
            s = s_ 
            ds = dstate(s)
            t+=1
        # if eps%250 == 0:
        # epsilon -= 0.0001
        # if epsilon < 0.001:
        #     epsilon = 0.001
        if eps%100 == 0:
            valuefunction = np.mean(np.max(qvaluefunction, 4))
            print("Generating episode: ",eps + 1, t, epsilon)
            # print("Terminal State: ", s_)
        steps.append(t)
        eps+=1
    cumulativet = cumulativet[1:]
    return steps, cumulativet

bins = [25, 40, 25, 40]
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








