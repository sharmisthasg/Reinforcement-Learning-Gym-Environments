import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from math import floor
from collections import defaultdict

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
    dt = 0.2

    d1 = (m1 * lc1**2) + m2*(l1**2 + lc2**2 + 2*l1*lc2*c2) + I1 + I2
    d2 = m2 * (lc2**2 + l1*lc2*c2) + I2

    phi2 = m2*lc2*g*np.cos(theta1 + theta2 - np.pi/2)
    phi1 = (-m2*l1*lc2*(v2**2)*s2) - (2*m2*l1*lc2*v2*v1*s2) + (m1*lc1 + m2*l1)*g*np.cos(theta1 - np.pi/2) + phi2

    acc2 = (1/(m2*(lc2**2) + I2 - d2**2/d1)) * (a - 1 + (d2/d1)*phi1 - m2*l1 *lc2*v1**2*s2 - phi2) #a-1 because torgue is {-1, 0, 1}
    acc1 = -(1/d1)*(d2*acc2 + phi1)

    theta1 += v1*dt
    v1 += acc1*dt
    theta2 += v2*dt
    v2 += acc2*dt

    c1 = np.cos(theta1)
    c2 = np.cos(theta2)
    s1 = np.sin(theta1)
    s2 = np.sin(theta2)

    s_ = (c1, c2, s1, s2, v1, v2)
    return s_


def dstate(s):
    c1, c2, s1, s2, v1, v2 = s[0], s[1], s[2], s[3], s[4], s[5]
    dc1 = floor((c1 + 1)/(2/bins))
    dc2 = floor((c2 + 1)/(2/bins))
    ds1 = floor((s1 + 1)/(2/bins))
    ds2 = floor((s2 + 1)/(2/bins))
    dv1 = floor((v1 + 12.567)/(25.134/bins))
    dv2 = floor((v2 + 28.274)/(56.548/bins))
    ds = (dc1, dc2, ds1, ds2, dv1, dv2)
    return ds


def terminal(s):
    c1, c2, s1, s2, v1, v2 = s[0], s[1], s[2], s[3], s[4], s[5]
    theta1 = np.arctan2(s1, c1)
    theta2 = np.arctan2(s2, c2)
    if -np.cos(theta1) - np.cos(theta2 + theta1) > 1:
        return True

def reward(s, a, s_):
    return -1


def DYNA(epsilon):
    # print("Entering fn.")
    n = 5
    alpha = 0.002
    gamma = 1
    qvaluefunction = np.zeros((bins, bins, bins, bins, bins, bins, 3))
    model = defaultdict(tuple)
    visited = defaultdict(tuple)
    eps = 0
    cumulativet=[0]
    steps = []
    while True and eps < 1000:
        """
        Generate an episode
        """
        cumulativet.append(cumulativet[-1])
        c1 = random.uniform(-0.1, +0.1)
        c2 = random.uniform(-0.1, +0.1)
        s1 = random.uniform(-0.1, +0.1)
        s2 = random.uniform(-0.1, +0.1)
        v1 = random.uniform(-0.1, +0.1)
        v2 = random.uniform(-0.1, +0.1)
        s = (c1, c2, s1, s2, v1, v2)
        ds = dstate(s)
        t = 0
        while not terminal(s) and t < 200:
            cumulativet[-1]+=1
            ##trying random policy
            # a = np.random.choice(range(2))
            if random.uniform(0, 1) < epsilon:
                a = np.random.choice(range(3))
            else:
                besti = np.argwhere(qvaluefunction[ds[0], ds[1], ds[2], ds[3], ds[4], ds[5], :] == np.max(qvaluefunction[ds[0], ds[1], ds[2], ds[3], ds[4], ds[5], :]))
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
            if terminal(s_):
                qvaluefunction[ds[0], ds[1], ds[2], ds[3], a] += alpha*(R - qvaluefunction[ds[0], ds[1], ds[2], ds[3], a])
            else:
                qvaluefunction[ds[0], ds[1], ds[2], ds[3], a] += alpha*(R + gamma*np.max(qvaluefunction[ds_[0], ds_[1], ds_[2], ds_[3], :]) - qvaluefunction[ds[0], ds[1], ds[2], ds[3], a])
            model[(ds[0], ds[1], ds[2], ds[3], ds[4], ds[5], a)] = [R, (ds_[0], ds_[1], ds_[2], ds_[3], ds_[4], ds_[5])]
            # print("visited: ", visited.keys())
            for i in range(n):
                indexs = np.random.choice(range(len(visited)))
                rs = list(visited.keys())[indexs] #already discretized
                ra = np.random.choice(visited[rs])
                rs_ = model[(rs[0], rs[1], rs[2], rs[3], rs[4], rs[5], ra)][1]
                rR = model[(rs[0], rs[1], rs[2], rs[3], rs[4], rs[5], ra)][0]
                if terminal(rs_):
                    qvaluefunction[rs[0], rs[1], rs[2], rs[3], ra] += alpha*(rR - qvaluefunction[rs[0], rs[1], rs[2], rs[3], ra])
                else:
                    qvaluefunction[rs[0], rs[1], rs[2], rs[3], ra] += alpha*(rR + gamma*np.max(qvaluefunction[rs_[0], rs_[1], rs_[2], rs_[3], :]) - qvaluefunction[rs[0], rs[1], rs[2], rs[3], ra])
            s = s_ 
            ds = dstate(s)
            t+=1
        if eps%100 == 0:
        # valuefunction = np.mean(np.max(qvaluefunction, 2))
            print("Generating episode: ",eps + 1, t)
        # print("Terminal State: ", s_)
        steps.append(t)
        eps+=1
    cumulativet = cumulativet[1:]
    return steps, cumulativet

bins = 12
epsilon = 0
runs = 1
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
ax[1].set_yticks(range(0, 300, 50))
ax[1].tick_params(labelright=True)
plt.show()








