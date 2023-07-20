import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import deque,namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


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


def terminal(s):
	s = s.view(-1).tolist()
	x, v, angularx, angularv = s[0], s[1], s[2], s[3]
	if s[0] <= -2.4 or s[0] >= 2.4:
		return True
	if s[2] <= -0.2095 or s[2] >= 0.2095:
		return True

def reward(s, a, s_):
	return 1
	

class DQN(nn.Module):
	def __init__(self):
		super(DQN, self).__init__()
		self.input = nn.Linear(4, 64)
		self.hiddenlayer = nn.Linear(64, 64)
		self.output = nn.Linear(64, 2)
	def forward(self, x):
		x = self.input(x)
		x = F.relu(x)
		x = self.hiddenlayer(x)
		x = F.relu(x)
		q = self.output(x)
		return q


def action(s, epsilon):
	with torch.no_grad():
		if random.uniform(0,1) < epsilon:
			a = np.random.choice([0, 1])
			return torch.tensor(a).view(1,1)	
		else:
			return policy(s).argmax().view(1,1)		

def dqn():
	optimizer = optim.AdamW(policy.parameters(), lr=1e-4, amsgrad=True)
	memory = deque([],1000)
	memory_recall_size = 128
	steps = []
	cumulativet=[0]
	gamma = 1
	epsilon = 0.9
	tau = 0.005
	eps = 0
	while True and eps < 1000:
		cumulativet.append(cumulativet[-1])
		x = random.uniform(-0.05, +0.05)
		v = random.uniform(-0.05, +0.05)
		angularx = random.uniform(-0.05, +0.05)
		angularv = random.uniform(-0.05, +0.05)
		s = (x, v, angularx, angularv)
		ts = torch.tensor(s, dtype=torch.float).view(1,-1)
		te = 0
		while not terminal(ts) and te < 200:
			cumulativet[-1]+=1
			a = action(ts,epsilon)
			# print(a)
			s_ = transition(s, a)
			R = reward(s, a, s_)
			ts_ = torch.tensor(s_, dtype=torch.float).view(1,-1)
			tR = torch.tensor([R])
			memory.append((ts, a, ts_, tR))
			s = s_
			ts = torch.tensor(s, dtype=torch.float).view(1,-1)
			if (len(memory) > memory_recall_size):
				rsample = random.sample(memory, memory_recall_size)
				states = []
				rewards = []
				actions = []
				nextstates = []
				for i in rsample:
					states.append(i[0])
					actions.append(i[1])
					nextstates.append(i[2])
					rewards.append(i[3])
				states = torch.stack(states).squeeze(1)
				actions = torch.stack(actions).squeeze(1)
				rewards = torch.stack(rewards).squeeze(1)

				qestimatelist = [] #q value for state, for action 
				qestimates = policy(states)
				for i in range(memory_recall_size):
					qestimatelist.append(qestimates[i, actions[i]])
				qestimatelist = torch.cat(qestimatelist)

				#target q value estimates
				nextqestimates = torch.zeros(memory_recall_size)
				for i in range(len(nextstates)):
					if terminal(nextstates[i]):
						nextqestimates[i] = 0
					else:
						with torch.no_grad():
							nextqestimates[i] = target(nextstates[i]).max().unsqueeze(0)
				targetqestimates = nextqestimates*gamma + rewards
				
				#loss
				criterion = nn.SmoothL1Loss()
				loss = criterion(qestimatelist.squeeze(), targetqestimates)
				optimizer.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_value_(policy.parameters(), 100)
				optimizer.step()
			target_params = target.state_dict()
			policy_params = policy.state_dict()
			#updating estimates
			for i in policy_params:
				target_params[i] = policy_params[i]*tau + target_params[i]*(1-tau)
			target.load_state_dict(target_params)
			te+=1
			epsilon *=.999
			if(epsilon<0.05):
				epsilon = 0.05
		steps.append(te)
		eps += 1
		print("Episode",eps,"duration",te)
	cumulativet = cumulativet[1:]
	return steps, cumulativet


runs = 1
tarray = [] 
cumulativetarray = [] 
for i in range(runs):
	print("RUN: ", i + 1)
	policy = DQN()
	target = DQN()
	target.load_state_dict(policy.state_dict())
	dqnobj = dqn()
	tarray.append(dqnobj[0])
	cumulativetarray.append(dqnobj[1])
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
