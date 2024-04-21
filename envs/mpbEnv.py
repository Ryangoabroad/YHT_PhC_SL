#!/usr/bin/env python
# coding: utf-8

# In[3]:


# %load mpbEnv.py
#!/usr/bin/env python

# In[3]:


import math
import meep as mp
import numpy as np
from meep import mpb
import meep.materials
import random
import gym
from gym import spaces, logger
from gym import utils
from gym.utils import seeding
from collections import namedtuple, deque
from itertools import count
import subprocess, time, signal
import numpy as np
import sys
from src import mpbRl
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
import logging
from gym.envs.registration import register
import scipy


# In[2]:


class mpbEnv(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self):
        # limits for net geometrical changes (states)
        self.maxdy = 0.2
        self.maxdx = 1
        self.maxRa = 0.1
        self.maxRb = 0.24
        self.maxr = 0.03
        # actions to take (i.e. alter the geometrical parameters)
        # self.delta = 0.5e-9
        # self.DR = 0.25e-9

        self.deltay = 0.001
        self.deltax = 0.001
        self.deltaRa = 0.001
        self.deltaRb = 0.001
        self.deltar = 0.001
        high = np.array(
            [
                self.maxdy * 1.5,
                self.maxdx * 1.5,
                self.maxRa * 1.5,
                self.maxRb * 1.5,
                self.maxr * 1.5,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # best geometrical shift values found so far
        
        self.goal_ng = 80  # optimization goal
        self.goal_NDBP=0.8
        self.goal_bandwidth =50
#         self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
    def step(self,action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action),err_msg

        netdy, netdx, netRa,netRb,netr = self.state

        if action == 0:
            netdy = netdy + self.deltay

        elif action == 1:
            netdy = netdy - self.deltay

        elif action == 2:
            netdx = netdx + self.deltax

        elif action == 3:
            netdx = netdx - self.deltax

        elif action == 4:
            netRa = netRa + self.deltaRa

        elif action == 5:
            netRa = netRa - self.deltaRa
        
        elif action == 6:
            netRb = netRb + self.deltaRb

        elif action == 7:
            netRb = netRb - self.deltaRb
            
        elif action == 8:
            netr = netr - self.deltar
            
        elif action == 9:
            netr = netr + self.deltar
            
        elif action == 10:
            pass

        # perform an action in fdtd and compute Q factor
        SL = mpbRl()
        # define conversion from m to nm
        NDBP,ng,bandwidth= SL.adjustdesignparams(netdy, netdx, netRa, netRb, netr)

        # update the state
        self.state = (netdy, netdx, netRa, netRb, netr)

        done = bool(
            netdx < -self.maxdx
            or netdx > self.maxdx
            or netdy < -self.maxdy
            or netdy > self.maxdy
            or netRa < -self.maxRa
            or netRa > self.maxRa
            or netRb < -self.maxRb
            or netRb > self.maxRb
            or netr < -self.maxr
            or netr > self.maxr         
        )

        # calculate the reward
        if not done:
            if ng<=80:
                r = (100 - (self.goal_bandwidth-bandwidth)*0.3-(self.goal_NDBP - NDBP)*100)
            else:
                r=(20)
            reward = np.float32(r)
        elif self.steps_beyond_done is None:
            # net changes out of limit, game over
            self.steps_beyond_done = 0
            if ng<=80:
                r = (100 - (self.goal_bandwidth-bandwidth)*0.3-(self.goal_NDBP - NDBP)*100)
            else:
                r=(20)
            reward = np.float32(r)
            print('State out of range, done! Restarting a new episode...')
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.dy_optim = round(random.uniform(0,0.16),3)
        self.dx_optim = round(random.uniform(0.08,0.18),3)
        self.Ra_optim = round(random.uniform(0,0.1),3)
        self.Rb_optim = round(random.uniform(-0.24,-0.08),3)
        self.r_optim = round(random.uniform(-0.015,0.015),3)
#         self.state = np.zeros((3,), dtype=np.float32)
#         self.dy_optim =0.1
#         self.dx_optim = 0.1663
#         self.Ra_optim = 0.0751
#         self.Rb_optim =-0.2039
        self.state = (self.dy_optim, self.dx_optim, self.Ra_optim, self.Rb_optim, self.r_optim)
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)


# In[ ]:




