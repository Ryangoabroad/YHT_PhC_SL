#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
from gym.envs.registration import register


# In[4]:


logger = logging.getLogger(__name__)

register(
    id='mpbEnv-v0',
    entry_point='.envs:mpbEnv',
    max_episode_steps=1000,
    reward_threshold=150,
)


# In[ ]:




