#!/usr/bin/env python
# coding: utf-8

# In[7]:


import sys
import gym
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
import scipy

# In[8]:
class mpbRl():
    def _init_(self):
        pass
    def find_nearest(self,array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx],idx

    def numpy_get_roots_with_two_piecewise_linear_by_same_X(self,Y1 , Y2 , X , plot = 0):
        # 注1：Y1与Y2长度相等，每条直线段最多只能有一个交点（不可能有两个）
        # 注1：X的值必须由小到大（X须为单调递增序列）
        if X == None:
            X = [i+1 for i in range(len(Y1))]
        else:
            pass

        roots = []
        for i in range(len(Y1)-1):
            Y_i_1 = [Y1[i], Y1[i+1]]
            Y_i_2 = [Y2[i], Y2[i+1]]
            X_i = [X[i], X[i+1]]

            poly_i_1 = np.poly1d(np.polyfit(X_i, Y_i_1, 1))
            poly_i_2 = np.poly1d(np.polyfit(X_i, Y_i_2, 1))
            poly_delta = poly_i_1 - poly_i_2

            # 非水平直线函数poly_delta必定有且只有一个解
            root_i = poly_delta.roots[0]
            y = poly_i_1(root_i)

            # 减少精度，找回因数据精度损失造成节点处丢失的根
            root_i = round(root_i, 6)
            y = round(y, 6)
            #print((root_i, y), X_i)
            if root_i >=  X_i[0] and root_i <= X_i[1]:
                #print((root_i, y), X_i)
                roots.append(root_i)

        roots = list(set(roots))
        roots.sort()
        return roots

    
    def adjustdesignparams(self,dy,dx,Ra,Rb,r2):
        num_bands = 14
        resolution = 16
        r=0.33+r2
        geometry = []
        geometry_lattice = mp.Lattice(size=mp.Vector3(1,np.sqrt(3)*6))
        #colomn 1 btm
        for i in range(2):
            geometry.append(mp.Cylinder(r, center=mp.Vector3(-1/2+i,-3*np.sqrt(3)),  material=mp.air))
            geometry.append(mp.Cylinder(r, center=mp.Vector3(-1/2+i,-2*np.sqrt(3)),  material=mp.air))
            geometry.append(mp.Cylinder(r, center=mp.Vector3(-1/2+i, -1*np.sqrt(3)),material=mp.air))
        #colomn 1 up

            geometry.append(mp.Cylinder(r, center=mp.Vector3(-1/2+i, 1*np.sqrt(3)), material=mp.air))
            geometry.append(mp.Cylinder(r, center=mp.Vector3(-1/2+i,2*np.sqrt(3)) , material=mp.air))
            geometry.append(mp.Cylinder(r, center=mp.Vector3(-1/2+i,3*np.sqrt(3)) , material=mp.air))
        #colomn 2 btm
        for j in range(1):
            geometry.append(mp.Cylinder(r, center=mp.Vector3(0,-5/2*np.sqrt(3)), material=mp.air))
            geometry.append(mp.Cylinder(r, center=mp.Vector3(0,-3/2*np.sqrt(3)), material=mp.air))
            geometry.append(mp.Ellipsoid(center=mp.Vector3(dx,-1/2*np.sqrt(3)+dy),size=mp.Vector3(2*(Ra+0.3),2*(Rb+0.34),mp.inf),e1=mp.Vector3(1,0),e2=mp.Vector3(0,1),
                                         material=mp.air))
        #colomn 2 up
            geometry.append(mp.Ellipsoid(center=mp.Vector3(dx,1*np.sqrt(3)/2-dy),size=mp.Vector3(2*(Ra+0.3),2*(Rb+0.34),mp.inf), e1=mp.Vector3(1,0),e2=mp.Vector3(0,1),
                                         material=mp.air))
            geometry.append(mp.Cylinder(r, center=mp.Vector3(0,3/2*np.sqrt(3)),material=mp.air))
            geometry.append(mp.Cylinder(r, center=mp.Vector3(0,5/2*np.sqrt(3)),material=mp.air))
        #     geometry.append(mp.Ellipsoid(center=mp.Vector3(w1_0+k*d,0.7*d), size=mp.Vector3(2*s_a,2*s_b,0.01),
        #                e1=mp.Vector3(1,0),e2=mp.Vector3(0,1),material=mp.air))
        #     geometry.append(mp.Ellipsoid(center=mp.Vector3(w1_0+k*d,-1/2*np.sqrt(3)*d), size=mp.Vector3(2*s_a,2*s_b,0.01),
        #                 e1=mp.Vector3(1,0),e2=mp.Vector3(0,1),material=mp.air))


        #geometry.append(mp.Block(center=mp.Vector3(w1_0,1/2*np.sqrt(3)*d), size=mp.Vector3(s_a,s_b),
                       #material=mp.Medium(epsilon=2.83)))
        #geometry.append(mp.Block(center=mp.Vector3(w1_0,-1/2*np.sqrt(3)*d), size=mp.Vector3(s_a,s_b),
                        #material=mp.Medium(epsilon=2.83)))
        k_points = [
            #mp.Vector3(),                           # Gamma
            mp.Vector3(0.28, 0),   
            mp.Vector3(1/2, 0),          # K
            #mp.Vector3(0.58, 0),   
            #mp.Vector3(1, 0),          # Gamma
        ]
        k_points = mp.interpolate(34, k_points)

        ms = mpb.ModeSolver(
            geometry=geometry,
            geometry_lattice=geometry_lattice,
            k_points=k_points,
            resolution=resolution,
            num_bands=num_bands,
            default_material=mp.Medium(index=2.83)
        )
        hfields = []

        def get_hfields(ms, band):
            hfields.append(ms.get_hfield(band, bloch_phase=True))

        ms.run_te(mpb.output_at_kpoint(mp.Vector3(0, 1/2), mpb.fix_hfield_phase,
                  get_hfields))
        te_freqs = ms.all_freqs
        te_gaps = ms.gap_list

        y =te_freqs[:,12]
        x = np.linspace(0.28,0.5,len(y))
        diff_x = []                       # 存储x列表中的两数之差
        for i in range(len(y)-1):
            diff = x[i+1]-x[i]
            diff_x.append(diff)
        diff_y = []                       # 存储y列表中的两数之差
        for i in range(len(y)-1):
            diff = abs(y[i+1]-y[i])
            diff_y.append(diff)
        slopes2 = []                       # 存储斜率
        for i in range(len(diff_y)):
            slopes2.append(diff_x[i] / diff_y[i])
        #slopes
        slopes= []                        # 存储一阶导数
        for i, j in zip(slopes2[0::], slopes2[1::]):        
            slopes.append((0.5 * (i + j))) # 根据离散点导数的定义，计算并存储结果
        slopes.insert(0, slopes[0])        # (左)端点的导数即为与其最近点的斜率
        slopes.append(slopes[-1])  
        diff_w = []                       # 存储x列表中的两数之差
        for i in range(len(y)):
            diff = 412/abs(y[i])
            diff_w.append(diff)
        wavelength=[]
        ng=[]
        freqs=[]
        for i in range(len(y)-1):
            if abs(slopes[i+1]-slopes[i])<=60 and slopes[i]>=5:
                wavelength.append(diff_w[i])
                ng.append(slopes[i])
                freqs.append(y[i])
            elif abs(slopes[i+1]-slopes[i])>60 and slopes[i]>=5:
                break
        ng_array=np.array(ng)
        peak_indexes = scipy.signal.argrelextrema(ng_array, np.greater, order=1)
        peak_indexes = peak_indexes[0]
        valley_indexes = scipy.signal.argrelextrema(ng_array, np.less, order=1)
        valley_indexes = valley_indexes[0]#求极值点
        if peak_indexes.size==0:
            root1=[]
            peak_value=0
        if peak_indexes.size==1:
            peak_value=ng_array[int(peak_indexes)]
        #     high_line=[peak_value]*len(ng)
        #     root1= numpy_get_roots_with_two_piecewise_linear_by_same_X(Y1=high_line,Y2=ng,X=wavelength)
        elif peak_indexes.size>1:
            peak_value=ng_array[peak_indexes]
            peak_value=max(peak_value)
        #     high_line=[peak_value]*len(ng)
        #     root1= numpy_get_roots_with_two_piecewise_linear_by_same_X(Y1=high_line,Y2=ng,X=wavelength)
        # Plot valleys
        if valley_indexes.size==0:
            root2=[]
            valley_value=0
        if valley_indexes.size==1:
            valley_value=ng_array[int(valley_indexes)]
        #     low_line=[valley_value]*len(ng)
        #     root2= numpy_get_roots_with_two_piecewise_linear_by_same_X(Y1=low_line,Y2=ng,X=wavelength)
        elif valley_indexes.size>1:
            valley_value=ng_array[valley_indexes]
            valley_value=min(valley_value)
        #     low_line=[valley_value]*len(ng)
        #     root2= numpy_get_roots_with_two_piecewise_linear_by_same_X(Y1=low_line,Y2=ng,X=wavelength)

        ng_value=(peak_value+valley_value)/2
        ng_cha=peak_value-valley_value
        high_line=[ng_value*1.1]*len(ng)
        root1= self.numpy_get_roots_with_two_piecewise_linear_by_same_X(Y1=high_line,Y2=ng,X=wavelength)
        low_line=[ng_value*0.9]*len(ng)
        root2=self.numpy_get_roots_with_two_piecewise_linear_by_same_X(Y1=low_line,Y2=ng,X=wavelength)
        # Plot valleys
        if ng_cha>0.2*ng_value:
            bandwidth=0
            NDBP=0
            ng_value_average=0
        else:
            if len(root1)==0 or len(root2)==0:
                bandwidth=0
                NDBP=0
                ng_value_average=0
#             if len(root1)==1 and len(root2)==1:
#                 bandwidth=wavelength[-1]-wavelength[0]
#                 w1=412/wavelength[-1]
#                 w2=412/wavelength[0]
#                 w1_ng,w1_index=self.find_nearest(te_freq,w1)
#                 w2_ng,w2_index=self.find_nearest(te_freq,w2)
#                 ng_value_list=[]
#                 for w_h,w_l,ng_w in zip(te_freq[w2_index:w1_index+1],te_freq[w2_index+1:w1_index+2],slopes[w2_index:w1_index+1]):
#                     ng_value_list.append(ng_w*(w_h-w_l))
#                 ng_value_average=sum(ng_value_list)/(w2_ng-w1_ng)
#                 NDBP=ng_value_average*(te_freq[w2_index]-te_freq[w1_index])/(te_freq[w2_index]+te_freq[w1_index])
#             elif len(root1)==1 and len(root2)>1:
#                 bandwidth=abs(root2[0]-wavelength[0])
#                 w1=412/root2[0]
#                 w2=412/wavelength[0]
#                 w1_ng,w1_index=self.find_nearest(te_freq,w1)
#                 w2_ng,w2_index=self.find_nearest(te_freq,w2)
#                 ng_value_list=[]
#                 for w_h,w_l,ng_w in zip(te_freq[w2_index:w1_index+1],te_freq[w2_index+1:w1_index+2],slopes[w2_index:w1_index+1]):
#                     ng_value_list.append(ng_w*(w_h-w_l))
#                 ng_value_average=sum(ng_value_list)/(w2_ng-w1_ng)
#                 NDBP=ng_value_average*(te_freq[w2_index]-te_freq[w1_index])/(te_freq[w2_index]+te_freq[w1_index])*2
#             #     NDBP=ng_value_average*(te_freq[w2_index]-te_freq[w1_index])/(te_freq[w2_index]+te_freq[w1_index])
#             # elif len(root1)!=0 and len(root2)==0:
#             #     bandwidth=root1[-1]-0
#             #     NDBP=0
#             #     ng_value_average=0
#             elif len(root1)>1 and len(root2)==1:
#                 bandwidth=abs(root1[0]-wavelength[-1])
#                 w1=412/wavelength[-1]
#                 w2=412/root1[0]
#                 w1_ng,w1_index=self.find_nearest(te_freq,w1)
#                 w2_ng,w2_index=self.find_nearest(te_freq,w2)
#                 ng_value_list=[]
#                 for w_h,w_l,ng_w in zip(te_freq[w2_index:w1_index+1],te_freq[w2_index+1:w1_index+2],slopes[w2_index:w1_index+1]):
#                     ng_value_list.append(ng_w*(w_h-w_l))
#                 ng_value_average=sum(ng_value_list)/(w2_ng-w1_ng)
#                 NDBP=ng_value_average*(te_freq[w2_index]-te_freq[w1_index])/(te_freq[w2_index]+te_freq[w1_index])*2
            elif len(root1)>=1 and len(root2)>=1:
                bandwidth=abs(root1[-1]-root2[0])
                w1=412/root1[-1]
                w2=412/root2[0]
                if w1<max(te_freqs[:,11]) or w2>min(te_freqs[:,13]):
                    bandwidth=0
                    NDBP=0
                    ng_value_average=0
                else:
                    w1_ng,w1_index=self.find_nearest(y,w1)
                    w2_ng,w2_index=self.find_nearest(y,w2)
                    if w1_ng>w1 and w2_ng<w2:
                        y=np.insert(y,w1_index+1, w1)
                        y=np.insert(y,w2_index, w2)
                        slopes=np.insert(slopes,w1_index+1,ng_value*1.1)
                        slopes=np.insert(slopes,w2_index,ng_value*0.9)
                        ng_value_list=[]
                        for w_h,w_l,ng_w in zip(y[w2_index:w1_index+3],y[w2_index+1:w1_index+4],slopes[w2_index:w1_index+3]):
                            ng_value_list.append(ng_w*(w_h-w_l))
                        ng_value_average=sum(ng_value_list)/(w2-w1)
                        NDBP=ng_value_average*(y[w2_index]-y[w1_index+2])/(y[w2_index]+y[w1_index+2])*2
                    elif w1_ng<w1 and w2_ng<w2:
                        y=np.insert(y,w1_index, w1)
                        y=np.insert(y,w2_index, w2)
                        slopes=np.insert(slopes,w1_index,ng_value*1.1)
                        slopes=np.insert(slopes,w2_index,ng_value*0.9)
                        ng_value_list=[]
                        for w_h,w_l,ng_w in zip(y[w2_index:w1_index+2],y[w2_index+1:w1_index+3],slopes[w2_index:w1_index+2]):
                            ng_value_list.append(ng_w*(w_h-w_l))
                        ng_value_average=sum(ng_value_list)/(w2-w1)
                        NDBP=ng_value_average*(y[w2_index]-y[w1_index+1])/(y[w2_index]+y[w1_index+1])*2
                    elif w1_ng<w1 and w2_ng>w2:
                        y=np.insert(y,w1_index, w1)
                        y=np.insert(y,w2_index+1, w2)
                        slopes=np.insert(slopes,w1_index,ng_value*1.1)
                        slopes=np.insert(slopes,w2_index+1,ng_value*0.9)
                        ng_value_list=[]
                        for w_h,w_l,ng_w in zip(y[w2_index+1:w1_index+2],y[w2_index+2:w1_index+3],slopes[w2_index+1:w1_index+2]):
                            ng_value_list.append(ng_w*(w_h-w_l))
                        ng_value_average=sum(ng_value_list)/(w2-w1)
                        NDBP=ng_value_average*(y[w2_index+1]-y[w1_index+1])/(y[w2_index+1]+y[w1_index+1])*2
                    elif w1_ng>w1 and w2_ng>w2:
                        y=np.insert(y,w1_index+1, w1)
                        y=np.insert(y,w2_index+1, w2)
                        slopes=np.insert(slopes,w1_index+1,ng_value*1.1)
                        slopes=np.insert(slopes,w2_index+1,ng_value*0.9)            
                        ng_value_list=[]
                        for w_h,w_l,ng_w in zip(y[w2_index+1:w1_index+3],y[w2_index+2:w1_index+4],slopes[w2_index+1:w1_index+3]):
                            ng_value_list.append(ng_w*(w_h-w_l))
                        ng_value_average=sum(ng_value_list)/(w2-w1)
                        NDBP=ng_value_average*(y[w2_index+1]-y[w1_index+2])/(y[w2_index+1]+y[w1_index+2])*2
        print(NDBP,ng_value_average,bandwidth)
        return NDBP,ng_value_average,bandwidth


# In[ ]:




