# -*- coding: utf-8 -*-
"""
Created on Wed Oct 04 15:33:49 2017

@author: danjr
"""

import numpy as np
import fluids2d.piv as piv
import fluids2d.geometry
import pickle
from fluids2d.piv import PIVDataProcessing
import matplotlib.pyplot as plt

parent_folder = r'C:\Users\danjr\Documents\Fluids Research\Data\misc\PIV\with_cap_closeUp\\'
case_name = r'PIV_sv_semiCloseUp_fps2k_grid3x3x10_withCap_Cam_20861_Cine5'
need2rotate = True

p = pickle.load(open(parent_folder+case_name+'.pkl'))
p.parent_folder = parent_folder
p.associate_flowfield()

if need2rotate:
    p.data.ff=piv.rotate_data_90(p.data.ff)    
ff = p.data.ff

g_orig = fluids2d.geometry.GeometryScaler(dx=p.dx,im_shape=(800,1280),origin_pos=(0.12,-1),origin_units='m')
g = fluids2d.geometry.create_piv_scaler(p,g_orig)

time = np.arange(0,np.shape(ff)[0]) * p.dt_frames

'''
Filter the velocity field
'''
ff=piv.clip_flowfield(ff,0.5)

grads = piv.compute_gradients(ff)



import pandas as pd

point = [20,30]
axis=1
daxis=0

point_data = ff[:,point[0],point[1],axis]
point_grad = grads[:,point[0],point[1],axis,daxis]

s_data = pd.Series(index=time,data=point_data)
s_grad = pd.Series(index=time,data=point_grad)

fig=plt.figure()
ax1=fig.add_subplot(221)
ax2=fig.add_subplot(222)
ax3=fig.add_subplot(223)
ax4=fig.add_subplot(224)

s_data.plot(ax=ax1)
s_grad.plot(ax=ax3)

pd.plotting.autocorrelation_plot(s_data,ax=ax2)
pd.plotting.autocorrelation_plot(s_grad,ax=ax4)