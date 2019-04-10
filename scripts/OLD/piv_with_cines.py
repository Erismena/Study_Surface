# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 19:06:59 2017

@author: user
"""

import matplotlib.pyplot as plt
import pims
import numpy as np
import fluids2d.piv as piv
import pickle
import fluids2d.geometry


parent_folder = r'E:\Experiments_Stephane\Grid column\PIV_measurements\Hollow_grid_3\20170918\\'
case_name = r'PIV_sv_fps4k_grid3x3x10_withCap_topRegion_Cam_20861_Cine9'

p = pickle.load(open(parent_folder+case_name+'.pkl'))
p.parent_folder = parent_folder
p.associate_flowfield()
ff = p.data.ff

'''
Filter the velocity field
'''

ff=piv.clip_flowfield(ff,0.5)

mean_flow = np.nanmean(ff,axis=0)

cine = pims.open(p.cine_filepath)
im_shape = np.shape(cine[0])

g_orig = fluids2d.geometry.GeometryScaler(dx=p.dx,im_shape=np.shape(cine[0]),origin_pos=(0.1,-0.04),origin_units='m')
g = fluids2d.geometry.create_piv_scaler(p,g_orig)

time = np.arange(0,np.shape(ff)[0]) * p.dt_frames 

frames_to_show = np.arange(0,1)
fig = plt.figure()
ax=fig.add_subplot(111)

plt.tight_layout()

for frame_to_show in frames_to_show:
    a_frame = cine[p.a_frames[frame_to_show]]
    
    inst_flow = ff[frame_to_show,:,:,:]
    speed = np.sqrt( (inst_flow[:,:,0]-mean_flow[:,:,0])**2 + (inst_flow[:,:,1]-mean_flow[:,:,1])**2 )
    speed[speed>0.2] = 0.2
    
    ax.clear()
    
    ax.imshow(a_frame,extent=g_orig.im_extent,cmap='gray')
    ax.imshow(np.linalg.norm(inst_flow,ord=2,axis=2),vmin=0.0,vmax=0.2,alpha=0.2,extent=g.im_extent,cmap='jet') #extent=[np.min(X),np.max(X),np.min(Y),np.max(Y)]
    ax.quiver(g.X,g.Y,ff[frame_to_show,:,:,0],ff[frame_to_show,:,:,1],speed,alpha=0.8,scale=20)
    
    plt.show()
    plt.pause(0.5)