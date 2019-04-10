# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:08:23 2017

@author: danjr
"""

import fluids2d.piv as piv
import matplotlib.pyplot as plt
import pickle
import numpy as np
import scipy.ndimage
from fluids2d.masking import MovingRectMasks
import fluids2d.geometry
import pims
from fluids2d.piv import PIVDataProcessing
import pandas as pd
import fluids2d.spectra as spectra

parent_folder = r'C:\Users\Luc Deike\data_comp3_C\180228\\'
cine_name = r'piv_4x4_center_sunbathing_on100_off400_fps1000'
case_name = cine_name

need2rotate = False

vmin = -0.2
vmax = 0.2

'''
Load the data and make the scalers
'''

p = pickle.load(open(parent_folder+case_name+'.pkl'))
p.parent_folder = parent_folder
p.name_for_save = case_name
p.associate_flowfield()
dt = p.dt_frames

if need2rotate:
    p.data.ff=piv.rotate_data_90(p.data.ff)
    p.data.ff=piv.rotate_data_90(p.data.ff)
    p.data.ff=piv.rotate_data_90(p.data.ff)
ff = p.data.ff

g_orig = fluids2d.geometry.GeometryScaler(dx=p.dx,im_shape=(2048,2048),origin_pos=(-1024,-1024),origin_units='pix')
#g_orig = fluids2d.geometry.GeometryScaler(dx=p.dx,im_shape=(512,512),origin_pos=(-400,-50),origin_units='pix')
g = fluids2d.geometry.create_piv_scaler(p,g_orig)

'''
Filter the velocity field
'''
ff=piv.clip_flowfield(ff,5)


start_frames = np.arange(0,5000,10)

x = [[np.random.uniform(-0.05,0.05)] for _ in start_frames]
y = [[np.random.uniform(-0.05,0.05)] for _ in start_frames]

frames_after_drop = np.arange(0,1000)
for fi,f in enumerate(frames_after_drop):
    
    for si,s in enumerate(start_frames):
    
        [loc_y,loc_x] = g.get_coords([y[si][-1],x[si][-1]])
        
        if loc_y==g.im_shape[0] or loc_y==0 or loc_x==g.im_shape[1] or loc_y==0:
            u = np.nan
            v = np.nan
            
        else:
            u = ff[f+s,loc_y,loc_x,0]
            v = ff[f+s,loc_y,loc_x,1]
        
        y[si].append(y[si][-1]+dt*v)
        x[si].append(x[si][-1]+dt*u)
    
fig = plt.figure()
ax = fig.add_subplot(111)

[ax.plot(X,Y,alpha=0.2) for X,Y in zip(x,y)]
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
fig.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(111)

[ax.plot(np.array(X)-X[0],np.array(Y)-Y[0]) for X,Y in zip(x,y)]


mean_displacement_list = []
lags_list = []
autocorr_lags_list = []
autocorr_corr_list = []
for bi,(x_vec,y_vec) in enumerate(zip(x,y)):
    
    
    x_vec = np.array(x_vec)
    y_vec = np.array(y_vec)
    
    print(bi)
    lags = np.arange(0,min(len(x_vec),1000)-2,1)
    mean_displacement = np.zeros(np.shape(lags))
    
    for li,lag in enumerate(lags):
        mean_displacement[li] = np.sqrt(np.nanmean((x_vec[0:len(x_vec)-lag]-x_vec[lag:len(x_vec)])**2 + (y_vec[0:len(x_vec)-lag]-y_vec[lag:len(x_vec)])**2))
        
    lags_list.append(lags)
    mean_displacement_list.append(mean_displacement)
    
    u = np.gradient(np.array(x_vec))/dt
    C,lags = spectra.autocorr(u,i_lags=np.arange(len(lags)),dt=dt)
    autocorr_lags_list.append(lags)
    autocorr_corr_list.append(C)
    
    
fig = plt.figure()
ax = fig.add_subplot(111)
urms = 0.1
[ax.loglog(l*dt,md**2/urms**2,alpha=0.2) for l,md in zip(lags_list,mean_displacement_list)]
ax.plot([.01,.1],[10**-2,10**0],'--',color='k',alpha=0.5)
ax.set_xlabel('lag [s]')
ax.set_ylabel('MSD / u_rms [s^2]')
fig.tight_layout()

fig = plt.figure()
ax_c = fig.add_subplot(111)
[ax_c.plot(l,c,alpha=0.1) for l,c in zip(autocorr_lags_list,autocorr_corr_list)]

fig = plt.figure()
ax_c = fig.add_subplot(111)
[ax_c.plot(l,np.cumsum(c)*(l[1]-l[0]),alpha=0.1) for l,c in zip(autocorr_lags_list,autocorr_corr_list)]
ax_c.set_xlabel('lag [s]')
ax_c.set_ylabel('integral of lagrangian autocorrelation [s]')
fig.tight_layout()








#
#x_bins = np.arange(-0.08,0.08,.005)
#y_bins = np.arange(-0.08,0.08,.005)
#X,Y = np.meshgrid(x_bins,y_bins)
#
#counts = np.zeros(np.shape(X))
#counts_norm = np.zeros(np.shape(X))
#end_y = np.zeros(np.shape(X))
#
#r_df = pd.DataFrame(index=frames_after_drop)
#
#ci = -1
#for x_vec,y_vec in zip(x,y):
#    ci = ci+1
#    for xi,yi in zip(x_vec,y_vec):
#        if ~np.isnan(xi) and ~np.isnan(yi):
#            
#            ind_y = np.digitize(yi,y_bins)
#            ind_x = np.digitize(xi,x_bins)
#            counts[ind_y,ind_x] = counts[ind_y,ind_x]+1
#            
#            ind_y_norm = np.digitize(yi-y_vec[0],y_bins)
#            ind_x_norm = np.digitize(xi-x_vec[0],x_bins)
#            counts_norm[ind_y_norm,ind_x_norm] = counts_norm[ind_y_norm,ind_x_norm]+1
#            
#    if ~np.isnan(x_vec[0]) and ~np.isnan(y_vec[0]):
#            
#        ind_start_y = np.digitize(y_vec[0],y_bins)
#        ind_start_x = np.digitize(x_vec[0],x_bins)    
#        y_no_nans = [i for i in y_vec if ~np.isnan(i)]        
#        end_y[ind_start_y,ind_start_x] = end_y[ind_start_y,ind_start_x] + y_no_nans[-1]
#        
#        r_vec = np.sqrt((np.array(x_vec)-x_vec[0])**2+(np.array(y_vec)-y_vec[0])**2)
#        r_df[ci] = r_vec[0:-1]
#             
#fig=plt.figure()
#ax=fig.add_subplot(111)
#ax.imshow(counts,vmin=0,vmax=10000,extent=g.im_extent,origin='lower')
#
#fig=plt.figure()
#ax=fig.add_subplot(111)
#ax.imshow(counts_norm,vmin=0,vmax=10000,extent=g.im_extent,origin='lower')
#
#fig=plt.figure()
#ax=fig.add_subplot(111)
#ax.imshow(end_y,extent=g.im_extent,origin='lower')
#
##r_df.index = r_df.index*dt
#r_mean = r_df.mean(axis=1)
#
#fig = plt.figure()
#ax = fig.add_subplot(111)
#
#ax.plot(r_mean,color='k',lw=2)
#ax.plot(r_mean+r_df.std(axis=1),color='gray')
#ax.plot(r_mean-r_df.std(axis=1),color='gray')
#
#ax.set_xlabel('time [s]')
#ax.set_ylabel('distance traveled [m]')
#
