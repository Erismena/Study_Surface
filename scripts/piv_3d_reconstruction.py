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
import fluids2d.spectra as spectra
from mpl_toolkits.mplot3d import Axes3D
import matplotlib


parent_folder = r'C:\Users\Luc Deike\data_comp3_C\180323\\'
#cine_name = r''

need2rotate = False


show = 'bubbles'

if show == 'bubbles':
    cine_name = r'piv_scanning_bubblesRising_angledUp_galvo100Hz_galvoAmp500mV_fps1000'  
    vmin = -0.005
    vmax = 0.005
    piv_start_frames = [3,4,5,6,7] # bubbles
    
elif show=='pumps':
    cine_name = r'piv_scanning_galvo100Hz_galvoAmp1V_fps1000_pumpsFiringTogether'  
    vmin = -0.08
    vmax = 0.08
    piv_start_frames = [4,5,6,7,8]  # pumps

case_name = cine_name

#piv_start_frames = [0,1,2,3,4,5,6,7,8,9]*2
#piv_start_frames = # pumps
#
piv_list = [pickle.load(open(parent_folder+case_name+'_start_frame_'+str(sf)+'.pkl')) for sf in piv_start_frames]


for p in piv_list:
    p.associate_flowfield()
    
n_rows = np.shape(p.data.ff)[1]
n_cols = np.shape(p.data.ff)[2]

cmap_l = plt.cm.seismic
norm_l = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax )

cmap_r = plt.cm.PuOr
norm_r = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax )

X,Y = np.meshgrid(range(n_rows),range(n_cols))


'''
Load the data and make the scalers
'''

num_to_avg = 5

num_avgd_frames = len(p.data.ff)/num_to_avg

all_data = np.zeros((num_avgd_frames,n_rows,n_cols,len(piv_list),2))

#fig,axs = plt.subplots(1,len(piv_list),figsize=(15,7))
fig = plt.figure(figsize=(10,9))
ax_3dl = fig.add_subplot(121,projection='3d')
ax_3dr = fig.add_subplot(122,projection='3d')


figfolder = r'C:\Users\Luc Deike\data_comp3_C\180323\scanning_pumps_frames\\'

fi = 0

#ax_3d.invert_yaxis()

for afi in range(num_avgd_frames):
    
    for ax_3d in [ax_3dl,ax_3dr]:
        ax_3d.clear()
        ax_3d.set_zlim(0,4)
        ax_3d.view_init(13.08, -77.9)
    
    
    '''
    Draw the planes on the 3d plot
    '''
    for pi,p in enumerate(piv_list):
        
        #ax = axs[pi]
        
        
        
        avgd_flow = np.nanmean(p.data.ff[afi*num_to_avg:afi*(num_to_avg+1),...],axis=0)
        
        flow_speed = np.sqrt(avgd_flow[...,0]**2+avgd_flow[...,1]**2)
        
        #ax.imshow(flow_speed,vmin=0,vmax=0.2)
        
        colors_l = cmap_l(norm_l(np.flipud(avgd_flow[:,:,0])))
        #ax_3dl.plot_surface(X,Y,np.zeros_like(X)+pi, cstride=1, rstride=1, facecolors=colors_l, linewidth=0, edgecolor=None, antialiased=False)
        
        colors_r = cmap_r(norm_r(np.flipud(avgd_flow[:,:,1])))
        #ax_3dr.plot_surface(X,Y,np.zeros_like(X)+pi, cstride=1, rstride=1, facecolors=colors_r, linewidth=0, edgecolor=None, antialiased=False)
        
        all_data[afi,:,:,pi,:] = avgd_flow.copy()
        
        #plt.pause(0.5)
        
    ax_3d.set_title('t = '+str(afi*num_to_avg*p.dt_frames)+' s')
#    '''
#    Draw individual planes in 2d
#    '''
#    for pi,p in enumerate(piv_list):
#        
#        ax_plane.clear()        
#        
#        
#        avgd_flow = np.nanmean(p.data.ff[afi*num_to_avg:afi*(num_to_avg+1),...],axis=0)
#        
#        flow_speed = np.sqrt(avgd_flow[...,0]**2+avgd_flow[...,1]**2)
#        
#        #ax.imshow(flow_speed,vmin=0,vmax=0.2)
#        
#        colors = cmap(norm(flow_speed))
#        #colors[...,3] = 0.5
#
#        #ax.plot_surface(X,Y,np.zeros_like(X)+pi, cstride=1, rstride=1, facecolors=colors, linewidth=0)
#        ax_plane.imshow(cmap(norm(flow_speed)))
#        
#        # highlight the plane on the 3d plot
#        rect = ax_3d.plot([np.min(X),np.min(X),np.max(X),np.max(X),np.min(X)],[np.min(Y),np.max(Y),np.max(Y),np.min(Y),np.min(Y)],zs=[pi]*5,color=[0,1,0])
#        
#        #plt.show()
#        #plt.pause(0.5)
#        
#        
    #fi = fi+1
    #fig.savefig(figfolder+'frame_'+str(fi)+'.png')
                
    #plt.show()
    #plt.pause(1)
    
plt.figure()
plt.imshow(np.nanmean(all_data[:,5:10,0,:,0],axis=1),aspect='auto')

mean_flow = np.nanmean(all_data,axis=0)
fluc = all_data - mean_flow
u_rms = np.sqrt(np.nanmean(fluc[...,0]**2+fluc[...,1]**2,axis=0))

fig,axs = plt.subplots(3,5,figsize=(15,8),sharex=True,sharey=True)
[axs[0,i].imshow(u_rms[:,:,i],vmin=0,vmax=vmax) for i in range(len(piv_list))]
[axs[1,i].imshow(mean_flow[:,:,i,0],vmin=vmin,vmax=vmax,cmap='PuOr') for i in range(len(piv_list))]
[axs[2,i].imshow(mean_flow[:,:,i,1],vmin=vmin,vmax=vmax,cmap='seismic') for i in range(len(piv_list))]


plt.figure()
plt.imshow(np.nanmean(mean_flow[:,5:10,:,0].T,axis=1),vmin=vmin,vmax=vmax,cmap='PuOr')
plt.figure()
plt.imshow(np.nanmean(mean_flow[5:10,:,:,0].T,axis=0),vmin=vmin,vmax=vmax,cmap='PuOr')
