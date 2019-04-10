# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 23:28:34 2017

@author: danjr
"""

import pims
import numpy as np
import openpiv.process
import openpiv.validation
import openpiv.filters
import matplotlib.pyplot as plt
from fluids2d.masking import MovingRectMasks
import pickle
import pandas as pd

import fluids2d.piv as piv
import fluids2d.geometry

parent_folder = r'C:\Users\danjr\Documents\Fluids Research\Data\misc\PIV\\'
fig_folder = r'C:\Users\danjr\Documents\Fluids Research\Writing\grid3_PIV_analysis\figures\\'

keys=('center_05','center_10','lower_05','lower_10')

res_to_load = {'lower_05':r'PIV_svCloseUpBottomThird_makro100mmzeiss_X0mm_Y05mm_fps2500_A15mm_f05Hz_grid3x4_10cycles',
               'center_05':r'PIV_svCloseUpCenterThird_makro100mmzeiss_X0mm_Y15mm_fps2500_A15mm_f05Hz_grid3x4_10cycles',
               'lower_10':r'PIV_svCloseUpBottomThird_makro100mmzeiss_X0mm_Y05mm_fps2500_A15mm_f10Hz_grid3x4_10cycles',
               'center_10':r'PIV_svCloseUpCenterThird_makro100mmzeiss_X0mm_Y15mm_fps2500_A15mm_f10Hz_grid3x4_10cycles'}

a = [5,5,5,10,10,10,15,15,15]
amplitudes = {k:a[i] for i,k in enumerate(keys)}

f = [5,8,10,5,8,10,5,8,10]
frequencies = {k:f[i] for i,k in enumerate(keys)}

p = {key:pickle.load(open(parent_folder+res_to_load[key]+'.pkl')) for key in keys}


geo_center = fluids2d.geometry.GeometryScaler(dx=p['center_05'].dx,origin_pos=0.1)
geo_lower = fluids2d.geometry.GeometryScaler(dx=p['lower_05'].dx,origin_pos=0.2)
g=[geo_lower,geo_center,geo_lower,geo_center]
geo = {k:g[i] for i,k in enumerate(keys)}


for key in keys:
    p[key].parent_folder = parent_folder
    p[key].associate_flowfield()

#p = {key:piv.load_processed_PIV_data(parent_folder,res_to_load[key]) for key in list(res_to_load.keys())}


vmin=0
vmax=0.2

def cleanup_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


u_max=0.9

'''
Filter data, and plot mean flow
'''

import scipy.ndimage
filtered_flowfield={}
fig = plt.figure(figsize=(16,9))
ax = {k:fig.add_subplot(2,2,i+1) for i,k in enumerate(keys)}
for k in keys:
    
    p[k].data.ff[p[k].data.ff>u_max] = u_max
    p[k].data.ff[p[k].data.ff<-1*u_max] = -1*u_max
    filtered_flowfield[k] = scipy.ndimage.median_filter(p[k].data.ff,size=[3,1,1,1])
    #filtered_flowfield[k][filtered_flowfield[k]>u_max]=u_max
    #filtered_flowfield[k][filtered_flowfield[k]<-1*u_max]=-1*u_max
    mean_flow = np.nanmean(filtered_flowfield[k],axis=0)
    #mean_speed = np.sqrt( mean_flow[:,:,0]**2 + mean_flow[:,:,1]**2 )
    
    # mean of the speed at each point
    #mean_speed=np.nanmean(np.sqrt(filtered_flowfield[k][:,:,:,0]**2+filtered_flowfield[k][:,:,:,1]**2),axis=0)
    mean_flow_magnitude = np.sqrt( mean_flow[:,:,0]**2 + mean_flow[:,:,1]**2 )
    
    # speed at ech point and time
    speed_inst=np.sqrt(filtered_flowfield[k][:,:,:,0]**2+filtered_flowfield[k][:,:,:,1]**2)
    
    # square root of the mean of the speed
    u_rms = np.sqrt(np.nanmean(speed_inst**2,axis=0))
    
    ax[k].imshow(mean_flow_magnitude,vmin=vmin,vmax=vmax)
    ax[k].quiver(mean_flow[:,:,0],mean_flow[:,:,1],color=[1,1,1,0.6])
    ax[k].set_title('case '+k)
    
    cleanup_axis(ax[k])



ax[keys[2]].set_xlabel('f = 5 Hz')
ax[keys[3]].set_xlabel('f = 10 Hz')

ax[keys[0]].set_ylabel('center third')
ax[keys[2]].set_ylabel('lower third')

plt.tight_layout()
plt.show()
plt.pause(0.5)

fig.savefig(fig_folder+'mean_flows_highRes.pdf')

'''
RMS velocity
'''

vmin=0
vmax=0.2
fig = plt.figure(figsize=(16,9))
ax1 = {k:fig.add_subplot(2,2,i+1) for i,k in enumerate(keys)}
for k in keys:
    
    # mean velocity at each point
    mean_flow = np.nanmean(filtered_flowfield[k],axis=0)
    
    mean_flow_magnitude = np.sqrt( mean_flow[:,:,0]**2 + mean_flow[:,:,1]**2 )
    
    # fluctuating components of each velocity
    fluc_inst = filtered_flowfield[k] - mean_flow
    
    # square root of the mean of the square of the fluctuations
    up_rms = np.sqrt( np.nanmean(fluc_inst**2,axis=0))
    
    up_rms_abs = np.sqrt( up_rms[:,:,0]**2 + up_rms[:,:,1]**2 )
    
    tke_inst = np.sqrt( fluc_inst[:,:,:,0]**2 + fluc_inst[:,:,:,1]**2 )
    
    ax1[k].imshow(np.nanmean(tke_inst,axis=0),vmin=vmin,vmax=vmax)
    #ax[k].quiver(u_rms[:,:,0],u_rms[:,:,1])
    ax1[k].set_title(k)
    
    cleanup_axis(ax1[k])
    
ax1[keys[2]].set_xlabel('f = 5 Hz')
ax1[keys[3]].set_xlabel('f = 10 Hz')

ax1[keys[0]].set_ylabel('center third')
ax1[keys[2]].set_ylabel('lower third')

plt.tight_layout()
plt.show()
plt.pause(0.5)

fig.savefig(fig_folder+'rms_velocity_highRes.pdf')


'''
Vertical velocity along center column
'''

column_slice = [50,54]

vmin=-0.1
vmax=0.1
fig = plt.figure(figsize=(16,9))
ax1 = {k:fig.add_subplot(2,2,i+1) for i,k in enumerate(keys)}
for k in keys:
    
    #ax1[k].imshow(np.nanmean(filtered_flowfield[k][:,:,column_slice[0]:column_slice[1],1],axis=2).transpose(),origin='upper',vmin=vmin,vmax=vmax,aspect='auto',cmap='seismic',extent=[0,10,0,(np.shape(filtered_flowfield[k])[1]+1)*p[k].dx*(p[k].window_size-p[k].overlap)*100])
    #ax[k].quiver(u_rms[:,:,0],u_rms[:,:,1])
    #ax1[k].set_title(k)
    
    cleanup_axis(ax1[k])
    
ax1[keys[2]].set_xlabel('f = 5 Hz')
ax1[keys[3]].set_xlabel('f = 10 Hz')

ax1[keys[0]].set_ylabel('center third')
ax1[keys[2]].set_ylabel('lower third')

for i in [0,1]:
    ax1[keys[i]].set_xlabel('t * f'); ax1[keys[i]].xaxis.set_label_position('top'); ax1[keys[i]].set_xticks(np.arange(0,11,2)); ax1[keys[i]].xaxis.tick_top(); ax1[keys[i]].spines['top'].set_visible(True)
    
for i in [1,3]:
    ax1[keys[i]].set_ylabel('vertical position [cm]'); ax1[keys[i]].yaxis.set_label_position('right'); ax1[keys[i]].set_yticks(np.arange(0,9,2));  ax1[keys[i]].yaxis.tick_right(); ax1[keys[i]].spines['right'].set_visible(True)

plt.tight_layout()
plt.show()
plt.pause(0.5)

fig.savefig(fig_folder+'central_velocity_vertical_highRes.pdf')

'''
Horizontal velocity along central column
'''

vmin=-0.1
vmax=0.1
fig = plt.figure(figsize=(16,9))
ax1 = {k:fig.add_subplot(2,2,i+1) for i,k in enumerate(keys)}
for k in keys:
    
    #ax1[k].imshow(np.nanmean(filtered_flowfield[k][:,:,column_slice[0]:column_slice[1],0],axis=2).transpose(),origin='upper',vmin=vmin,vmax=vmax,aspect='auto',cmap='PuOr',extent=[0,10,0,(np.shape(filtered_flowfield[k])[1]+1)*p[k].dx*(p[k].window_size-p[k].overlap)*100])
    ax1[k].imshow(np.nanmean(filtered_flowfield[k][:,:,column_slice[0]:column_slice[1],0],axis=2).transpose(),origin='upper',vmin=vmin,vmax=vmax,aspect='auto',cmap='PuOr',extent=[0,10,0,(np.shape(filtered_flowfield[k])[1]+1)*p[k].dx*(p[k].window_size-p[k].overlap)*100])
    #ax[k].quiver(u_rms[:,:,0],u_rms[:,:,1])
    #ax1[k].set_title(k)
    
    cleanup_axis(ax1[k])
    
ax1[keys[2]].set_xlabel('f = 5 Hz')
ax1[keys[3]].set_xlabel('f = 10 Hz')

ax1[keys[0]].set_ylabel('center third')
ax1[keys[2]].set_ylabel('lower third')

for i in [0,1]:
    ax1[keys[i]].set_xlabel('t * f'); ax1[keys[i]].xaxis.set_label_position('top'); ax1[keys[i]].set_xticks(np.arange(0,11,2)); ax1[keys[i]].xaxis.tick_top(); ax1[keys[i]].spines['top'].set_visible(True)
    
for i in [1,3]:
    ax1[keys[i]].set_ylabel('vertical position [cm]'); ax1[keys[i]].yaxis.set_label_position('right'); ax1[keys[i]].set_yticks(np.arange(0,9,2));  ax1[keys[i]].yaxis.tick_right(); ax1[keys[i]].spines['right'].set_visible(True)

plt.tight_layout()
plt.show()
plt.pause(0.5)

fig.savefig(fig_folder+'central_velocity_horizontal_highRes.pdf')

'''
Vertical velocity along central span
'''

span_slice = [28,32]

vmin=-0.1
vmax=0.1
fig = plt.figure(figsize=(16,9))
ax1 = {k:fig.add_subplot(2,2,i+1) for i,k in enumerate(keys)}
for k in keys:
    
    ax1[k].imshow(np.nanmean(filtered_flowfield[k][:,span_slice[0]:span_slice[1],:,1],axis=1).transpose(),vmin=vmin,vmax=vmax,aspect='auto',cmap='seismic',extent=[0,10,0,(np.shape(filtered_flowfield[k])[2]+1)*p[k].dx*(p[k].window_size-p[k].overlap)*100])
    #ax[k].quiver(u_rms[:,:,0],u_rms[:,:,1])
    #ax1[k].set_title(k)
    
    cleanup_axis(ax1[k])
    
ax1[keys[2]].set_xlabel('f = 5 Hz')
ax1[keys[3]].set_xlabel('f = 10 Hz')

ax1[keys[0]].set_ylabel('center third')
ax1[keys[2]].set_ylabel('lower third')

for i in [0,1]:
    ax1[keys[i]].set_xlabel('t * f'); ax1[keys[i]].xaxis.set_label_position('top'); ax1[keys[i]].set_xticks(np.arange(0,11,2)); ax1[keys[i]].xaxis.tick_top(); ax1[keys[i]].spines['top'].set_visible(True)
    
for i in [1,3]:
    ax1[keys[i]].set_ylabel('spanwise position [cm]'); ax1[keys[i]].yaxis.set_label_position('right'); ax1[keys[i]].set_yticks(np.arange(0,16,3));  ax1[keys[i]].yaxis.tick_right(); ax1[keys[i]].spines['right'].set_visible(True)

plt.tight_layout()
plt.show()
plt.pause(0.5)

fig.savefig(fig_folder+'span_velocity_vertical_highRes.pdf')


'''
Horizontal velocity along central span
'''

vmin=-0.1
vmax=0.1
fig = plt.figure(figsize=(16,9))
ax1 = {k:fig.add_subplot(2,2,i+1) for i,k in enumerate(keys)}
for k in keys:
    
    ax1[k].imshow(np.nanmean(filtered_flowfield[k][:,span_slice[0]:span_slice[1],:,0],axis=1).transpose(),vmin=vmin,vmax=vmax,aspect='auto',cmap='PuOr',extent=[0,10,0,(np.shape(filtered_flowfield[k])[2]+1)*p[k].dx*(p[k].window_size-p[k].overlap)*100])
    #ax[k].quiver(u_rms[:,:,0],u_rms[:,:,1])
    #ax1[k].set_title(k)
    
    cleanup_axis(ax1[k])
    
ax1[keys[2]].set_xlabel('f = 5 Hz')
ax1[keys[3]].set_xlabel('f = 10 Hz')

ax1[keys[0]].set_ylabel('center third')
ax1[keys[2]].set_ylabel('lower third')

for i in [0,1]:
    ax1[keys[i]].set_xlabel('t * f'); ax1[keys[i]].xaxis.set_label_position('top'); ax1[keys[i]].set_xticks(np.arange(0,11,2)); ax1[keys[i]].xaxis.tick_top(); ax1[keys[i]].spines['top'].set_visible(True)
    
for i in [1,3]:
    ax1[keys[i]].set_ylabel('spanwise position [cm]'); ax1[keys[i]].yaxis.set_label_position('right'); ax1[keys[i]].set_yticks(np.arange(0,16,3));  ax1[keys[i]].yaxis.tick_right(); ax1[keys[i]].spines['right'].set_visible(True)

plt.tight_layout()
plt.show()
plt.pause(0.5)

fig.savefig(fig_folder+'span_velocity_horizontal_highRes.pdf')


'''
Show the slices over the mean flow
'''
fig=plt.figure(figsize=(10,6))
ax=fig.add_subplot(111)
ax.imshow(mean_flow_magnitude,vmin=vmin,vmax=vmax)

ax.axhspan(span_slice[0],span_slice[1],alpha=0.5,color='r')
ax.axvspan(column_slice[0],column_slice[1],alpha=0.5,color='orange')

cleanup_axis(ax)
plt.tight_layout()
fig.savefig(fig_folder+'slice_locations_highRes.pdf')


'''
RMS velocity
'''

fig=plt.figure()
ax=fig.add_subplot(111)
vmin=0
vmax=0.2
fig = plt.figure(figsize=(16,9))
ax1 = {k:fig.add_subplot(2,2,i+1) for i,k in enumerate(keys)}
for k in keys:
    
    # mean velocity at each point
    mean_flow = np.nanmean(filtered_flowfield[k],axis=0)
    
    # fluctuating components of each velocity
    fluc_inst = filtered_flowfield[k] - 1*mean_flow
    
    # square root of the mean of the square of the fluctuations
    up_rms = np.sqrt( np.nanmean(fluc_inst**2,axis=0))
    
    up_rms_abs = np.sqrt( up_rms[:,:,0]**2 + up_rms[:,:,1]**2)
    
    ax1[k].imshow(up_rms_abs,vmin=vmin,vmax=vmax)
    #ax[k].quiver(u_rms[:,:,0],u_rms[:,:,1])
    ax1[k].set_title(k)
    
    # along the vertical axis
    ax.plot(np.nanmean(up_rms_abs[:,42:60],axis=1),label=k)
    
ax1[keys[2]].set_xlabel('f = 5 Hz')
ax1[keys[3]].set_xlabel('f = 10 Hz')

ax1[keys[0]].set_ylabel('center third')
ax1[keys[2]].set_ylabel('lower third')

plt.tight_layout()
plt.show()
plt.pause(0.5)

fig.savefig(fig_folder+'rms_velocity_highRes.pdf')

ax.legend()

'''
Correlation
'''

forcing_dict = {'center_05':(5,0.015),
                'lower_05':(5,0.015),
                'center_10':(10,0.015),
                'lower_10':(10,0.015)}

for k in keys:
    x_vec = np.arange(0,np.shape(mean_flow)[0]) *p[k].dx *(p[k].window_size-p[k].overlap)
    
    C_1 = 0.25
    f = float(forcing_dict[k][0])
    S = forcing_dict[k][1] * 2
    M = 0.03
    uprime_corr = C_1 * f * S * np.sqrt(S * M) / x_vec
    ax.plot(uprime_corr)
    
    
'''
Histogram of velocity at a point
'''

f=plt.figure();fake_ax=f.add_subplot(111)

fig = plt.figure(figsize=(8,6))
ax = {k:fig.add_subplot(2,2,i+1,sharex=fake_ax,sharey=fake_ax) for i,k in enumerate(keys)}
for k in keys:
    
    point_data_u = filtered_flowfield[k][:,span_slice[0]:span_slice[1],column_slice[0]:column_slice[1],0].flatten()
    #point_data_u = point_data_u[~np.isnan(point_data_u)]    
    as_series_u = pd.Series(point_data_u)
    
    point_data_v = filtered_flowfield[k][:,span_slice[0]:span_slice[1],column_slice[0]:column_slice[1],1].flatten()
    #point_data_v = point_data_v[~np.isnan(point_data_v)]    
    as_series_v = pd.Series(point_data_v)
    
    #as_series_u.plot.kde(ax=ax[k],color='b',label='horizontal component')
    #as_series_v.plot.kde(ax=ax[k],color='orange',label='vertical component')
    #ax[k].hist(point_data,bins=500)
    
    ax[k].scatter(point_data_u,point_data_v,alpha=0.01)
    
    ax[k].set_xlim([-0.25,0.25])
    ax[k].set_ylim([-0.25,0.25])
    
ax[keys[0]].legend()

ax[keys[2]].set_xlabel('Velocity [m/s]')
ax[keys[3]].set_xlabel('Velocity [m/s]')

ax[keys[0]].set_ylabel('Density')
ax[keys[2]].set_ylabel('Density')

[ax[k].set_title(k) for k in keys]
fig.savefig(fig_folder+'velocity_kde.pdf')