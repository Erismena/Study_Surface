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

import fluids2d.piv as piv

parent_folder = r'C:\Users\danjr\Documents\Fluids Research\Data\misc\PIV\\'
fig_folder = r'C:\Users\danjr\Documents\Fluids Research\Writing\grid3_PIV_analysis\figures\\'

keys=('05_05','05_08','05_10','10_05','10_08','10_10','15_05','15_08','15_10')
res_to_load = {'05_05':r'PIV_sv_vp_makro100mmzeiss_X0mm_Y15mm_fps10k_A05mm_f05Hz_grid3x4_5cycles_Cam_20861_Cine3',
               '05_08':r'PIV_sv_vp_makro100mmzeiss_X0mm_Y15mm_fps10k_A05mm_f08Hz_grid3x4_5cycles_Cam_20861_Cine3',
               '05_10':r'PIV_sv_vp_makro100mmzeiss_X0mm_Y15mm_fps10k_A05mm_f10Hz_grid3x4_5cycles_Cam_20861_Cine3',
               '10_05':r'PIV_sv_vp_makro100mmzeiss_X0mm_Y15mm_fps10k_A10mm_f05Hz_grid3x4_5cycles_Cam_20861_Cine3',
               '10_08':r'PIV_sv_vp_makro100mmzeiss_X0mm_Y15mm_fps10k_A10mm_f08Hz_grid3x4_5cycles_Cam_20861_Cine3',
               '10_10':r'PIV_sv_vp_makro100mmzeiss_X0mm_Y15mm_fps10k_A10mm_f10Hz_grid3x4_5cycles_Cam_20861_Cine3',
               '15_05':r'PIV_sv_vp_makro100mmzeiss_X0mm_Y15mm_fps10k_A15mm_f05Hz_grid3x4_5cycles_Cam_20861_Cine3',
               '15_08':r'PIV_sv_vp_makro100mmzeiss_X0mm_Y15mm_fps10k_A15mm_f08Hz_grid3x4_5cycles_Cam_20861_Cine3',
               '15_10':r'PIV_sv_vp_makro100mmzeiss_X0mm_Y15mm_fps10k_A15mm_f10Hz_grid3x4_5cycles_Cam_20861_Cine3'}
#keys = list(res_to_load.keys())

# define the start of the 4 complete cycles
cycle_start = {'05_05':102,
               '05_08':102,
               '05_10':112,
               '10_05':130,
               '10_08':37,
               '10_10':26,
               '15_05':173,
               '15_08':46,
               '15_10':39}

cycle_lengths={'05_05':2000,
               '05_08':1250,
               '05_10':1000,
               '10_05':2000,
               '10_08':1250,
               '10_10':1000,
               '15_05':2000,
               '15_08':1250,
               '15_10':1000}

# colors map to amplitude
colors =      {'05_05':'b',
               '05_08':'b',
               '05_10':'b',
               '10_05':'g',
               '10_08':'g',
               '10_10':'g',
               '15_05':'r',
               '15_08':'r',
               '15_10':'r'}

# line styles map to frequency
linestyles =  {'05_05':'-',
               '05_08':'--',
               '05_10':'-.',
               '10_05':'-',
               '10_08':'--',
               '10_10':'-.',
               '15_05':'-',
               '15_08':'--',
               '15_10':'-.'}

a = [5,5,5,10,10,10,15,15,15]
amplitudes = {k:a[i] for i,k in enumerate(keys)}

f = [5,8,10,5,8,10,5,8,10]
frequencies = {k:f[i] for i,k in enumerate(keys)}

p = {key:pickle.load(open(parent_folder+res_to_load[key]+'.pkl')) for key in keys}

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
ax = {k:fig.add_subplot(3,3,i+1) for i,k in enumerate(keys)}
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



ax[keys[6]].set_xlabel('f = 5 Hz')
ax[keys[7]].set_xlabel('f = 8 Hz')
ax[keys[8]].set_xlabel('f = 10 Hz')

ax[keys[0]].set_ylabel('A = 5 mm')
ax[keys[3]].set_ylabel('A = 10 mm')
ax[keys[6]].set_ylabel('A = 15 Hz')

plt.tight_layout()
plt.show()
plt.pause(0.5)

fig.savefig(fig_folder+'mean_flows.pdf')

'''
Histogram of velocity at a point
'''

fig = plt.figure()

point_data = filtered_flowfield[k][:,18,40,1]
point_data = point_data[~np.isnan(point_data)]
plt.hist(point_data,bins=20)


'''
RMS velocity
'''

vmin=0
vmax=0.2
fig = plt.figure(figsize=(16,9))
ax1 = {k:fig.add_subplot(3,3,i+1) for i,k in enumerate(keys)}
for k in keys:
    
    # mean velocity at each point
    mean_flow = np.nanmean(filtered_flowfield[k],axis=0)
    
    mean_flow_magnitude = np.sqrt( mean_flow[:,:,0]**2 + mean_flow[:,:,1]**2 )
    
    # fluctuating components of each velocity
    fluc_inst = filtered_flowfield[k] - mean_flow
    
    # square root of the mean of the square of the fluctuations
    up_rms = np.sqrt( np.nanmean(fluc_inst**2,axis=0))
    
    up_rms_abs = np.sqrt( up_rms[:,:,0]**2 + up_rms[:,:,1]**2)
    
    ax1[k].imshow(up_rms_abs,vmin=vmin,vmax=vmax)
    #ax[k].quiver(u_rms[:,:,0],u_rms[:,:,1])
    ax1[k].set_title(k)
    
    cleanup_axis(ax1[k])
    
ax1[keys[6]].set_xlabel('f = 5 Hz')
ax1[keys[7]].set_xlabel('f = 8 Hz')
ax1[keys[8]].set_xlabel('f = 10 Hz')

ax1[keys[0]].set_ylabel('A = 5 mm')
ax1[keys[3]].set_ylabel('A = 10 mm')
ax1[keys[6]].set_ylabel('A = 15 mm')

plt.tight_layout()
plt.show()
plt.pause(0.5)

fig.savefig(fig_folder+'rms_velocity.pdf')


'''
Vertical velocity along span
'''

span_slice = [39,42]

vmin=-0.1
vmax=0.1
fig = plt.figure(figsize=(16,9))
ax1 = {k:fig.add_subplot(3,3,i+1) for i,k in enumerate(keys)}
for k in keys:
    
    ax1[k].imshow(np.nanmean(filtered_flowfield[k][:,:,span_slice[0]:span_slice[1],0],axis=2).transpose(),vmin=vmin,vmax=vmax,aspect='auto',cmap='seismic',extent=[0,5,0,(np.shape(filtered_flowfield[k])[1]+1)*p[k].dx*(p[k].window_size-p[k].overlap)*100])
    #ax[k].quiver(u_rms[:,:,0],u_rms[:,:,1])
    #ax1[k].set_title(k)
    
    cleanup_axis(ax1[k])
    
ax1[keys[6]].set_xlabel('f = 5 Hz')
ax1[keys[7]].set_xlabel('f = 8 Hz')
ax1[keys[8]].set_xlabel('f = 10 Hz')

ax1[keys[0]].set_ylabel('A = 5 mm')
ax1[keys[3]].set_ylabel('A = 10 mm')
ax1[keys[6]].set_ylabel('A = 15 mm')

for i in [0,1,2]:
    ax1[keys[i]].set_xlabel('t * f'); ax1[keys[i]].xaxis.set_label_position('top'); ax1[keys[i]].set_xticks(np.arange(0,6)); ax1[keys[i]].xaxis.tick_top(); ax1[keys[i]].spines['top'].set_visible(True)
    
for i in [2,5,8]:
    ax1[keys[i]].set_ylabel('spanwise position [cm]'); ax1[keys[i]].yaxis.set_label_position('right'); ax1[keys[i]].set_yticks(np.arange(0,15,5));  ax1[keys[i]].yaxis.tick_right(); ax1[keys[i]].spines['right'].set_visible(True)

plt.tight_layout()
plt.show()
plt.pause(0.5)

fig.savefig(fig_folder+'span_velocity_vertical.pdf')

'''
Horizontal velocity along span
'''

vmin=-0.1
vmax=0.1
fig = plt.figure(figsize=(16,9))
ax1 = {k:fig.add_subplot(3,3,i+1) for i,k in enumerate(keys)}
for k in keys:
    
    ax1[k].imshow(np.nanmean(filtered_flowfield[k][:,:,span_slice[0]:span_slice[1],1],axis=2).transpose(),vmin=vmin,vmax=vmax,aspect='auto',cmap='PuOr',extent=[0,5,0,(np.shape(filtered_flowfield[k])[1]+1)*p[k].dx*(p[k].window_size-p[k].overlap)*100])
    #ax[k].quiver(u_rms[:,:,0],u_rms[:,:,1])
    #ax1[k].set_title(k)
    
    cleanup_axis(ax1[k])
    
ax1[keys[6]].set_xlabel('f = 5 Hz')
ax1[keys[7]].set_xlabel('f = 8 Hz')
ax1[keys[8]].set_xlabel('f = 10 Hz')

ax1[keys[0]].set_ylabel('A = 5 mm')
ax1[keys[3]].set_ylabel('A = 10 mm')
ax1[keys[6]].set_ylabel('A = 15 mm')

for i in [0,1,2]:
    ax1[keys[i]].set_xlabel('t * f'); ax1[keys[i]].xaxis.set_label_position('top'); ax1[keys[i]].set_xticks(np.arange(0,6)); ax1[keys[i]].xaxis.tick_top(); ax1[keys[i]].spines['top'].set_visible(True)
    
for i in [2,5,8]:
    ax1[keys[i]].set_ylabel('spanwise position [cm]'); ax1[keys[i]].yaxis.set_label_position('right'); ax1[keys[i]].set_yticks(np.arange(0,15,5));  ax1[keys[i]].yaxis.tick_right(); ax1[keys[i]].spines['right'].set_visible(True)

plt.tight_layout()
plt.show()
plt.pause(0.5)

fig.savefig(fig_folder+'span_velocity_horizontal.pdf')

'''
Vertical velocity along central column
'''

column_slice = [18,21]

vmin=-0.1
vmax=0.1
fig = plt.figure(figsize=(16,9))
ax1 = {k:fig.add_subplot(3,3,i+1) for i,k in enumerate(keys)}
for k in keys:
    
    ax1[k].imshow(np.nanmean(filtered_flowfield[k][:,column_slice[0]:column_slice[1],:,0],axis=1).transpose(),origin='lower',vmin=vmin,vmax=vmax,aspect='auto',cmap='seismic',extent=[0,5,0,(np.shape(filtered_flowfield[k])[2]+1)*p[k].dx*(p[k].window_size-p[k].overlap)*100])
    #ax[k].quiver(u_rms[:,:,0],u_rms[:,:,1])
    #ax1[k].set_title(k)
    
    cleanup_axis(ax1[k])
    
ax1[keys[6]].set_xlabel('f = 5 Hz')
ax1[keys[7]].set_xlabel('f = 8 Hz')
ax1[keys[8]].set_xlabel('f = 10 Hz')

ax1[keys[0]].set_ylabel('A = 5 mm')
ax1[keys[3]].set_ylabel('A = 10 mm')
ax1[keys[6]].set_ylabel('A = 15 mm')

for i in [0,1,2]:
    ax1[keys[i]].set_xlabel('t * f'); ax1[keys[i]].xaxis.set_label_position('top'); ax1[keys[i]].set_xticks(np.arange(0,6)); ax1[keys[i]].xaxis.tick_top(); ax1[keys[i]].spines['top'].set_visible(True)
    
for i in [2,5,8]:
    ax1[keys[i]].set_ylabel('vertical position [cm]'); ax1[keys[i]].yaxis.set_label_position('right'); ax1[keys[i]].set_yticks([0,10,20,30]);  ax1[keys[i]].yaxis.tick_right(); ax1[keys[i]].spines['right'].set_visible(True)

plt.tight_layout()
plt.show()
plt.pause(0.5)

fig.savefig(fig_folder+'central_velocity_vertical.pdf')


'''
Horizontal velocity along central column
'''

vmin=-0.1
vmax=0.1
fig = plt.figure(figsize=(16,9))
ax1 = {k:fig.add_subplot(3,3,i+1) for i,k in enumerate(keys)}
for k in keys:
    
    ax1[k].imshow(np.nanmean(filtered_flowfield[k][:,column_slice[0]:column_slice[1],:,1],axis=1).transpose(),origin='lower',vmin=vmin,vmax=vmax,aspect='auto',cmap='PuOr',extent=[0,5,0,(np.shape(filtered_flowfield[k])[2]+1)*p[k].dx*(p[k].window_size-p[k].overlap)*100])
    #ax[k].quiver(u_rms[:,:,0],u_rms[:,:,1])
    #ax1[k].set_title(k)
    
    cleanup_axis(ax1[k])
    
ax1[keys[6]].set_xlabel('f = 5 Hz')
ax1[keys[7]].set_xlabel('f = 8 Hz')
ax1[keys[8]].set_xlabel('f = 10 Hz')

ax1[keys[0]].set_ylabel('A = 5 mm')
ax1[keys[3]].set_ylabel('A = 10 mm')
ax1[keys[6]].set_ylabel('A = 15 mm')

for i in [0,1,2]:
    ax1[keys[i]].set_xlabel('t * f'); ax1[keys[i]].xaxis.set_label_position('top'); ax1[keys[i]].set_xticks(np.arange(0,6)); ax1[keys[i]].xaxis.tick_top(); ax1[keys[i]].spines['top'].set_visible(True)
    
for i in [2,5,8]:
    ax1[keys[i]].set_ylabel('vertical position [cm]'); ax1[keys[i]].yaxis.set_label_position('right'); ax1[keys[i]].set_yticks([0,10,20,30]);  ax1[keys[i]].yaxis.tick_right(); ax1[keys[i]].spines['right'].set_visible(True)

plt.tight_layout()
plt.show()
plt.pause(0.5)

fig.savefig(fig_folder+'central_velocity_horizontal.pdf')


'''
Show the slices over the mean flow
'''
fig=plt.figure(figsize=(10,6))
ax=fig.add_subplot(111)
ax.imshow(mean_flow_magnitude,vmin=vmin,vmax=vmax)

ax.axvspan(span_slice[0],span_slice[1],alpha=0.5,color='r')
ax.axhspan(column_slice[0],column_slice[1],alpha=0.5,color='orange')

cleanup_axis(ax)
plt.tight_layout()
fig.savefig(fig_folder+'slice_locations.pdf')


'''
Speed at a point vs angle
'''
fig=plt.figure()
ax1=fig.add_subplot(211)
ax2=fig.add_subplot(212)

for k in keys:
    
    # x is time non-dimensionalized by the frequency
    xi = np.arange(cycle_start[k], cycle_start[k] + 4*cycle_lengths[k])  /10
    x = (xi-xi[0]) / (float(cycle_lengths[k])/10.)
    
    # y is the speed non-dimensionalized by the frequency and amplitude
    characteristic_speed = float(amplitudes[k]) * float(frequencies[k])    
    y_dim = np.nanmean(np.nanmean(filtered_flowfield[k][xi,17:19,38:42,0],axis=1),axis=1)
    y = y_dim / characteristic_speed
    
    ax1.plot(x,y,color=colors[k],ls=linestyles[k])
    
    ax2.plot(xi,y_dim,color=colors[k],ls=linestyles[k])
    
plt.tight_layout()
plt.show()
plt.pause(0.5)

'''
RMS velocity
'''

fig=plt.figure()
ax=fig.add_subplot(111)
vmin=0
vmax=0.2
fig = plt.figure(figsize=(16,9))
ax1 = {k:fig.add_subplot(3,3,i+1) for i,k in enumerate(keys)}
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

ax.legend()