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

figfolder = r'C:\Users\Luc Deike\Documents\dan_turbulence_project\figures\\'


parent_folder = r'C:\Users\Luc Deike\highspeed_data\171221\\'
case_name = r'piv_4pumps_topDown_sched-halfHalf_T1000ms_fullView_fps4000_dur4s'

need2rotate = False

vmin = -1
vmax = 1

'''
Load the data and make the scalers
'''

p = pickle.load(open(parent_folder+case_name+'.pkl'))
p.parent_folder = parent_folder
p.associate_flowfield()

if need2rotate:
    p.data.ff=piv.rotate_data_90(p.data.ff)
ff = p.data.ff

g_orig = fluids2d.geometry.GeometryScaler(dx=p.dx,im_shape=(800,1280),origin_pos=(-370,-360),origin_units='pix')
g = fluids2d.geometry.create_piv_scaler(p,g_orig)

row_lims,col_lims = g.get_coords(np.array([[0.005,-0.005],[-0.005,0.005]]))

time = np.arange(0,np.shape(ff)[0]) * p.dt_frames

point_locs = [[-0.070,0.064],
              [-0.063,0.057],
              [-0.056,0.051],
              [-0.046,0.041],
              [-0.037,0.031],
              [-0.027,0.021]]

'''
Filter the velocity field
'''
ff=piv.clip_flowfield(ff,3)

'''
phase-average
'''

num_periods = 4
period_dur = 2000

pa = np.zeros((num_periods,)+np.shape(ff[0:period_dur,:,:,:]))
for pi in range(num_periods):
    start_idx = period_dur*pi
    end_idx = period_dur*(pi+1)
    pa[pi,:,:,:,:] = ff[start_idx:end_idx,:,:,:]
pa = np.nanmean(pa,axis=0)
time_pa = time[0:period_dur]
speed = np.sqrt(pa[:,:,:,0]**2+pa[:,:,:,1]**2)

    

mean_flow=np.nanmean(ff,axis=0)
fluc = ff-mean_flow
u_rms = np.sqrt( np.nanmean( (fluc[:,:,:,0])**2,axis=0) + np.nanmean( (fluc[:,:,:,1])**2,axis=0) )
inst_speed = np.linalg.norm(ff,ord=2,axis=3)


cmap_horz = 'PuOr'
cmap_vert = 'seismic'

fig=plt.figure()
ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122)
piv.add_fieldimg_to_ax(mean_flow[:,:,0],g,ax1,vel_dir='horizontal',vmin=vmin,vmax=vmax)
piv.add_fieldimg_to_ax(mean_flow[:,:,1],g,ax2,vel_dir='vertical',vmin=vmin,vmax=vmax)

'''
Along jet axis
'''

fig = plt.figure(figsize=(9,5))
ax = fig.add_subplot(111)

#point_locs = [[-0.070,0.064],
#              [-0.063,0.057],
#              [-0.056,0.051],
#              [-0.046,0.041],
#              [-0.037,0.031],
#              [-0.027,0.021]]

point_locs = [[-0.070,0.064],
              [-0.046,0.041],
              [-0.027,0.021]]

c = ['orange','cyan','red']


    
ax_inset=fig.add_axes([0.72,0.65,0.25,0.25])
ax_inset.imshow(np.nanmean(inst_speed,axis=0),vmin=0,vmax=1.5)
ax_inset.get_xaxis().set_visible(False)
ax_inset.get_yaxis().set_visible(False)

for pi,point_loc in enumerate(point_locs):
    row,col = g.get_coords(np.array(point_loc))
    print(row,col)
        
    ax.plot(time_pa,speed[:,row,col],alpha=0.8,color=c[pi])
    
    ax_inset.plot(col,row,'x',color=c[pi])
    
ax.set_ylabel('$|\mathbf{U}|$ [m/s]')
ax.set_xlabel('$t$ [ms]')
fig.tight_layout()
fig.savefig(figfolder+'jet_axis_timeplots.pdf')

    
'''
Along jet crosssection
'''

'''

fig = plt.figure()
ax = fig.add_subplot(111)

point_locs = [[-0.069,0.041],
              [-0.065,0.045],
              [-0.060,0.052],]

for point_loc in point_locs:
    row,col = g.get_coords(np.array(point_loc))
    print(row,col)
    
    [a.plot(point_loc[1],point_loc[0],'x',color='white') for a in [ax1,ax2]]
    
    ax.plot(time_pa,speed[:,row,col],alpha=0.8)
'''
    
    
'''
snapshots along axis
'''
origin = [-0.078,0.073]
origin_pt = g.get_coords(np.array(origin))
line_x = np.arange(origin_pt[1],origin_pt[1]-min(origin_pt),-1)
line_y = np.arange(origin_pt[0],origin_pt[0]-min(origin_pt),-1)
line_dx = np.sqrt(2.*g.dx**2)
line_dist = np.arange(0,len(line_x))*line_dx

la = np.zeros((period_dur,len(line_x),2))
for ti in range(period_dur):
    for li in range(len(line_x)):
        la[ti,li,:] = pa[ti,line_y[li],line_x[li],:]
la_speed = np.sqrt(la[:,:,0]**2+la[:,:,1]**2)
        
fig = plt.figure(figsize=(9,5))
ax = fig.add_subplot(111)
cax=ax.imshow(la_speed.T,aspect='auto',extent=[0,max(time_pa),0,max(line_dist)],vmin=0,vmax=1.5)
ax.set_ylabel('distance along line [m]')
ax.set_xlabel('$t$ [s]')
ax.set_ylim([0.015,0.20])

cbar = fig.colorbar(cax)
cbar.set_label('$|\mathbf{U}|$ [m/s]')

ax=fig.add_axes([0.70,0.83,0.15,0.15])
ax.imshow(np.nanmean(inst_speed,axis=0),vmin=0,vmax=1.5)
ax.plot(line_x,line_y,color='r')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
    


fig.tight_layout()
fig.savefig(figfolder+'jet_centerline_speed.pdf')


#cbar.ax.set_yticklabels(['< -1', '0', '> 1'])  # vertically oriented colorbar

fig = plt.figure()
ax = fig.add_subplot(111)

import scipy.ndimage.filters

la_speed_filt = scipy.ndimage.filters.uniform_filter(la_speed,size=(10,0))

for ti in np.linspace(0,period_dur-1,20).astype(int):
    ax.plot(line_dist,la_speed_filt[ti,:],color=[.5,.5,ti/float(period_dur)])