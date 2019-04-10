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

parent_folder = r'D:\high_speed_data\171204\\'
case_name = r'piv_4pumps_allAlwaysOn_fps4000_sideView'

need2rotate = False

vmin = -1.
vmax = 1.

'''
Load the data and make the scalers
'''

p = pickle.load(open(parent_folder+case_name+'.pkl'))
p.parent_folder = parent_folder
p.associate_flowfield()

if need2rotate:
    p.data.ff=piv.rotate_data_90(p.data.ff)
ff = p.data.ff

g_orig = fluids2d.geometry.GeometryScaler(dx=p.dx,im_shape=(800,1280),origin_pos=(-400,-600),origin_units='pix')
g = fluids2d.geometry.create_piv_scaler(p,g_orig)

row_lims,col_lims = g.get_coords(np.array([[0.005,-0.005],[-0.005,0.005]]))

time = np.arange(0,np.shape(ff)[0]) * p.dt_frames

'''
Filter the velocity field
'''
ff=piv.clip_flowfield(ff,5)

'''
quantiles of velocity at each vertical point along the column
'''
piv.vertical_percentile_distributions(ff,g,time,row_lims,col_lims,other_percentiles=[10,25,75,90])

'''
Reduce horizontal and vertical dimensions to get composite images
'''
piv.composite_image_plots(ff,g,time,row_lims,col_lims,vmin=vmin,vmax=vmax)
'''
Mean flow and fluctuations
'''

mean_flow=np.nanmean(ff,axis=0)
fluc = ff-mean_flow
u_rms = np.sqrt( np.nanmean( (fluc[:,:,:,0])**2,axis=0) + np.nanmean( (fluc[:,:,:,1])**2,axis=0) )
inst_speed = np.linalg.norm(ff,ord=2,axis=3)

fig,ax=plt.subplots(1,3,figsize=(9,6))
piv.add_fieldimg_to_ax(u_rms / np.nanmean(inst_speed,axis=0),g,ax[0],time=time,slice_dir=None,vel_dir=None,vmin=0,vmax=2)
piv.add_fieldimg_to_ax(np.sqrt(np.nanmean(ff[:,:,:,0],axis=0)**2+np.nanmean(ff[:,:,:,1],axis=0)**2),g,ax[1],time=time,slice_dir=None,vel_dir=None,vmin=0,vmax=2)
#ax[1].set_title('saturates at 2 m/s')
piv.add_fieldimg_to_ax(u_rms,g,ax[2],time=time,slice_dir=None,vel_dir=None,vmin=0,vmax=2)

#ax[0].set_title(r'''$ \sqrt{ \overline{U}^2 + \overline{V}^2 } $''')
ax[0].set_title('Turbulence intensity')
#ax[1].set_title(r'''$ \overline{ \sqrt{ {U}^2 + {V}^2 } } $''')
ax[1].set_title(r'''$ \sqrt{ \overline{ U}^2 + \overline{V}^2 } $''')
ax[2].set_title(r'''$ \sqrt{ \overline{u'^2} + \overline{v'^2} } $''')


cmap_horz = 'PuOr'
cmap_vert = 'seismic'

fig=plt.figure()
ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122)
#ax1.imshow(mean_flow[:,:,0],vmin=vmin,vmax=vmax,cmap=cmap_horz,extent=g.im_extent)
piv.add_fieldimg_to_ax(mean_flow[:,:,0],g,ax1,vel_dir='horizontal',vmin=vmin,vmax=vmax)
piv.add_fieldimg_to_ax(mean_flow[:,:,1],g,ax2,vel_dir='vertical',vmin=vmin,vmax=vmax)
#ax2.imshow(mean_flow[:,:,1],vmin=vmin,vmax=vmax,cmap=cmap_vert,extent=g.im_extent)

# Grid locs at each frame
#m = p.maskers[0]
#frame_as = p.a_frames % m.period_frames
lims = list()
#for mask_name in list(m.boundary_locs.keys()):
#    lims.append(m.boundary_locs[mask_name].loc[frame_as].mean(axis=1))

y1,x1 = g.get_coords(np.array([-0.06,0.07]))
piv.plot_both_components(ff[:,y1,x1,:],time=time)

ff_grad = piv.compute_gradients(ff)

piv.plot_both_components(ff[:,row_lims[0],col_lims[0],:],time=time)

piv.plot_both_components(np.nanmean(np.nanmean(ff[:,row_lims[0]:row_lims[1],col_lims[0]:col_lims[1],:],axis=1),axis=1),time=time)


stophere

'''
Other plots...
'''

fig=plt.figure()
ax1=fig.add_subplot(131)
ax2=fig.add_subplot(132,sharey=ax1)
#ax3=fig.add_axes([.3,.3,.1,.1])
ax3=fig.add_subplot(133,sharey=ax1)
cols_to_use = np.arange(12,26)

for i in np.arange(0,819,10):
    ax1.clear()
    ax2.clear()
    
    ax1.axvline(0,color='gray',alpha=0.3)
    ax2.axvline(0,color='gray',alpha=0.3)
    
    # Velocity components
    
    ax1.plot(np.nanmean(np.nanmean(ff[:,:,cols_to_use,0],axis=0),axis=1),g.y,color='red',alpha=0.5)
    ax1.plot(np.nanmean(np.nanmean(ff[i:i+10,:,cols_to_use,0],axis=0),axis=1),g.y,color='red',label='horizontal velocity')
    
    ax1.plot(np.nanmean(np.nanmean(ff[:,:,cols_to_use,1],axis=0),axis=1),g.y,color='blue',alpha=0.5)
    ax1.plot(np.nanmean(np.nanmean(ff[i:i+10,:,cols_to_use,1],axis=0),axis=1),g.y,color='blue',label='vertical velocity')
    
    ax1.set_xlim(-0.2,0.2)
    ax1.set_ylim([g.y[0],g.y[-1]])
    ax1.legend(loc=2)
    
    # TKE
    speed_mean = np.linalg.norm(np.nanmean(ff[:,:,:,:],axis=0),ord=2,axis=2)
    speed_now = np.linalg.norm(np.nanmean(ff[i:i+10,:,:,:],axis=0),ord=2,axis=2)
    ax2.plot(np.nanmean(speed_mean[:,cols_to_use],axis=1),g.y,color='k',alpha=0.5)
    ax2.plot(np.nanmean(speed_now[:,cols_to_use],axis=1),g.y,color='k')
    ax2.set_xlim(0,0.2)
    ax2.set_ylim([g.y[0],g.y[-1]])
    
    plt.show()
    plt.pause(0.05)
    
    
'''
Velocity component animation
'''    
    
fig,ax=plt.subplots(2,2,figsize=(12,8))
ax=ax.flatten()
plt.tight_layout()
    
vmin=-1
vmax = 1
for i in np.arange(0,1600,1):

    [a.clear() for a in ax]
    
    this_flow = np.nanmean(ff[i:i+1,:,:,:],axis=0)
    this_fluc = this_flow - mean_flow
    
    ax[0].imshow(this_flow[:,:,0],vmin=vmin,vmax=vmax,cmap=cmap_horz,extent=g.im_extent,origin='lower')
    ax[1].imshow(this_flow[:,:,1],vmin=vmin,vmax=vmax,cmap=cmap_vert,extent=g.im_extent,origin='lower')
    
    ax[2].imshow(this_fluc[:,:,0],vmin=vmin,vmax=vmax,cmap=cmap_horz,extent=g.im_extent,origin='lower')
    ax[3].imshow(this_fluc[:,:,1],vmin=vmin,vmax=vmax,cmap=cmap_vert,extent=g.im_extent,origin='lower')
    
    for loc in lims:
        [a.axhline(g_orig(loc.iloc[i]),color='cyan') for a in ax]
        
    for a in ax:
        g.set_axes_limits(a)
    
    #plt.show()
    #plt.pause(0.2)
    fig.savefig(parent_folder+case_name+r'\\frame_'+str(i)+'.png')
    
piv.plot_both_components(ff[:,40,30,:],time=time)