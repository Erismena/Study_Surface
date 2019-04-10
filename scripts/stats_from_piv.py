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

parent_folder = r'\\Mae-deike-lab3\c\Users\Luc Deike\data_comp3_C\180323\\'
cine_name = r'piv_sunbathing_topDown_middlePlane_on100_off400'
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

if need2rotate:
    p.data.ff=piv.rotate_data_90(p.data.ff)
    p.data.ff=piv.rotate_data_90(p.data.ff)
    p.data.ff=piv.rotate_data_90(p.data.ff)
ff = p.data.ff

g_orig = fluids2d.geometry.GeometryScaler(dx=p.dx,im_shape=(1,1),origin_pos=(-400,-640),origin_units='pix')
#g_orig = fluids2d.geometry.GeometryScaler(dx=p.dx,im_shape=(512,512),origin_pos=(-400,-50),origin_units='pix')
g = fluids2d.geometry.create_piv_scaler(p,g_orig)

row_lims,col_lims = g.get_coords(np.array([[0.005,-0.005],[-.005,0.005]]))

time = np.arange(0,np.shape(ff)[0]) * p.dt_frames

'''
Filter the velocity field
'''
ff=piv.clip_flowfield(ff,.5)

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

mean_speed = np.sqrt(mean_flow[:,:,0]**2+mean_flow[:,:,1]**2)

fig,ax=plt.subplots(1,3,figsize=(9,6))
piv.add_fieldimg_to_ax(np.log10(u_rms / mean_speed),g,ax[0],time=time,slice_dir=None,vel_dir=None,vmin=-1,vmax=1,cmap_other='coolwarm')
piv.add_fieldimg_to_ax(mean_speed,g,ax[1],time=time,slice_dir=None,vel_dir=None,vmin=0,vmax=0.2)
piv.add_fieldimg_to_ax(u_rms,g,ax[2],time=time,slice_dir=None,vel_dir=None,vmin=0,vmax=0.2)



#ax[0].set_title(r'''$ \sqrt{ \overline{U}^2 + \overline{V}^2 } $''')
ax[0].set_title('Turbulence intensity')
ax[1].set_title(r'''$  \sqrt{ \overline{U}^2 + \overline{V}^2 } } $''')
ax[2].set_title(r'''$ \sqrt{ \overline{u'^2} + \overline{v'^2} } $''')

fig = plt.figure()
ax = fig.add_subplot(111)
isotropy_rms = np.sqrt( np.nanmean( (fluc[:,:,:,0])**2,axis=0)) / np.sqrt( np.nanmean( (fluc[:,:,:,1])**2,axis=0))
piv.add_fieldimg_to_ax(np.log2(isotropy_rms),g,ax,time=time,slice_dir=None,vel_dir=None,vmin=-2,vmax=2,cmap_other='PRGn')

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
    

ff_grad = piv.compute_gradients(ff)

piv.plot_both_components(ff[:,row_lims[0],col_lims[0],:],time=time)
piv.plot_both_components(np.nanmean(np.nanmean(ff[:,row_lims[0]:row_lims[1],col_lims[0]:col_lims[1],:],axis=1),axis=1),time=time)


stophere
'''
Integral length scales
'''
center_size = 1 # must be odd

center_row = (np.shape(ff)[1]-1)/2
center_col = (np.shape(ff)[2]-1)/2

search_x = np.shape(ff)[2] - center_size
search_y = np.shape(ff)[1] - center_size

res,g_r = spectra.calculate_spatial_correlations(g,ff,center_row,center_col,center_size,search_x,search_y)
temporal_and_spatial_average = np.nanmean(res,axis=(0,1,2))
polar,r,line_avg,integral = spectra.make_radial_correlations(temporal_and_spatial_average,g_r,dr=0.5)
#radial_vals.append((temporal_and_spatial_average,g_r,polar,r,line_avg,integral))
#axs_Lint0[ai].plot(r,integral[:,0],color='blue')
#axs_Lint0[ai].plot(r,integral[:,1],color='red')
plt.figure()
plt.plot(r,integral[:,0],color='blue')
plt.plot(r,integral[:,1],color='red')
plt.show()


stophere

A = spectra.AutocorrResults(g)
A.run_autocorr(ff,time,4000,[0,21,0,63])
#A.run_autocorr(ff,time,2000,[0,89,80,90])


plt.figure()
plt.plot(A.lags,A.C_avg)

plt.figure()
plt.plot(A.lags,np.cumsum(A.C_avg,axis=0)*(A.lags[1]-A.lags[0]))


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
    
    # Grid position
    ax3.clear()
    for loc in lims:
        ax1.axhline(g_orig(loc.iloc[i]),color='k')
        ax2.axhline(g_orig(loc.iloc[i]),color='k')
        ax3.plot(time,g_orig(loc),color='k')
        ax3.plot(time[i],g_orig(loc.iloc[i]),'o',color='r')
    
    plt.show()
    plt.pause(0.05)
    
    
    
'''
Velocity component animation
'''    
    
fig,ax=plt.subplots(2,2,figsize=(6,9))
ax=ax.flatten()
plt.tight_layout()
    
vmin= -.5
vmax = .5
for i in np.arange(0,250,1):

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
    
    plt.show()
    plt.pause(0.2)
    
piv.plot_both_components(ff[:,40,30,:],time=time)

    
'''
Velocity component animation
'''    
    
fig=plt.figure(figsize=(13,8))
ax=fig.add_subplot(111)
fig.tight_layout()

figfolder = r'C:\Users\Luc Deike\data_comp3_C\180228\frames\\'
    
vmin= -.2
vmax = .2
sep=2
for i in np.arange(0,5740,sep*2):

    ax.clear()
    
    this_flow = np.nanmean(ff[i:i+sep,:,:,:],axis=0)
    
    ax.imshow(np.sqrt(this_flow[:,:,0]**2+this_flow[:,:,1]**2),vmin=0,vmax=0.4,cmap=None,extent=g.im_extent,origin='upper')
    ax.set_title('t = '+str(int(i*p.dt_frames*1000))+' ms')
    
    #ax.quiver(g.x,g.y,this_flow[:,:,0],this_flow[:,:,1])
    

    g.set_axes_limits(ax)
    
    if i==0:
        fig.tight_layout()
    
    fig.savefig(figfolder+'frame_'+str(i)+'.png')