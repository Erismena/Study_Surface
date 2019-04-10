# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:08:23 2017

@author: danjr
"""

import fluids2d.piv as piv
import fluids2d.spectra as spectra
import matplotlib.pyplot as plt
import matplotlib.patches
import pickle
import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.signal
from fluids2d.masking import MovingRectMasks
import fluids2d.geometry
import pims
from fluids2d.piv import PIVDataProcessing
from skimage import measure

figfolder = r'C:\Users\Luc Deike\Documents\dan_turbulence_project\figures\\'

meta = pd.DataFrame()

case_names = [r'piv_4pumps_topDown_sched-tenthOn_T0250ms_fullView_fps4000_dur4s',
              r'piv_4pumps_topDown_sched-tenthOn_T0500ms_fullView_fps4000_dur4s',
              r'piv_4pumps_topDown_sched-fifthOn_T0100ms_fullView_fps4000_dur4s',
              r'piv_4pumps_topDown_sched-fifthOn_T0250ms_fullView_fps4000_dur4s',
              r'piv_4pumps_topDown_sched-thirdOn_T0100ms_fullView_fps4000_dur4s',
              r'piv_4pumps_topDown_sched-thirdOn_T0250ms_fullView_fps4000_dur4s',
              r'piv_4pumps_topDown_sched-thirdOn_T0500ms_fullView_fps4000_dur4s',
              r'piv_4pumps_topDown_sched-halfHalf_T0100ms_fullView_fps4000_dur4s',
              r'piv_4pumps_topDown_sched-halfHalf_T250ms_fullView_fps4000_dur4s',
              r'piv_4pumps_topDown_sched-halfHalf_T0500ms_fullView_fps4000_dur4s',
              r'piv_4pumps_topDown_sched-halfHalf_T1000ms_fullView_fps4000_dur4s']
offsets = [(-345,-360)]*len(case_names)
parent_folders = [r'C:\Users\Luc Deike\highspeed_data\171221\\']*len(case_names)
diffusers = [False] * len(case_names)
periods = [250,500,100,250,100,250,500,100,250,500,1000]
on_portions = [0.1,0.1,0.2,0.2,1./3,1./3,1./3,0.5,0.5,0.5,0.5]

c_onportions = {0.1:'b',0.2:'g',1./3:'orange',0.5:'r'} #,1:'cyan'
ls_periods = {100:':',250:'-.',500:'--',1000:'-'}


meta['case_name'] = case_names
meta['offset'] = offsets
meta['parent_folder'] = parent_folders
meta['diffuser'] = diffusers
meta['period'] = periods
meta['on_portion'] = on_portions


need2rotate = False

vmin = -.5
vmax = .5

'''
Initialize the figures
'''

#fig_turb,axs_turb = plt.subplots(3,len(case_names),figsize=(11,8))

Figs = piv.PIVComparisonsFigures(3,4,figsize=(11,8.3),max_speed=0.5,vmin=-.5,vmax=0.5,legend_axi=7)

fig_fft = plt.figure()
ax_fft = fig_fft.add_subplot(111)

'''
Loop through each case and add the plots to the figures
'''


C_dict = {}
for i in meta.index:
    
    parent_folder = meta.loc[i,'parent_folder']
    case_name = meta.loc[i,'case_name']
    offset = meta.loc[i,'offset']
    
    ai = i
    if i>6:
        ai=ai+1
    
    color = c_onportions[meta.loc[i,'on_portion']]
    ls = ls_periods[meta.loc[i,'period']]
    
    label = 'T = '+str(meta.loc[i,'period'])+' ms\n$\phi$ = 0.'+str(int(100*meta.loc[i,'on_portion']))

    p = pickle.load(open(parent_folder+case_name+'.pkl'))
    p.parent_folder = parent_folder
    p.associate_flowfield()   
    
    if need2rotate:
        p.data.ff=piv.rotate_data_90(p.data.ff)
    ff = p.data.ff
    
    g_orig = fluids2d.geometry.GeometryScaler(dx=p.dx,im_shape=(1,1),origin_pos=offset,origin_units='pix')
    g = fluids2d.geometry.create_piv_scaler(p,g_orig)
    
    time = np.arange(0,np.shape(ff)[0]) * p.dt_frames
    
    lim = 0.02
    center_rows,center_cols = g.get_coords(np.array([[lim,-1*lim],[-1*lim,lim]]))
    
    '''
    Filter the velocity field
    '''
    ff=piv.clip_flowfield(ff,3)
    
    '''
    Mean flow and fluctuations
    '''
    
    mean_flow=np.nanmean(ff,axis=0)
    fluc = ff-mean_flow
    u_rms = np.sqrt( np.nanmean( (fluc[:,:,:,0])**2,axis=0) + np.nanmean( (fluc[:,:,:,1])**2,axis=0) )
    inst_speed = np.linalg.norm(ff,ord=2,axis=3)    
    meanflow_speed = np.sqrt((mean_flow[:,:,0])**2 + (mean_flow[:,:,1])**2)
    
    dudx = np.gradient(fluc[:,:,:,0],axis=1)
    dvdy = np.gradient(fluc[:,:,:,1],axis=2) 
    
    '''
    Add to the figures
    '''
    Figs.add_case(ai,ff,g,time)
    Figs.update_limits(ai,[-0.075,0.075],[-0.075,0.075])
    Figs.add_text(ai,.07,.07,label)
    Figs.add_rect(ai,-1*lim,-1*lim,lim*2,lim*2,color=color,ls=ls,)
    
    '''
    FFTs
    '''
    
    '''
    num_lags = 200
    #num_lags = 200
    
    #C_arr = np.zeros((num_lags,(center_rows[1]-center_rows[0]+1),(center_cols[1]-center_cols[0]+1),2))
    C_avg = np.zeros((num_lags,2))
    
    keep_first=num_lags/2
    favg= np.zeros((keep_first,2))
    
    for d in [0,1]:
        fluc[:,:,:,d] = piv.fill_nans_3d(fluc[:,:,:,d])
    C_arr,lags = spectra.autocorr(fluc[:,center_rows[0]:center_rows[1]+1,center_cols[0]:center_cols[1]+1,:],num_lags=num_lags,dt=p.dt_frames)
    for d in [0,1]:    
        C_avg[:,d] = np.nanmean(np.nanmean(C_arr[:,:,:,d],axis=1),axis=1)        
        favg[:,d],freq = spectra.temporal_spectrum_from_autocorr(C_avg[:,d],lags)
    
    C_dict[case_name] = C_arr
    '''
    
    '''
    Some scalar data
    '''
    meta.loc[i,'u_rms'] = np.nanmean(u_rms[center_rows[0]:center_rows[1],center_cols[0]:center_cols[1]])
    meta.loc[i,'mean_speed'] = np.nanmean(meanflow_speed[center_rows[0]:center_rows[1],center_cols[0]:center_cols[1]])
    meta.loc[i,'epsilon_0'] = 1e-6 * 15 * np.nanmean((dudx/g.dx)**2)
    meta.loc[i,'epsilon_1'] = 1e-6 * 15 * np.nanmean((dvdy/g.dx)**2)
    
stophere

fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(111)

ax.plot([np.nan,np.nan],[np.nan,np.nan],'o-',color='k',label='$\epsilon_0$')
ax.plot([np.nan,np.nan],[np.nan,np.nan],'x--',color='k',label='$\epsilon_1$')
for key in np.sort(list(c_onportions.keys())):
    color = c_onportions[key]
    ax.plot([np.nan,np.nan],[np.nan,np.nan],'-',color=color,label='$\phi$ = 0.'+str(int(100.*key)))
ax.legend()

for o in meta['on_portion'].unique():
    meta_on = meta[meta['on_portion']==o]
    ax.plot(meta_on['period'],meta_on['epsilon_0'],c=c_onportions[o],ls='-')
    ax.plot(meta_on['period'],meta_on['epsilon_1'],c=c_onportions[o],ls='--')
    
ax.scatter(meta['period'],meta['epsilon_0'],c=[c_onportions[o] for o in meta['on_portion']],marker='o',s=20)
ax.scatter(meta['period'],meta['epsilon_1'],c=[c_onportions[o] for o in meta['on_portion']],marker='x',s=20)

ax.set_ylabel('$\epsilon$ [W/kg]')
ax.set_xlabel('T [ms]')
fig.savefig(figfolder+'epsilon_calc.pdf')

#np.save(figfolder+'C_dict_180103c_2cmsq_shorter.np',C_dict)

Figs.tight_layout()
Figs.add_legends()

fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)

'''
Legend
'''
ax.plot([np.nan,np.nan],[np.nan,np.nan],'o-',color='k',label='$u_\mathrm{rms}$')
ax.plot([np.nan,np.nan],[np.nan,np.nan],'x--',color='k',label='$|\overline{\mathbf{U}}|$')
for key in np.sort(list(c_onportions.keys())):
    color = c_onportions[key]
    ax.plot([np.nan,np.nan],[np.nan,np.nan],'-',color=color,label='$\phi$ = 0.'+str(int(100.*key)))
ax.legend()


for o in meta['on_portion'].unique():
    meta_on = meta[meta['on_portion']==o]
    ax.plot(meta_on['period'],meta_on['u_rms'],c=c_onportions[o],ls='-')
    ax.plot(meta_on['period'],meta_on['mean_speed'],c=c_onportions[o],ls='--')
    
meta_diff = meta[meta['diffuser']==True]
ax.scatter(meta_diff['period'],meta_diff['u_rms'],c='k',marker='s',s=40)
ax.scatter(meta_diff['period'],meta_diff['mean_speed'],c='k',marker='s',s=40)

ax.scatter(meta['period'],meta['u_rms'],c=[c_onportions[o] for o in meta['on_portion']],marker='o',s=20)
ax.scatter(meta['period'],meta['mean_speed'],c=[c_onportions[o] for o in meta['on_portion']],marker='x',s=20)

ax.set_xlabel('T [ms]')
ax.set_ylabel('speed [m/s]')
ax.set_ylim([0,0.4])

fig.tight_layout()
stophere
fig.savefig(figfolder+'baseline_turb_avgs.pdf')


'''
Plot the FFTs
'''
    
fig_ffts,ax_ffts = plt.subplots(4,2,figsize=(10,7),sharex=True,sharey=True)
#ax_ffts = ax_ffts.flatten()
ax_dict = {np.flipud(np.sort(list(ls_periods.keys())))[i]:ax_ffts[i] for i in range(len(ls_periods))}


for i in meta.index:
    
    parent_folder = meta.loc[i,'parent_folder']
    case_name = meta.loc[i,'case_name']
    offset = meta.loc[i,'offset']
    
    color = c_onportions[meta.loc[i,'on_portion']]
    ls = ls_periods[meta.loc[i,'period']]
    
    label = 'T = '+str(meta.loc[i,'period'])+' ms, A = '+str(int(100*meta.loc[i,'on_portion']))+' pct'
    
    C_arr = C_dict[case_name]
    
    C_avg = np.nanmean(np.nanmean(C_arr[:,:,:,:],axis=1),axis=1)
    for d in [0,1]:
        favg[:,d],freq = piv.temporal_spectrum_from_autocorr(C_avg[:,d],lags)
        
    ax = ax_dict[meta.loc[i,'period']]
    
    [ax[j].axvline(1000./int(meta.loc[i,'period']),color='gray',alpha=0.5) for j in [0,1]]
    
    ax[0].loglog(freq,favg[:,0],color=color,alpha=0.8)
    ax[1].loglog(freq,favg[:,1],color=color,alpha=0.8)

ax_ffts[0,0].set_title('$B_{1,1}$')
ax_ffts[0,1].set_title('$B_{2,2}$')

ax_ffts[3,0].set_xlabel('$f$ [Hz]')
ax_ffts[3,1].set_xlabel('$f$ [Hz]')

[ax_ffts[i,0].set_ylabel('$T$ = '+str(int(np.flipud(np.sort(list(ls_periods.keys())))[i]))+' ms') for i in range(len(ls_periods))]

[a.plot([10**1,10**1.5],[10**2,10**(2-0.5*5./3)],color='gray',ls='--',alpha=0.5) for a in ax_ffts.flatten()]
#[a.set_ylim([10**-1,10**3]) for a in ax_ffts.flatten()]


fig_ffts.tight_layout()

'''
Animation
'''

animation_folder = r'C:\Users\Luc Deike\highspeed_data\171221\animation_frames\\'

g_dict = {}
ff_dict = {}
for i in meta.index:
    
    parent_folder = meta.loc[i,'parent_folder']
    case_name = meta.loc[i,'case_name']
    offset = meta.loc[i,'offset']
    
    ai = i
    if i>6:
        ai=ai+1
    
    #color = c_periods[meta.loc[i,'period']]
    color = c_onportions[meta.loc[i,'on_portion']]
    #ls = ls_onportions[meta.loc[i,'on_portion']]
    ls = ls_periods[meta.loc[i,'period']]
    
    

    p = pickle.load(open(parent_folder+case_name+'.pkl'))
    p.parent_folder = parent_folder
    p.associate_flowfield()   
    
    if need2rotate:
        p.data.ff=piv.rotate_data_90(p.data.ff)
    ff = p.data.ff
    
    g_orig = fluids2d.geometry.GeometryScaler(dx=p.dx,im_shape=(1,1),origin_pos=offset,origin_units='pix')
    g = fluids2d.geometry.create_piv_scaler(p,g_orig)
    g_dict[i] = g
    
    time = np.arange(0,np.shape(ff)[0]) * p.dt_frames
    
    '''
    Filter the velocity field
    '''
    ff=piv.clip_flowfield(ff,3)
    
    ff_dict[i] = ff
    
sep = 5
for fi,f in enumerate(range(0,8000,sep)):
    
    fig,axs = plt.subplots(3,4,figsize=(11,8.3)); axs = axs.flatten()
    
    for i in meta.index:
        
        label = 'T = '+str(meta.loc[i,'period'])+' ms\n$\phi$ = 0.'+str(int(100*meta.loc[i,'on_portion']))
        
        ai = i
        if i>6:
            ai=ai+1
        
        g = g_dict[i]
        ff = ff_dict[i]
        
        im = np.nanmean(np.sqrt( ff[f:f+sep,:,:,0]**2 + ff[f:f+sep,:,:,1]**2),axis=0)
        
        piv.add_fieldimg_to_ax(im,g,axs[ai],time=time,slice_dir=None,vel_dir=None,vmin=0,vmax=1.5)
        
        [a[ai].set_xlim([-0.075,0.075]) for a in [axs]]
        [a[ai].set_ylim([-0.075,0.075]) for a in [axs]]
        [a[ai].text(0.07,0.07,label,ha='right',va='top',color='white',fontsize=12) for a in [axs]]
        
        if ai not in [0,4,8]:
            [a[ai].yaxis.set_ticklabels([]) for a in [axs]]
        else:
            [a[ai].set_ylabel('y [m]') for a in [axs]]
            
        if ai not in [8,9,10,11]:
            [a[ai].xaxis.set_ticklabels([]) for a in [axs]]
        else:
            [a[ai].set_xlabel('x [m]') for a in [axs]]
        
    cbar,ax_cbar = convert_ax_to_cbar(axs[7],'viridis',0,1.5)
    ax_cbar.set_title('flow speed [m/s]')
    
    fig.savefig(animation_folder+r'frame_'+str(f)+r'.png')
    plt.close('all')