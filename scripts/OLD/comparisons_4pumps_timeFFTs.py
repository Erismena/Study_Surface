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

case_names = [r'piv_4pumps_topDown_sched-fifthOn_T0100ms_centerView_fps4000',
              r'piv_4pumps_topDown_sched-fifthOn_T0250ms_centerView_fps4000',
              r'piv_4pumps_topDown_sched-halfHalf_T0100ms_centerView_fps4000',
              r'piv_4pumps_topDown_sched-halfHalf_T0500ms_centerView_fps4000',
              r'piv_4pumps_topDown_sched-tenthOn_T0250ms_centerView_fps4000',
              r'piv_4pumps_topDown_sched-thirdOn_T0100ms_centerView_fps4000',
              r'piv_4pumps_topDown_sched-thirdOn_T0250ms_centerView_fps4000',
              r'piv_4pumps_topDown_sched-thirdOn_T0500ms_centerView_fps4000',
              r'piv_4pumps_topDown_sched-halfHalf_T1000ms_centerView_fps4000']
offsets = [(-345+256,-360+256)]*len(case_names)
parent_folders = [r'C:\Users\Luc Deike\highspeed_data\171221\\']*len(case_names)
diffusers = [False] * len(case_names)
periods = [100,250,100,500,250,100,250,500,1000]
on_portions = [.2,.2,.5,.5,.1,1./3,1./3,1./3,.5]

c_onportions = {0.1:'b',0.2:'g',1./3:'orange',0.5:'r',1:'cyan'}
ls_periods = {100:':',250:'-.',500:'--',1000:'-'}

meta['case_name'] = case_names
meta['offset'] = offsets
meta['parent_folder'] = parent_folders
meta['diffuser'] = diffusers
meta['period'] = periods
meta['on_portion'] = on_portions


need2rotate = False

'''
Function to get the autocorrelation and associated length scales.
'''



'''
FFT with longer data
'''

#fig = plt.figure()
#ax_fft = fig.add_subplot(311)
#ax_fftlog = fig.add_subplot(312)
#ax_fftavg = fig.add_subplot(313)

fig = plt.figure(figsize=(11,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

fig = plt.figure()
ax_C1 = fig.add_subplot(121)
ax_C2 = fig.add_subplot(122)

fig = plt.figure()
ax_hist_u = fig.add_subplot(211)
ax_hist_v = fig.add_subplot(212)


fig_intns,axs_intns = plt.subplots(3,3,figsize=(11,8)); axs_intns = axs_intns.flatten()
fig_mean,axs_mean = plt.subplots(3,3,figsize=(11,8)); axs_mean = axs_mean.flatten()
fig_fluc,axs_fluc = plt.subplots(3,3,figsize=(11,8)); axs_fluc = axs_fluc.flatten()
fig_isgood,axs_isgood = plt.subplots(3,3,figsize=(11,8)); axs_isgood = axs_isgood.flatten()



C_dict = {}

for i in [6]:
    
    parent_folder = meta.loc[i,'parent_folder']
    case_name = meta.loc[i,'case_name']
    offset = meta.loc[i,'offset']
    
    print(case_name)
    
    ai = i
    
    color = c_onportions[meta.loc[i,'on_portion']]
    ls = ls_periods[meta.loc[i,'period']]
    
    label = 'T = '+str(meta.loc[i,'period'])+' ms, A = '+str(int(100*meta.loc[i,'on_portion']))+' pct'
    
    p = pickle.load(open(parent_folder+case_name+'.pkl'))
    p.parent_folder = parent_folder
    p.associate_flowfield()
    
    ff = p.data.ff[0:42000,:,:,:]
    
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
    fluc_speed = np.sqrt(fluc[:,:,:,0]**2+fluc[:,:,:,1]**2)
    u_rms = np.sqrt( np.nanmean( (fluc[:,:,:,0])**2,axis=0) + np.nanmean( (fluc[:,:,:,1])**2,axis=0) )
    inst_speed = np.linalg.norm(ff,ord=2,axis=3)
    
    dudx = np.gradient(fluc[:,:,:,0],axis=1)
    dvdy = np.gradient(fluc[:,:,:,1],axis=2) 
    
    meanflow_speed = np.sqrt((mean_flow[:,:,0])**2 + (mean_flow[:,:,1])**2)
    
    piv.add_fieldimg_to_ax(np.log10(u_rms / meanflow_speed),g,axs_intns[i],time=time,slice_dir=None,vel_dir=None,vmin=-1,vmax=1)
    #piv.add_fieldimg_to_ax(np.sqrt(np.nanmean(ff[:,:,:,0],axis=0)**2+np.nanmean(ff[:,:,:,1],axis=0)**2),g,axs_turb[1,i],time=time,slice_dir=None,vel_dir=None,vmin=0,vmax=0.5)
    piv.add_fieldimg_to_ax(meanflow_speed,g,axs_mean[i],time=time,slice_dir=None,vel_dir=None,vmin=0,vmax=0.5)
    piv.add_fieldimg_to_ax(u_rms,g,axs_fluc[i],time=time,slice_dir=None,vel_dir=None,vmin=0,vmax=0.5)
    
    intns = u_rms / meanflow_speed
    is_good = np.ones(np.shape(intns))
    is_good[intns<2] = 0
    is_good[meanflow_speed> 0.05] = 0
    is_good[u_rms < 0.05] = 0
    
    is_good = scipy.signal.medfilt(is_good,3)
    
    piv.add_fieldimg_to_ax(is_good,g,axs_isgood[i],time=time,slice_dir=None,vel_dir=None,vmin=0,vmax=1)

    
    '''
    s_hist_u = pd.Series(ff[:,center_rows[0]:center_rows[1],center_cols[0]:center_cols[1],0].flatten())
    s_hist_v = pd.Series(ff[:,center_rows[0]:center_rows[1],center_cols[0]:center_cols[1],1].flatten())
    
    if meta.loc[i,'diffuser']==True:
        s_hist_u.plot.kde(ax=ax_hist_u,alpha=0.8,color='k',lw=3,ls='-',label='_nolegend_')
        s_hist_v.plot.kde(ax=ax_hist_v,alpha=0.8,color='k',lw=3,ls='-',label='_nolegend_')
        label = label+', with diffusers'
    s_hist_u.plot.kde(ax=ax_hist_u,alpha=0.8,label=label,color=color,ls=ls)
    s_hist_v.plot.kde(ax=ax_hist_v,alpha=0.8,label=label,color=color,ls=ls)
    
    plt.show()
    plt.pause(0.5)
    
    '''
    
    '''
    FFTs
    '''
    
#    num_lags = 36000
#    #num_lags = 200
#    
#    #C_arr = np.zeros((num_lags,(center_rows[1]-center_rows[0]+1),(center_cols[1]-center_cols[0]+1),2))
#    C_avg = np.zeros((num_lags,2))
#    
#    keep_first=num_lags/2
#    favg= np.zeros((keep_first,2))
#    
#    for d in [0,1]:
#        fluc[:,:,:,d] = piv.fill_nans_3d(fluc[:,:,:,d])
#    C_arr,lags = spectra.autocorr(fluc[:,center_rows[0]:center_rows[1]+1,center_cols[0]:center_cols[1]+1,:],num_lags=num_lags,dt=p.dt_frames)
#    for d in [0,1]:    
#        C_avg[:,d] = spectra.nanmean(np.nanmean(C_arr[:,:,:,d],axis=1),axis=1)        
#        favg[:,d],freq = piv.temporal_spectrum_from_autocorr(C_avg[:,d],lags)
#    
#    C_dict[case_name] = C_arr
    
#    lim = 0.02
#    center_rows,center_cols = g.get_coords(np.array([[lim,-1*lim],[-1*lim,lim]]))
#    
#    num_lags = 16000
#    #num_lags = 200
#    
#    #C_arr = np.zeros((num_lags,(center_rows[1]-center_rows[0]+1),(center_cols[1]-center_cols[0]+1),2))
#    C_avg = np.zeros((num_lags,2))
#    
#    keep_first=num_lags/2
#    favg= np.zeros((keep_first,2))
#    
#    for d in [0,1]:
#        fluc[:,:,:,d] = piv.fill_nans_3d(fluc[:,:,:,d])
#    C_arr,lags = spectra.autocorr(fluc[:,center_rows[0]:center_rows[1]+1,center_cols[0]:center_cols[1]+1,:],num_lags=num_lags,dt=p.dt_frames)
#    for d in [0,1]:    
#        C_avg[:,d] = np.nanmean(np.nanmean(C_arr[:,:,:,d],axis=1),axis=1)
#        favg[:,d],freq = spectra.temporal_spectrum_from_autocorr(C_avg[:,d],lags)
#    
#    C_dict[case_name] = C_arr
#    
    '''
    Show the center rectangle
    '''    
    for a in [axs_intns,axs_mean,axs_fluc,axs_isgood]:
        r = matplotlib.patches.Rectangle([-1*lim,-1*lim],lim*2,lim*2,edgecolor=color,ls=ls,lw=2,fill=False)
        a[ai].add_patch(r)
        
    stophere
    
'''
Plot the FFTs
'''

#np.save(figfolder+'C_dict_180103b_1cmsq.np',C_dict)
pd.to_pickle(C_dict,figfolder+'C_dict_180116.pkl')
    
fig_ffts,ax_ffts = plt.subplots(4,2,figsize=(10,7),sharex=True,sharey=True)
#ax_ffts = ax_ffts.flatten()
ax_dict = {np.flipud(np.sort(list(ls_periods.keys())))[i]:ax_ffts[i] for i in range(len(ls_periods))}

num_use = 24000
lags = lags[0:num_use]
favg= np.zeros((num_use/2,2))

for i in meta.index:
    
    parent_folder = meta.loc[i,'parent_folder']
    case_name = meta.loc[i,'case_name']
    offset = meta.loc[i,'offset']
    
    color = c_onportions[meta.loc[i,'on_portion']]
    ls = ls_periods[meta.loc[i,'period']]
    
    label = 'T = '+str(meta.loc[i,'period'])+' ms, A = '+str(int(100*meta.loc[i,'on_portion']))+' pct'
    
    C_arr = C_dict[case_name]
    
    C_arr = C_arr[0:num_use,:,:,:]
    
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

fig_ffts.tight_layout()
fig_ffts.savefig(figfolder+'baselines_timespectra.pdf')

'''
FFTs by on-portion
'''

#fig_ffts,ax_ffts = plt.subplots(5,2,figsize=(10,7),sharex=True,sharey=True)
##ax_ffts = ax_ffts.flatten()
#ax_dict = {np.flipud(np.sort(list(c_onportions.keys())))[i]:ax_ffts[i] for i in range(len(c_onportions))}
#
#for i in meta.index:
#    
#    parent_folder = meta.loc[i,'parent_folder']
#    case_name = meta.loc[i,'case_name']
#    offset = meta.loc[i,'offset']
#    
#    color = c_onportions[meta.loc[i,'on_portion']]
#    ls = ls_periods[meta.loc[i,'period']]
#    
#    #label = '$\phi$ = '+str(meta.loc[i,'on_portion'])+' ms, A = '+str(int(100*meta.loc[i,'on_portion']))+' pct'
#    
#    C_arr = C_dict[case_name]
#    
#    C_avg = np.nanmean(np.nanmean(C_arr[:,:,:,:],axis=1),axis=1)
#    for d in [0,1]:
#        favg[:,d],freq = piv.temporal_spectrum_from_autocorr(C_avg[:,d],lags)
#        
#    ax = ax_dict[meta.loc[i,'on_portion']]
#    
#    #[ax[j].axvline(1000./int(meta.loc[i,'period']),color='gray',alpha=0.5) for j in [0,1]]
#    
#    ax[0].loglog(freq,favg[:,0],color=color,ls=ls,alpha=0.8)
#    ax[1].loglog(freq,favg[:,1],color=color,ls=ls,alpha=0.8)
#
#ax_ffts[0,0].set_title('$B_{1,1}$')
#ax_ffts[0,1].set_title('$B_{2,2}$')
#
#ax_ffts[3,0].set_xlabel('$f$ [Hz]')
#ax_ffts[3,1].set_xlabel('$f$ [Hz]')
#
#[ax_ffts[i,0].set_ylabel('$\phi$ = '+str(int(100*np.flipud(np.sort(list(c_onportions.keys())))[i]))) for i in range(len(c_onportions))]
#
#[a.plot([10**1,10**1.5],[10**2,10**(2-0.5*5./3)],color='gray',ls='--',alpha=0.5) for a in ax_ffts.flatten()]
#
#fig_ffts.tight_layout()
        
'''
Plots for the timescales
'''
    
fig_intscale = plt.figure()
ax1_intscale = fig_intscale.add_subplot(121)
ax2_intscale = fig_intscale.add_subplot(122)
for i in meta.index:
    case_name=meta.loc[i,'case_name']
    
    color = c_onportions[meta.loc[i,'on_portion']]
    ls = ls_periods[meta.loc[i,'period']]
    
    C_arr = C_dict[case_name]
    C_avg = np.nanmean(np.nanmean(C_arr,axis=1),axis=1)
    ax1_intscale.plot(lags,np.cumsum(C_avg[:,0])/2000,color=color,ls=ls,alpha=0.8)
    ax2_intscale.plot(lags,np.cumsum(C_avg[:,1])/2000,color=color,ls=ls,alpha=0.8)
    


'''
Animation
'''

#fig=plt.figure()
#ax=fig.add_subplot(111)
#for i in range(0,500):
#    ax.clear()
#    ax.imshow(fluc[i,:,:,0],vmin=-0.5,vmax=0.5)
#    plt.show()
#    plt.pause(0.1)