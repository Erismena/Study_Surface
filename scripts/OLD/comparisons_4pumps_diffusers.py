# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:08:23 2017

@author: danjr
"""

import fluids2d.piv as piv
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

'''
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
offsets = [(-370,-360)]*len(case_names)
parent_folders = [r'C:\Users\Luc Deike\highspeed_data\171221\\']*len(case_names)
diffusers = [False] * len(case_names)
periods = [250,500,100,250,100,250,500,100,250,500,1000]
on_portions = [0.1,0.1,0.2,0.2,1./3,1./3,1./3,0.5,0.5,0.5,0.5]
'''

'''
parent_folders = parent_folders+[r'D:\high_speed_data\171204\\']
case_names = case_names + [r'piv_4pumps_allAlwaysOn_fps4000']
offsets = offsets+[(-360,-640)]
diffusers=diffusers+[False]
periods = periods+[1000]
on_portions = on_portions+[1]
'''

parent_folders = []
diffusers = []
offsets = []
periods = []
on_portions = []
case_names = []

c_periods = {100:'b',250:'g',500:'r',1000:'orange'}
ls_onportions = {0.1:':',0.2:'-.',1./3:'--',0.5:'-',1:'-'}

c_onportions = {0.1:'b',0.2:'g',1./3:'orange',0.5:'r',1:'cyan'} #,1:'cyan'
ls_periods = {100:':',250:'-.',500:'--',1000:'-'}

case_names = case_names+[r'piv_4pumpsDiffusers_topDown_sched-thirdOn_T0100ms_fullView_fps4000_dur4s',
              r'piv_4pumpsDiffusers_topDown_sched-thirdOn_T0250ms_fullView_fps4000_dur4s',
              r'piv_4pumpsDiffusers_topDown_sched-thirdOn_T0500ms_fullView_fps4000_dur4s',]
              #r'piv_4pumpsDiffusers_topDown_sched-alwaysOn_fullView_fps4000_dur4s']
parent_folders = parent_folders+[r'D:\high_speed_data\171222\\']*len(case_names)
diffusers = diffusers + [True]*len(case_names)
offsets=offsets+[(-415,-360)]*3 #+ [(-370,-640)]
periods = periods + [100,250,500,]#1000
on_portions = on_portions + [1./3,1./3,1./3,]#1

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

fig_num_rows = 1
fig_num_cols = 4
legend_ax_num = 3

fig_width = 11.
fig_height = 2.0


fig_intns,axs_intns = plt.subplots(fig_num_rows,fig_num_cols,figsize=(fig_width,fig_height)); axs_intns = axs_intns.flatten()
fig_mean,axs_mean = plt.subplots(fig_num_rows,fig_num_cols,figsize=(fig_width,fig_height)); axs_mean = axs_mean.flatten()
fig_fluc,axs_fluc = plt.subplots(fig_num_rows,fig_num_cols,figsize=(fig_width,fig_height)); axs_fluc = axs_fluc.flatten()
fig_isgood,axs_isgood = plt.subplots(fig_num_rows,fig_num_cols,figsize=(fig_width,fig_height)); axs_isgood = axs_isgood.flatten()


fig_fft = plt.figure()
ax_fft = fig_fft.add_subplot(111)




#fig_mean,axs_mean = plt.subplots(2,len(case_names),figsize=(11,8))

fig_hist = plt.figure()
ax_hist_u = fig_hist.add_subplot(211)
ax_hist_v = fig_hist.add_subplot(212,sharex=ax_hist_u)

fig_contour = plt.figure()
ax_contour = fig_contour.add_subplot(111)

fig = plt.figure()
ax_C1 = fig.add_subplot(121)
ax_C2 = fig.add_subplot(122)


fig = plt.figure(figsize=(11,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)


'''
Loop through each case and add the plots to the figures
'''

#for i,(parent_folder,case_name) in enumerate(zip(parent_folders,case_names)):
C_dict = {}
for i in meta.index:
    
    parent_folder = meta.loc[i,'parent_folder']
    case_name = meta.loc[i,'case_name']
    offset = meta.loc[i,'offset']
    
    ai = i
    if i>=legend_ax_num:
        ai=ai+1
    
    #color = c_periods[meta.loc[i,'period']]
    color = c_onportions[meta.loc[i,'on_portion']]
    #ls = ls_onportions[meta.loc[i,'on_portion']]
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
    
    piv.add_fieldimg_to_ax(np.log10(u_rms / meanflow_speed),g,axs_intns[ai],time=time,slice_dir=None,vel_dir=None,vmin=-1,vmax=1,cmap_other='coolwarm')
    #piv.add_fieldimg_to_ax(np.sqrt(np.nanmean(ff[:,:,:,0],axis=0)**2+np.nanmean(ff[:,:,:,1],axis=0)**2),g,axs_turb[1,i],time=time,slice_dir=None,vel_dir=None,vmin=0,vmax=0.5)
    piv.add_fieldimg_to_ax(meanflow_speed,g,axs_mean[ai],time=time,slice_dir=None,vel_dir=None,vmin=0,vmax=0.5)
    piv.add_fieldimg_to_ax(u_rms,g,axs_fluc[ai],time=time,slice_dir=None,vel_dir=None,vmin=0,vmax=0.5)
    
    #piv.add_fieldimg_to_ax(mean_flow[:,:,0],g,axs_mean[0,i],vel_dir='horizontal',vmin=vmin,vmax=vmax)
    #piv.add_fieldimg_to_ax(mean_flow[:,:,1],g,axs_mean[1,i],vel_dir='vertical',vmin=vmin,vmax=vmax)
    
    '''
    Contours
    '''
    
    #intns = scipy.signal.medfilt(u_rms,5) / scipy.signal.medfilt(meanflow_speed,5) 
    
    intns = u_rms / meanflow_speed
    is_good = np.ones(np.shape(intns))
    is_good[intns<2] = 0
    is_good[meanflow_speed> 0.05] = 0
    is_good[u_rms < 0.05] = 0
    
    is_good = scipy.signal.medfilt(is_good,3)
    
    piv.add_fieldimg_to_ax(is_good,g,axs_isgood[ai],time=time,slice_dir=None,vel_dir=None,vmin=0,vmax=1)
    
    
    thresh = 0.5
    contours = measure.find_contours(is_good,thresh)
    
    for n, contour in enumerate(contours):
        contour = g.get_loc(contour).T
        if meta.loc[i,'diffuser']==True:
            ax_contour.plot(contour[:, 1], contour[:, 0],alpha=0.8,color='k',lw=3,ls='-',label='_nolegend_')
        ax_contour.plot(contour[:, 1], contour[:, 0], color=color,ls=ls,alpha=0.8)
        
    plt.show()
    plt.pause(0.1)
    
    #[a[i].set_title(label) for a in [axs_intns,axs_mean,axs_fluc]]
    #[a[ai].set_xlim([-0.07,0.07]) for a in [axs_intns,axs_mean,axs_fluc,axs_isgood]]
    [g.set_axes_limits(a[ai]) for a in [axs_intns,axs_mean,axs_fluc,axs_isgood]]
    #[a[ai].set_ylim([-0.075,0.075]) for a in [axs_intns,axs_mean,axs_fluc,axs_isgood]]
    [a[ai].text(0.070,0.032,label,ha='right',va='top',color='white',fontsize=10) for a in [axs_intns,axs_mean,axs_fluc,axs_isgood]]
    
    if ai not in [0,4,8]:
        [a[ai].yaxis.set_ticklabels([]) for a in [axs_intns,axs_mean,axs_fluc,axs_isgood]]
    else:
        [a[ai].set_ylabel('y [m]') for a in [axs_intns,axs_mean,axs_fluc,axs_isgood]]
        

    [a[ai].set_xlabel('x [m]') for a in [axs_intns,axs_mean,axs_fluc,axs_isgood]]
    
    '''
    FFTs
    '''
#    
    '''
    lim = 0.02
    center_rows,center_cols = g.get_coords(np.array([[lim,-1*lim],[-1*lim,lim]]))
    
    num_lags = 6000
    
    C_arr = np.zeros((num_lags,(center_rows[1]-center_rows[0]+1),(center_cols[1]-center_cols[0]+1),2))
    C_avg = np.zeros((num_lags,2))
    
    keep_first=num_lags/2
    favg= np.zeros((keep_first,2))
    
    for d in [0,1]:
        fluc[:,:,:,d] = piv.fill_nans_3d(fluc[:,:,:,d])
        C_arr[:,:,:,d],lags = piv.autocorr(fluc[:,center_rows[0]:center_rows[1]+1,center_cols[0]:center_cols[1]+1,d],num_lags=num_lags,dt=p.dt_frames)
        C_avg[:,d] = np.nanmean(np.nanmean(C_arr[:,:,:,d],axis=1),axis=1)
        
        favg[:,d],freq = piv.temporal_spectrum_from_autocorr(C_avg[:,d],lags)
    
    C_dict[case_name] = C_arr            
    '''
    
    '''
    Histogram of the center points
    '''
    
    lim = 0.02
    center_rows,center_cols = g.get_coords(np.array([[lim,-1*lim],[-1*lim,lim]]))
#    
#    s_hist_u = pd.Series(ff[:,center_rows[0]:center_rows[1],center_cols[0]:center_cols[1],0].flatten())
#    s_hist_v = pd.Series(ff[:,center_rows[0]:center_rows[1],center_cols[0]:center_cols[1],1].flatten())
#    
#    if meta.loc[i,'diffuser']==True:
#        s_hist_u.plot.kde(ax=ax_hist_u,alpha=0.8,color='k',lw=3,ls='-',label='_nolegend_')
#        s_hist_v.plot.kde(ax=ax_hist_v,alpha=0.8,color='k',lw=3,ls='-',label='_nolegend_')
#        label = label+', with diffusers'
#    s_hist_u.plot.kde(ax=ax_hist_u,alpha=0.8,label=label,color=color,ls=ls)
#    s_hist_v.plot.kde(ax=ax_hist_v,alpha=0.8,label=label,color=color,ls=ls)
#    
#    plt.show()
#    plt.pause(0.5)
#    
#    [a[i].set_title(label) for a in [axs_intns,axs_mean,axs_fluc]]
    
    '''
    Show the center rectangle
    '''    
    for a in [axs_intns,axs_mean,axs_fluc,axs_isgood]:
        r = matplotlib.patches.Rectangle([-1*lim,-1*lim],lim*2,lim*2,edgecolor=color,ls=ls,lw=2,fill=False)
        a[ai].add_patch(r)
    
    '''
    Some scalar data
    '''
    meta.loc[i,'u_rms'] = np.nanmean(u_rms[center_rows[0]:center_rows[1],center_cols[0]:center_cols[1]])
    meta.loc[i,'mean_speed'] = np.nanmean(meanflow_speed[center_rows[0]:center_rows[1],center_cols[0]:center_cols[1]])
    
    
ax_hist_u.legend()
ax_hist_u.set_xlim([-1,1])
ax_hist_u.set_yscale('log')
ax_hist_u.set_ylim([0.0001,10])
ax_hist_v.set_yscale('log')
ax_hist_v.set_ylim([0.0001,10])

ax_contour.set_aspect('equal')

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
    for d in meta['diffuser'].unique():
        meta_on = meta[(meta['on_portion']==o)&(meta['diffuser']==d)]
        if d==True:
            ax.plot(meta_on['period'],meta_on['u_rms'],c='k',ls='-',lw=3)
            ax.plot(meta_on['period'],meta_on['mean_speed'],c='k',ls='-',lw=3)
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
#fig.savefig(figfolder+'baseline_turb_avgs.pdf')

for f in [fig_intns,fig_mean,fig_fluc,fig_isgood]:
    f.tight_layout()
    
    
def convert_ax_to_cbar(ax,cmap,vmin,vmax):
        
    import matplotlib as mpl
    pos = ax.get_position().bounds
    ax.set_position([pos[0],pos[1]+pos[3]/2,pos[2],(pos[3])/5])
    
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    return cb1,ax


cbar,ax_cbar = convert_ax_to_cbar(axs_intns[legend_ax_num],'coolwarm',-1,1)
cbar.set_ticks([-1,0,1])
cbar.set_ticklabels([0.1,1,10])
ax_cbar.set_title('turbulence intensity')

cbar,ax_cbar = convert_ax_to_cbar(axs_mean[legend_ax_num],'viridis',0,0.5)
ax_cbar.set_title('mean flow speed [m/s]')

cbar,ax_cbar = convert_ax_to_cbar(axs_fluc[legend_ax_num],'viridis',0,0.5)
ax_cbar.set_title('$u_\mathrm{rms}$ [m/s]')

fig_intns.savefig(figfolder+'diffusers_intns.pdf')
fig_mean.savefig(figfolder+'diffusers_mean.pdf')
fig_fluc.savefig(figfolder+'diffusers_fluc.pdf')

'''
FFTs on separate subplots
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
    
    
    for d in [0,1]:
        C_avg = np.nanmean(np.nanmean(C_arr[:,:,:,d],axis=1),axis=1)
        favg[:,d],freq = piv.temporal_spectrum_from_autocorr(C_avg,lags)
        
    ax = ax_dict[meta.loc[i,'period']]
    
    [ax[j].axvline(1000./int(meta.loc[i,'period']),color='gray',alpha=0.5) for j in [0,1]]
    
    ax[0].loglog(freq,favg[:,0],color=color,alpha=0.8)
    ax[1].loglog(freq,favg[:,1],color=color,alpha=0.8)

ax_ffts[0,0].set_title('$B_{1,1}$')
ax_ffts[0,1].set_title('$B_{2,2}$')

ax_ffts[3,0].set_xlabel('$f$ [Hz]')
ax_ffts[3,1].set_xlabel('$f$ [Hz]')

[ax_ffts[i,0].set_ylabel('T = '+str(int(np.flipud(np.sort(list(ls_periods.keys())))[i]))+' ms') for i in range(len(ls_periods))]

[a.plot([10**1,10**1.5],[10**2,10**(2-0.5*5./3)],color='gray',ls='--',alpha=0.5) for a in ax_ffts.flatten()]

fig_ffts.tight_layout()
fig_ffts.savefig(figfolder+'diffusers_timespectra.pdf')