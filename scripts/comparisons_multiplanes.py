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

figfolder = r'C:\Users\Luc Deike\Documents\multi_orientation_figures\\'

meta4 = pd.DataFrame()

case_names = [r'piv_sunbathing_centerPlane_on100_off400',
              r'piv_sunbathing_quarterBackFromCenterPlane_on100_off400',
              r'piv_sunbathing_halfBackFromCenterPlane_on100_off400',
              r'piv_sunbathing_threeQuartersBackFromCenterPlane_on100_off400',
              r'piv_sunbathing_inPlaneWithJetsBackFromCenterPlane_on100_off400',
              r'piv_sunbathing_topDown_middlePlane_on100_off400',
              r'piv_sunbathing_topDown_halfUpFromMiddlePlane_on100_off400',
              r'piv_sunbathing_topDown_topJetPlane_on100_off400']
offsets = [(-400,-640)]*len(case_names)
parent_folders = [r'C:\Users\Luc Deike\data_comp3_C\180320\\']*3 + [r'C:\Users\Luc Deike\data_comp3_C\180322\\']*2 + [r'C:\Users\Luc Deike\data_comp3_C\180323\\']*3 
diffusers = [False] * len(case_names)
periods = [500]*len(case_names)
on_portions = [.2]*len(case_names)

meta4['case_name'] = case_names
meta4['offset'] = offsets
meta4['parent_folder'] = parent_folders
meta4['diffuser'] = diffusers
meta4['period'] = periods
meta4['on_portion'] = on_portions
meta4['need2rotate'] = [False,]*len(case_names)
meta4['grid'] = [False,]*len(case_names)
meta4['plane'] = [0, .25, .5, .75, 1.,0,.5,1]
meta4['orientation'] = ['vertical']*5+['horizontal']*3
meta4['description'] = ['center','quarter back','half back','three quarters back','jet plane','center, horz','half up, horz','jet plane, horz']

meta = meta4
meta.index = range(len(meta))

import matplotlib as mpl
import matplotlib.cm as cm

norm = mpl.colors.Normalize(vmin=meta['on_portion'].min(), vmax=meta['on_portion'].max())
cmap = cm.YlGnBu
m = cm.ScalarMappable(norm=norm, cmap=cmap)

c_onportions = {o:m.to_rgba(o) for o in meta['on_portion'].unique()} #,
ls_periods = {100:':',250:'-.',500:'--',0:'-'}
c_periods = {250:'r',500:'cyan'}

colors = ['r','g','b','cyan','orange','purple','magenta','yellow','gray']

#c_desc = {'center':'r','quarter back':'b','half back':'g'}
c_desc = {d:colors[di] for di,d in enumerate(meta['description'].unique())}


vmin = -0.2
vmax = 0.2

'''
Initialize the figures
'''

#fig_turb,axs_turb = plt.subplots(3,len(case_names),figsize=(11,8))

#Figs = piv.PIVComparisonsFigures(3,10,figsize=(17,10),max_speed=0.2,vmin=-.15,vmax=0.15,legend_axi=0)

Figs = piv.PIVComparisonsFigures(2,5,figsize=(17,7.5),max_speed=0.25,vmin=vmin,vmax=vmax,legend_axi=8)
fig_Lint0,axs_Lint0 = plt.subplots(2,5,figsize=(17,7.5),sharex=True,sharey=True); axs_Lint0=axs_Lint0.flatten()
fig_dirHist,axs_dirHist = plt.subplots(2,5,figsize=(17,7.5),subplot_kw=dict(projection='polar')); axs_dirHist=axs_dirHist.flatten()

skip_ax=[6,8]

fig_hist = plt.figure()
ax_hist = fig_hist.add_subplot(111)

fig_fft = plt.figure()
ax_fft = fig_fft.add_subplot(111)

'''
Loop through each case and add the plots to the figures
'''

radial_vals = []
C_dict = {}
for i in meta.index:
    
    parent_folder = meta.loc[i,'parent_folder']
    case_name = meta.loc[i,'case_name']
    offset = meta.loc[i,'offset']
    
    ai = i
    for si in [Figs.legend_axi] + skip_ax:
        if ai>=si:
            ai = ai+1
    
    color = c_desc[meta.loc[i,'description']]
    ls = ls_periods[meta.loc[i,'period']]
    
    label = 'T = '+str(meta.loc[i,'period'])+' ms\n$\phi$ = '+str(int(100*meta.loc[i,'on_portion']))+'/100'
    if meta.loc[i,'description'] is not None:
        label = label+'\n'+meta.loc[i,'description']

    p = pickle.load(open(parent_folder+case_name+'.pkl'))
    p.parent_folder = parent_folder
    p.name_for_save = case_name
    p.associate_flowfield()   
    
    if meta.loc[i,'need2rotate']:
        p.data.ff=piv.rotate_data_90(p.data.ff)
        p.data.ff=piv.rotate_data_90(p.data.ff)
        p.data.ff=piv.rotate_data_90(p.data.ff)
    ff = p.data.ff
    
    g_orig = fluids2d.geometry.GeometryScaler(dx=p.dx,im_shape=(1,1),origin_pos=offset,origin_units='pix')
    g = fluids2d.geometry.create_piv_scaler(p,g_orig)
    
    print(np.shape(ff))
    print(g.im_shape)
    print(g.im_extent)
    
    time = np.arange(0,np.shape(ff)[0]) * p.dt_frames
    
    lim = 0.03
    center_rows,center_cols = g.get_coords(np.array([[0.05,-.05],[-.05,0.05]]))
    
    '''
    Filter the velocity field
    '''
    ff=piv.clip_flowfield(ff,1.5)
    
    '''
    Mean flow and fluctuations
    '''
    
    mean_flow=np.nanmean(ff,axis=0)
    fluc = ff-mean_flow
    u_rms = np.sqrt( np.nanmean( (fluc[:,:,:,0])**2,axis=0) + np.nanmean( (fluc[:,:,:,1])**2,axis=0) )
    u_rms_components = np.sqrt(np.nanmean(fluc**2,axis=0))
    inst_speed = np.linalg.norm(ff,ord=2,axis=3)    
    meanflow_speed = np.sqrt((mean_flow[:,:,0])**2 + (mean_flow[:,:,1])**2)
    
    dudx = np.gradient(fluc[:,:,:,0],axis=1)
    dvdy = np.gradient(fluc[:,:,:,1],axis=2) 
    
    '''
    Add to the figures
    '''
    Figs.add_case(ai,ff,g,time)
    #Figs.update_limits(ai,[-0.1,0.05],[-0.12,0.1])
    Figs.update_limits(ai,[-0.15,0.15],[-0.1,0.1]) # (axis, xlims, ylims)
    Figs.add_text(ai,0.,.08,label)
    Figs.add_rect(ai,-.05,-.05,0.1,0.1,color=color,ls=ls,)
    
    #piv.composite_image_plots(ff,g,time,center_rows,center_cols,vmin=vmin,vmax=vmax)
    
    '''
    piv.add_fieldimg_to_ax(np.nanmean(ff[:,:,center_cols[0]:center_cols[1]+1,0],axis=2),g,axs_composite_compare1[i,0],time=time,slice_dir='vertical',vel_dir='horizontal',vmin=vmin,vmax=vmax)
    piv.add_fieldimg_to_ax(np.nanmean(ff[:,:,center_cols[0]:center_cols[1]+1,1],axis=2),g,axs_composite_compare1[i,1],time=time,slice_dir='vertical',vel_dir='vertical',vmin=vmin,vmax=vmax)

    piv.add_fieldimg_to_ax(np.nanmean(ff[:,center_cols[0]:center_cols[1]+1,:,0],axis=1),g,axs_composite_compare2[i,0],time=time,slice_dir='horizontal',vel_dir='horizontal',vmin=vmin,vmax=vmax)
    piv.add_fieldimg_to_ax(np.nanmean(ff[:,center_cols[0]:center_cols[1]+1,:,1],axis=1),g,axs_composite_compare2[i,1],time=time,slice_dir='horizontal',vel_dir='vertical',vmin=vmin,vmax=vmax)
    '''
    
#    '''
#    temporal autocorrs
#    '''    
#    smaller_rows,smaller_cols = g.get_coords(np.array([[0.005,-.005],[-.005,0.005]]))
#    A = spectra.AutocorrResults(g)
#    A.run_autocorr(ff,time,6000,[smaller_rows[0],smaller_rows[1],smaller_cols[0],smaller_cols[1]])    
#    C_dict[case_name] = A.C_avg
#    lags = A.lags
#    
    '''
    Some scalar data
    '''
    meta.loc[i,'u_rms'] = np.nanmean(u_rms[center_rows[0]:center_rows[1],center_cols[0]:center_cols[1]])
    meta.loc[i,'u_rms_0'] = np.nanmean(u_rms_components[center_rows[0]:center_rows[1],center_cols[0]:center_cols[1],0])
    meta.loc[i,'u_rms_1'] = np.nanmean(u_rms_components[center_rows[0]:center_rows[1],center_cols[0]:center_cols[1],1])
    meta.loc[i,'mean_speed'] = np.nanmean(meanflow_speed[center_rows[0]:center_rows[1],center_cols[0]:center_cols[1]])
    meta.loc[i,'epsilon_0'] = 1e-6 * 15 * np.nanmean((dudx/g.dx)**2)
    meta.loc[i,'epsilon_1'] = 1e-6 * 15 * np.nanmean((dvdy/g.dx)**2)
    
    data_hist = ff[:,center_rows[0]:center_rows[1]+1,center_cols[0]:center_cols[1]+1,0] - np.nanmean(ff[:,center_rows[0]:center_rows[1]+1,center_cols[0]:center_cols[1]+1,0],axis=0)
    data_hist = data_hist.flatten()
    data_hist = data_hist[~np.isnan(data_hist)]
    
    ax_hist.hist(data_hist,bins=np.linspace(-0.5,0.5,1001),alpha=0.5,color=color)
    
    vel_data_u = ff[:,center_rows[0]:center_rows[1],center_cols[0]:center_cols[1],0].flatten()
    vel_data_v = ff[:,center_rows[0]:center_rows[1],center_cols[0]:center_cols[1],1].flatten()
    vel_data_speed = np.sqrt(vel_data_u**2+vel_data_v**2)
    vel_data_dir = np.arctan2(vel_data_v,vel_data_u)
    
    fluc_data_u = fluc[:,center_rows[0]:center_rows[1],center_cols[0]:center_cols[1],0].flatten()
    fluc_data_v = fluc[:,center_rows[0]:center_rows[1],center_cols[0]:center_cols[1],1].flatten()
    fluc_data_speed = np.sqrt(fluc_data_u**2+fluc_data_v**2)
    fluc_data_dir = np.arctan2(fluc_data_v,fluc_data_u)
    
    axs_dirHist[ai].hist(vel_data_dir[~np.isnan(vel_data_dir)],bins=100,alpha=0.5)
    axs_dirHist[ai].hist(fluc_data_dir[~np.isnan(fluc_data_dir)],bins=100,alpha=0.5)
    
#    '''
#    Integral length scales
#    '''
#    center_size = 1 # must be odd
#    
#    center_row = (np.shape(ff)[1]-1)/2
#    center_col = (np.shape(ff)[2]-1)/2
#    
#    search_x = np.shape(ff)[2] - center_size
#    search_y = np.shape(ff)[1] - center_size
#    
#    res,g_r = spectra.calculate_spatial_correlations(g,ff,center_row,center_col,center_size,search_x,search_y)
#    temporal_and_spatial_average = np.nanmean(res,axis=(0,1,2))
#    polar,r,line_avg,integral = spectra.make_radial_correlations(temporal_and_spatial_average,g_r,dr=0.5)
#    radial_vals.append((temporal_and_spatial_average,g_r,polar,r,line_avg,integral))
#    axs_Lint0[ai].plot(r,integral[:,0],color='blue')
#    axs_Lint0[ai].plot(r,integral[:,1],color='red')
#    plt.show()
#    plt.pause(1)

    
Figs.tight_layout()
Figs.add_legends()
#[Figs.remove_axes(a) for a in skip_ax]
[Figs.remove_axes(a) for a in [6]]
Figs.save_figs(figfolder,'multi_orientation')
#Figs.save_figs(figfolder,'4x4')

fig_dirHist.tight_layout()
fig_dirHist.savefig(figfolder+'piv_directional_histograms.pdf')

axs_Lint0[0].set_ylabel('integral of spatial autocorr [m]')
#axs_Lint0[0].set_ylabel('''\int_0^r <u(0) u(r) > ''')
axs_Lint0[5].set_ylabel('integral of spatial autocorr [m]')

[a.set_xlabel('r [m]') for a in axs_Lint0[3:5]]
axs_Lint0[8].plot(np.nan,np.nan,color='b',label='horiztonal velocity')
axs_Lint0[8].plot(np.nan,np.nan,color='r',label='vertical velocity')
axs_Lint0[8].legend()
fig_Lint0.tight_layout()
fig_Lint0.savefig(figfolder+'integral_length_scales.pdf')


fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(111)

c_dict = {None:'k','no grid':'b','grid':'r'}

for i in meta.index:
    c = c_desc[meta.loc[i,'description']]
    ax.plot(meta.loc[i,'plane'],meta.loc[i,'u_rms'],'o',c=c,)
    ax.plot(meta.loc[i,'plane'],meta.loc[i,'mean_speed'],'x',c=c,)
#    
#fig = plt.figure(figsize=(8,4))
#ax = fig.add_subplot(111)
#
#c_dict = {None:'k','big':'b','small':'r'}
#
#for i in meta.index:
#    c = c_dict[meta.loc[i,'description']]
#    ax.plot(meta.loc[i,'on_portion'],meta.loc[i,'L_int_0'],'o',c=c,)
#    ax.plot(meta.loc[i,'on_portion'],meta.loc[i,'L_int_1'],'x',c=c,)


stophere

'''
Integral length scale calculations
'''

fig = plt.figure(figsize=(8,3))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1t=ax1.twinx();ax2t=ax2.twinx()

#c_onportion = {.4:'k',.2:'b'}

[ax1.plot(np.nan,np.nan,label=str(key),color=c_plane[key]) for key in list(c_plane.keys())]
ax1.legend()

for i in meta.index:
    t = radial_vals[i]
    color = c_onportion[meta.loc[i,'description']]
    

    ax1.plot(t[3],t[4][:,0]/t[4][0,0],c=color,ls='--')
    ax1t.plot(t[3],t[5][:,0],c=color,ls='-')
    
    ax2.plot(t[3],t[4][:,1]/t[4][0,1],c=color,ls='--')
    ax2t.plot(t[3],t[5][:,1],c=color,ls='-')
    
ax1.set_ylim([-0.2,1.0])
ax1t.set_ylim([0,0.035])
ax2.set_ylim([-0.2,1.0])
ax2t.set_ylim([0,0.035])

ax1.set_xlim([0,.15])
ax2.set_xlim([0,0.15])
ax1t.set_xlim([0,.15])
ax2t.set_xlim([0,0.15])
        
[ax.set_xlabel('r [m]') for ax in [ax1,ax2]]

ax2.set_yticklabels([])
ax1t.set_yticklabels([])

ax1.set_ylabel('$B_{ii}$ [-]')
ax2t.set_ylabel('$L_{ii}$ [m]')

ax1.set_title('$x$-dir')
ax2.set_title('$y$-dir')

fig.tight_layout()
fig.savefig(figfolder+'4x4_spatial_scales.pdf')





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

fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)


'''
Plot the FFTs
'''
    
fig_ffts,ax = plt.subplots(1,2,figsize=(10,7),sharex=True,sharey=True)
#ax_ffts = ax_ffts.flatten()



for i in meta.index:
    
    parent_folder = meta.loc[i,'parent_folder']
    case_name = meta.loc[i,'case_name']
    offset = meta.loc[i,'offset']
    
    color = c_desc[meta.loc[i,'description']]
    ls = ls_periods[meta.loc[i,'period']]
    
    label = 'T = '+str(meta.loc[i,'period'])+' ms, A = '+str(int(100*meta.loc[i,'on_portion']))+' pct'
    
    C_avg = C_dict[case_name]
    
    favg = []
    for d in [0,1]:
        f,freq = spectra.temporal_spectrum_from_autocorr(C_avg[:,d],lags)
        favg.append(f)
        
    #ax = ax_dict[meta.loc[i,'period']]
    
    #[ax[j].axvline(1000./int(meta.loc[i,'period']),color='gray',alpha=0.5) for j in [0,1]]
    
    ax[0].loglog(freq,favg[0],color=color,alpha=0.8)
    ax[1].loglog(freq,favg[1],color=color,alpha=0.8)

ax_ffts[0,0].set_title('$B_{1,1}$')
ax_ffts[0,1].set_title('$B_{2,2}$')

ax_ffts[3,0].set_xlabel('$f$ [Hz]')
ax_ffts[3,1].set_xlabel('$f$ [Hz]')

[ax_ffts[i,0].set_ylabel('$T$ = '+str(int(np.flipud(np.sort(list(ls_periods.keys())))[i]))+' ms') for i in range(len(ls_periods))]

[a.plot([10**1,10**1.5],[10**2,10**(2-0.5*5./3)],color='gray',ls='--',alpha=0.5) for a in ax_ffts.flatten()]
[a.set_ylim([10**-1,10**3]) for a in ax_ffts.flatten()]
[a.set_xlim([10**-1,10**3]) for a in ax_ffts.flatten()]

fig_ffts.tight_layout()