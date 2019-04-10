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
import os.path

figfolder = r'C:\Users\Luc Deike\Documents\larger_tank_data_figures\\'


meta4 = pd.DataFrame()

case_names = [r'piv_4pumps_stackedParallel_alwaysOn_T10s_L450mm_h080mm_fps4000',
              r'piv_4pumps_stackedParallel_grids5cm_alwaysOn_T10s_L450mm_h080mm_fps4000',
              r'piv_4pumps_stackedParallel_grids10cm_alwaysOn_T10s_L450mm_h080mm_fps4000']
offsets = [(-400,-730)]*len(case_names)
parent_folders = [r'D:\data_comp3_D\180205\\'] * len(case_names)
diffusers = [False] * len(case_names)
periods = [0]*len(case_names)
on_portions = [1]*len(case_names)

meta4['case_name'] = case_names
meta4['offset'] = offsets
meta4['parent_folder'] = parent_folders
meta4['diffuser'] = diffusers
meta4['period'] = periods
meta4['on_portion'] = on_portions
meta4['need2rotate'] = False
meta4['grid'] = [False,'5 cm','10 cm']
meta4['grid_dist'] = [0,5,10]
meta4['description'] = [None,'grid 5 cm','grid 10 cm']

meta5 = pd.DataFrame()

case_names = [r'piv_4pumps_grid5cm_stackedParallel_sunbathing_h080mm_L450mm_T10s_meanon100_meanoff400_fps4000',
              r'piv_4pumps_grid5cm_stackedParallel_sunbathing_h080mm_L450mm_T10s_meanon200_meanoff300_fps4000',
              r'piv_4pumps_grid10cm_stackedParallel_sunbathing_h080mm_L450mm_T10s_meanon100_meanoff400_fps4000',
              r'piv_4pumps_grid10cm_stackedParallel_sunbathing_h080mm_L450mm_T10s_meanon200_meanoff300_fps4000']
offsets =  [(-500,-500)]*4
parent_folders = [r'D:\data_comp3_D\180202\\']*4
diffusers =  [False]*4
periods = [500,500,500,500]
on_portions = [.2,.4,.2,.4]
grids = [5]*2 + [10]*2

meta5['case_name'] = case_names
meta5['offset'] = offsets
meta5['parent_folder'] = parent_folders
meta5['diffuser'] = diffusers
meta5['period'] = periods
meta5['on_portion'] = on_portions
meta5['need2rotate'] = False
meta5['grid'] = grids
meta5['grid_dist'] = [5,5,10,10]
meta5['description'] =['grid 5 cm']*2 + ['grid 10 cm']*2


meta = pd.concat([meta4,meta5],axis=0)
meta.index = range(len(meta))



import matplotlib as mpl
import matplotlib.cm as cm

norm = mpl.colors.Normalize(vmin=meta['on_portion'].min(), vmax=meta['on_portion'].max())
cmap = cm.YlGnBu
m = cm.ScalarMappable(norm=norm, cmap=cmap)


c_onportions = {o:m.to_rgba(o) for o in meta['on_portion'].unique()} #,
ls_periods = {100:':',250:'-.',500:'--',0:'-'}
c_periods = {0:'pink',250:'r',500:'cyan'}

'''
Initialize the figures
'''

#fig_turb,axs_turb = plt.subplots(3,len(case_names),figsize=(11,8))

#Figs = piv.PIVComparisonsFigures(3,10,figsize=(17,10),max_speed=0.2,vmin=-.15,vmax=0.15,legend_axi=0)
Figs = piv.PIVComparisonsFigures(2,4,figsize=(13,4.5),max_speed=0.4,vmin=-.5,vmax=0.5,legend_axi=3)
skip_ax = []

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
    for si in [Figs.legend_axi] + skip_ax:
        if ai>=si:
            ai = ai+1
    
    color = c_onportions[meta.loc[i,'on_portion']]
    ls = ls_periods[meta.loc[i,'period']]
    
    label = 'T = '+str(meta.loc[i,'period'])+' ms\n$\phi$ = '+str(int(100*meta.loc[i,'on_portion']))+'/100'
    if meta.loc[i,'description'] is not None:
        label = label+'\n'+meta.loc[i,'description']

    p = pickle.load(open(parent_folder+case_name+'.pkl'))
    p.parent_folder = parent_folder
    p.name_for_save = case_name
    p.associate_flowfield()   
    
    print(piv.check_concatenation(p))
    
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
    center_rows,center_cols = g.get_coords(np.array([[0.05,-.05],[-.06,0.06]]))
    
    '''
    Filter the velocity field
    '''
    ff=piv.clip_flowfield(ff,5)
    
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
    Figs.update_limits(ai,[-0.2,0.15],[-0.12,0.12])
    #Figs.update_limits(ai,[-0.1,0.15],[-0.12,0.1])
    Figs.add_text(ai,-.08,.09,label)
    Figs.add_rect(ai,-.06,-.05,0.12,0.1,color=color,ls=ls,)
    
#    '''
#    FFTs
#    '''
#    smaller_rows,smaller_cols = g.get_coords(np.array([[0.01,-.01],[-.01,0.01]]))
#    A = spectra.AutocorrResults(g,parent_folder=parent_folder,case_name=case_name)
#    
#    if os.path.isfile(A.filepath):
#        print('loading the existing file')
#        A_old = pickle.load(open(A.filepath,'rb'))
#        A = A_old
#    else:
#        A.run_autocorr(ff,time,2000*8,coords_used=[smaller_rows[0],smaller_rows[1],smaller_cols[0],smaller_cols[1]])
#    
#    C_dict[case_name] = A.C_avg
#    lags = A.lags
#    
#    
    '''
    Some scalar data
    '''
    meta.loc[i,'u_rms'] = np.nanmean(u_rms[center_rows[0]:center_rows[1],center_cols[0]:center_cols[1]])
    meta.loc[i,'mean_speed'] = np.nanmean(meanflow_speed[center_rows[0]:center_rows[1],center_cols[0]:center_cols[1]])
    meta.loc[i,'epsilon_0'] = 1e-6 * 15 * np.nanmean((dudx/g.dx)**2)
    meta.loc[i,'epsilon_1'] = 1e-6 * 15 * np.nanmean((dvdy/g.dx)**2)
    
Figs.tight_layout()
Figs.add_legends()
#[Figs.remove_axes(a) for a in skip_ax]
Figs.save_figs(figfolder,'larger_grids')


fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(111)

#ax.plot(meta['grid_dist'],meta['u_rms'],'-o')
#ax.plot(meta['grid_dist'],meta['mean_speed'],'-^')


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
Legend
'''
ax.plot([np.nan,np.nan],[np.nan,np.nan],'o--',color='k',label='$u_\mathrm{rms}$')
ax.plot([np.nan,np.nan],[np.nan,np.nan],'^-.',color='k',label='$|\overline{\mathbf{U}}|$')

c_grids = {0:'k',5:'g',10:'orange'}

for key in np.sort(list(c_grids.keys())):
    color = c_grids[key]
    #ax.plot([np.nan,np.nan],[np.nan,np.nan],'-',color=color,label='$\phi$ = '+str(int(100.*key))+'/100')
    ax.plot([np.nan,np.nan],[np.nan,np.nan],'-',color=color,label=str(key)+' cm')


ax.legend()


for o in meta['grid_dist'].unique():
    meta_on = meta[meta['grid_dist']==o].sort_values('on_portion')
    ax.plot(meta_on['on_portion'],meta_on['u_rms'],c=c_grids[o],ls='--',marker='o')
    ax.plot(meta_on['on_portion'],meta_on['mean_speed'],c=c_grids[o],ls='-.',marker='^')


ax.set_xlabel('$\phi$ [-]')
ax.set_ylabel('speed [m/s]')
ax.set_ylim([0,0.25])

fig.tight_layout()
fig.savefig(r'C:\Users\Luc Deike\Documents\larger_tank_data_figures\\larger_grids_center_flow_comparisons.pdf')


'''
Plot the FFTs
'''
    
fig_ffts,ax_ffts = plt.subplots(4,2,figsize=(10,7),sharex=True,sharey=True)
#ax_ffts = ax_ffts.flatten()
ax_dict = {np.flipud(np.sort(list(ls_periods.keys())))[i]:ax_ffts[i] for i in range(len(ls_periods))}

fig_C,axs_C = plt.subplots(1,2); axs_C=axs_C.flatten()

c_i = ['r','g','b']
for i in meta.index:
    
    parent_folder = meta.loc[i,'parent_folder']
    case_name = meta.loc[i,'case_name']
    offset = meta.loc[i,'offset']
    
    #color = c_onportions[meta.loc[i,'on_portion']]
    color = c_i[i]
    ls = ls_periods[meta.loc[i,'period']]
    
    label = 'T = '+str(meta.loc[i,'period'])+' ms, A = '+str(int(100*meta.loc[i,'on_portion']))+' pct'
    
    C_arr = C_dict[case_name]
    
    #C_avg = np.nanmean(np.nanmean(C_arr[:,:,:,:],axis=1),axis=1)
    C_avg = C_arr
    favg = np.zeros((len(C_avg)/2,2))
    for d in [0,1]:
        favg[:,d],freq = spectra.temporal_spectrum_from_autocorr(C_avg[:,d],lags)
        
    ax = ax_dict[meta.loc[i,'period']]
    
    #[ax[j].axvline(1000./int(meta.loc[i,'period']),color='gray',alpha=0.5) for j in [0,1]]
    
    ax[0].loglog(freq,favg[:,0],color=color,alpha=0.8)
    ax[1].loglog(freq,favg[:,1],color=color,alpha=0.8)
    
    axs_C[0].plot(lags,C_avg[:,0],color=color)
    axs_C[1].plot(lags,C_avg[:,1],color=color)

ax_ffts[0,0].set_title('$B_{1,1}$')
ax_ffts[0,1].set_title('$B_{2,2}$')

ax_ffts[3,0].set_xlabel('$f$ [Hz]')
ax_ffts[3,1].set_xlabel('$f$ [Hz]')

[ax_ffts[i,0].set_ylabel('$T$ = '+str(int(np.flipud(np.sort(list(ls_periods.keys())))[i]))+' ms') for i in range(len(ls_periods))]

[a.plot([10**1,10**1.5],[10**2,10**(2-0.5*5./3)],color='gray',ls='--',alpha=0.5) for a in ax_ffts.flatten()]
[a.set_ylim([10**-1,10**3]) for a in ax_ffts.flatten()]
[a.set_xlim([10**-1,10**3]) for a in ax_ffts.flatten()]

fig_ffts.tight_layout()