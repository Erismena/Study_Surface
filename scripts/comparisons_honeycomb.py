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
import os

figfolder = r'C:\Users\Luc Deike\Documents\larger_tank_data_figures\\'

case_names = [r'piv_4pumps_stackedParallel_sunbathing_h093mm_L460mm_T10s_meanon050_meanoff450_fps4000',
              r'piv_4pumps_stackedParallel_sunbathing_h093mm_L460mm_T10s_meanon100_meanoff400_fps4000',
              r'piv_4pumps_stackedParallel_sunbathing_h093mm_L460mm_T10s_meanon200_meanoff300_fps4000',]
offsets = [(-550,-500)]*len(case_names)
parent_folders = [r'C:\Users\Luc Deike\data_comp3_C\180123\\']*2 + [r'C:\Users\Luc Deike\data_comp3_C\180124\\']*1
diffusers = [False] * len(case_names)
periods = [500]*len(case_names)
on_portions = [.1,.2,.4]

meta4=pd.DataFrame()
meta4['case_name'] = case_names
meta4['offset'] = offsets
meta4['parent_folder'] = parent_folders
meta4['diffuser'] = diffusers
meta4['period'] = periods
meta4['on_portion'] = on_portions
meta4['need2rotate'] = True
meta4['grid'] = False
meta4['description'] = None


meta3 = pd.DataFrame()

case_names = [r'piv_4pumps_stackedParallel_honeycomb1_sep6cm_sunbathing_meanon050_meanoff450_L450mm_h080mm_fps4000',
              r'piv_4pumps_stackedParallel_honeycomb1_sep6cm_sunbathing_meanon100_meanoff400_L450mm_h080mm_fps4000',
              r'piv_4pumps_stackedParallel_honeycomb1_sep12cm_sunbathing_meanon100_meanoff400_L450mm_h080mm_fps4000',
              r'piv_4pumps_stackedParallel_honeycomb1_sep12cm_sunbathing_meanon200_meanoff300_L450mm_h080mm_fps4000',]
offsets = [(-384,-450)]*len(case_names)
parent_folders = [r'D:\data_comp3_D\180208\\']*len(case_names)
periods = [500]*len(case_names)
on_portions = [.1,.2,.2,.4]

meta3['case_name'] = case_names
meta3['offset'] = offsets
meta3['parent_folder'] = parent_folders
meta3['diffuser'] = False
meta3['period'] = periods
meta3['on_portion'] = on_portions
meta3['need2rotate'] = False
meta3['grid'] = True
meta3['description'] = ['6 cm','6 cm','12 cm','12 cm']


meta = pd.concat([meta4,meta3])
meta.index = range(len(meta))


import matplotlib as mpl
import matplotlib.cm as cm

norm = mpl.colors.Normalize(vmin=meta['on_portion'].min(), vmax=meta['on_portion'].max())
cmap = cm.YlGnBu
m = cm.ScalarMappable(norm=norm, cmap=cmap)


c_onportions = {o:m.to_rgba(o) for o in meta['on_portion'].unique()} #,
ls_periods = {100:':',250:'-.',500:'--',0:'-'}
c_periods = {250:'r',500:'cyan'}

'''
Initialize the figures
'''

#fig_turb,axs_turb = plt.subplots(3,len(case_names),figsize=(11,8))

#Figs = piv.PIVComparisonsFigures(3,10,figsize=(17,10),max_speed=0.2,vmin=-.15,vmax=0.15,legend_axi=0)
Figs = piv.PIVComparisonsFigures(2,4,figsize=(10,4.5),max_speed=0.2,vmin=-.15,vmax=0.15,legend_axi=3)
fig_Lint0,axs_Lint0 = plt.subplots(2,4,figsize=(10,4.5),sharex=True,sharey=True); axs_Lint0=axs_Lint0.flatten()

skip_ax = []


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
    
    color = c_onportions[meta.loc[i,'on_portion']]
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
    center_rows,center_cols = g.get_coords(np.array([[0.05,-.05],[-.03,0.03]]))
    
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
    #Figs.update_limits(ai,[-0.1,0.05],[-0.12,0.1])
    Figs.update_limits(ai,[-0.13,0.1],[-0.12,0.1])
    Figs.add_text(ai,0.,.09,label)
    Figs.add_rect(ai,-.03,-.05,0.06,0.1,color=color,ls=ls,)
    
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
#        A.run_autocorr(ff,time,2000*8,coords_used=[center_rows[0],center_rows[1],center_cols[0],center_cols[1]])
    

    
    '''
    Some scalar data
    '''
    meta.loc[i,'u_rms'] = np.nanmean(u_rms[center_rows[0]:center_rows[1],center_cols[0]:center_cols[1]])
    meta.loc[i,'mean_speed'] = np.nanmean(meanflow_speed[center_rows[0]:center_rows[1],center_cols[0]:center_cols[1]])
    meta.loc[i,'epsilon_0'] = 1e-6 * 15 * np.nanmean((dudx/g.dx)**2)
    meta.loc[i,'epsilon_1'] = 1e-6 * 15 * np.nanmean((dvdy/g.dx)**2)
    
#    smaller_rows,smaller_cols = g.get_coords(np.array([[0.005,-.005],[-.005,0.005]]))
#    
#    center_row,center_col = g.get_coords(np.array([[0],[0]]))
#    A = spectra.AutocorrResults(g)
#    A.run_autocorr(ff,time,6000,[smaller_rows[0],smaller_rows[1],smaller_cols[0],smaller_cols[1]])
#    
#    C_dict[case_name] = A.C_avg
#    lags = A.lags
    
#    int_scales = []
#    for d in [0,1]:
#        try:
#            zero_crossing = np.argwhere(A.C_avg[:,d]<0)[0]
#        except:
#            zero_crossing = len(A.C_avg)
#        int_scales.append(np.cumsum(A.C_avg[:,d])[zero_crossing]*(time[1]-time[0]))
#        
#    meta.loc[i,'t_int_0'] = int_scales[0]
#    meta.loc[i,'t_int_1'] = int_scales[1]
    
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
    radial_vals.append((temporal_and_spatial_average,g_r,polar,r,line_avg,integral))
    axs_Lint0[ai].plot(r,integral[:,0],color='blue')
    axs_Lint0[ai].plot(r,integral[:,1],color='red')
    plt.show()
    plt.pause(1)
    
Figs.tight_layout()
Figs.add_legends()
[Figs.remove_axes(a) for a in skip_ax]

Figs.save_figs(figfolder,'honeycomb')

fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(111)

c_dict = {None:'k','6 cm':'b','12 cm':'r'}



c_var = 'description'
norm = mpl.colors.Normalize(vmin=meta['on_portion'].min(), vmax=meta['on_portion'].max())
cmap = cm.YlGnBu
m = cm.ScalarMappable(norm=norm, cmap=cmap)
c_dict = c_onportions = {o:m.to_rgba(o) for o in meta['on_portion'].unique()} #,


ax.plot([np.nan,np.nan],[np.nan,np.nan],'o-',color='k',label='$\epsilon_0$')
ax.plot([np.nan,np.nan],[np.nan,np.nan],'x--',color='k',label='$\epsilon_1$')
for key in np.sort(list(c_periods.keys())):
    color = c_periods[key]
    ax.plot([np.nan,np.nan],[np.nan,np.nan],'-',color=color,label='$\phi$ = 0.'+str(int(100.*key)))
ax.legend()

#for o in meta['period'].unique():
#    meta_on = meta[meta['period']==o]
#    ax.plot(meta_on['on_portion'],meta_on['t_int_0'],c=c_periods[o],ls='-')
#    ax.plot(meta_on['on_portion'],meta_on['t_int_1'],c=c_periods[o],ls='--')
    
ax.scatter(meta['on_portion'],meta['t_int_0'],c=[c_onportions[o] for o in meta['on_portion']],marker='o',s=20)
ax.scatter(meta['on_portion'],meta['t_int_1'],c=[c_onportions[o] for o in meta['on_portion']],marker='x',s=20)

ax.set_ylabel('$\epsilon$ [W/kg]')
ax.set_xlabel('T [ms]')



fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)

'''
Legend
'''
ax.plot([np.nan,np.nan],[np.nan,np.nan],'o--',color='k',label='$u_\mathrm{rms}$')
ax.plot([np.nan,np.nan],[np.nan,np.nan],'^-.',color='k',label='$|\overline{\mathbf{U}}|$')

c_grids = {0:'k',6:'g',12:'orange'}

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
fig.savefig(figfolder+'honeycomb_center_flow_comparisons.pdf')



'''
Integral length scale calculations
'''

fig,axs = plt.subplots(2,3,figsize=(8,4))
axs_dict = {phi:axs[:,pi] for pi,phi in enumerate(meta['on_portion'].unique())}
axst_dict = {key:[a.twinx() for a in axs_dict[key]] for key in list(axs_dict.keys())}
axst = np.array(list(axst_dict.values())).T

for i in meta.index:
    t = radial_vals[i]
    ax = axs_dict[meta.loc[i,'on_portion']]
    axt = axst_dict[meta.loc[i,'on_portion']]
    color = c_grids[meta.loc[i,'grid_dist']]

    meta_on = meta[meta['grid_dist']==o].sort_values('on_portion')
    
    for d in [0,1]:
    
        ax[d].plot(t[3],t[4][:,d]/t[4][0,d],c=color,ls='--')
        axt[d].plot(t[3],t[5][:,d],c=color,ls='-')
        
        ax[d].set_ylim([-0.2,1.2])
        axt[d].set_ylim([0,0.035])
        
        ax[d].set_xlim([0,.15])
        axt[d].set_xlim([0,0.15])
        
[ax.set_xticklabels([]) for ax in axs[0,:]]
[axt.set_xticklabels([]) for axt in axst[0,:]]
[ax.set_xlabel('r [m]') for ax in axs[1,:]]
#[axt.set_xticklabels([]) for axt in np.array(list(axst_dict.values())).T[:,:].flatten()]

[ax.set_yticklabels([]) for ax in axs[:,1:].flatten()]
axs[0,0].set_ylabel('$B_{00}$ [-]')
axs[1,0].set_ylabel('$B_{11}$ [-]')

[axt.set_yticklabels([]) for axt in np.array(list(axst_dict.values())).T[:,:-1].flatten()]
axst[0,-1].set_ylabel('$L_{00}$ [m]')
axst[1,-1].set_ylabel('$L_{11}$ [m]')
#[axt.yaxis.set_label_position("right") for axt in np.array(list(axst_dict.values())).T[:,-1]]

[axs_dict[key][0].set_title('$\phi = $'+str(key)) for key in list(axs_dict.keys())]

fig.tight_layout()
fig.savefig(figfolder+'honeycombs_spatial_scales.pdf')


'''
Plot the FFTs
'''
    
fig_ffts,ax_ffts = plt.subplots(4,2,figsize=(10,7),sharex=True,sharey=True)
#ax_ffts = ax_ffts.flatten()
ax_dict = {np.flipud(np.sort(list(ls_periods.keys())))[i]:ax_ffts[i] for i in range(len(ls_periods))}

fig_C,axs_C = plt.subplots(1,2); axs_C=axs_C.flatten()


for i in meta.index:
    
    parent_folder = meta.loc[i,'parent_folder']
    case_name = meta.loc[i,'case_name']
    offset = meta.loc[i,'offset']
    
    color = c_onportions[meta.loc[i,'on_portion']]
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