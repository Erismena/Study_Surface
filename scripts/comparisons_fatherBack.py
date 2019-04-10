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

#meta1 = pd.DataFrame()
#
#case_names = [r'piv_4pumps_stackedParallel_h093mm_L460mm_schedHalfOn_T0100ms_fps4000',
#              r'piv_4pumps_stackedParallel_h093mm_L460mm_schedTwentiethOn_T0250ms_fps4000',
#              r'piv_4pumps_stackedParallel_h093mm_L460mm_schedTenthOn_T0250ms_fps4000',
#              r'piv_4pumps_stackedParallel_h093mm_L460mm_schedFifthOn_T0250ms_fps4000',]              
#              
#offsets = [(-450,-700)]*len(case_names)
#parent_folders = [r'D:\data_comp3_D\180118\\']*len(case_names)
#diffusers = [False] * len(case_names)
#periods = [100,250,250,250]
#on_portions = [.5,.05,.1,.2]
#
#c_onportions = {.05:'purple',0.1:'b',0.15:'g',.2:'orange',0.25:'r',.3:'cyan',.5:'yellow'} #,
#ls_periods = {100:':',250:'-.',500:'--',0:'-'}
#
#c_periods = {100:'r',250:'cyan'}
#
#
#meta1['case_name'] = case_names
#meta1['offset'] = offsets
#meta1['parent_folder'] = parent_folders
#meta1['diffuser'] = diffusers
#meta1['period'] = periods
#meta1['on_portion'] = on_portions
#meta1['need2rotate'] = False
#
#meta2 = pd.DataFrame()
#
#case_names = [r'piv_4pumps_stackedParallel_sunbathing_h093mm_L460mm_T10s_meanon050_meanoff050_fps4000',
#              r'piv_4pumps_stackedParallel_sunbathing_h093mm_L460mm_T10s_meanon012p5_meanoff237p5_fps4000_good',
#              r'piv_4pumps_stackedParallel_sunbathing_h093mm_L460mm_T10s_meanon025_meanoff225_fps4000',
#              r'piv_4pumps_stackedParallel_sunbathing_h093mm_L460mm_T10s_meanon050_meanoff200_fps4000',]
#offsets = [(-500,-450)]*len(case_names)
#parent_folders = [r'D:\data_comp3_D\180121\\']*len(case_names)
#diffusers = [True] * len(case_names)
#periods = [100,250,250,250,]
#on_portions = [.5,.05,.1,.2]
#
#meta2['case_name'] = case_names
#meta2['offset'] = offsets
#meta2['parent_folder'] = parent_folders
#meta2['diffuser'] = diffusers
#meta2['period'] = periods
#meta2['on_portion'] = on_portions
#meta2['need2rotate'] = True

meta3 = pd.DataFrame()

case_names = [r'piv_4pumps_stackedParallel_sunbathing_h093mm_L460mm_T10s_meanon025_meanoff225_fps4000',
              r'piv_4pumps_stackedParallel_sunbathing_h093mm_L460mm_T10s_meanon037p5_meanoff212p5_fps4000',
              r'piv_4pumps_stackedParallel_sunbathing_h093mm_L460mm_T10s_meanon050_meanoff200_fps4000',
              r'piv_4pumps_stackedParallel_sunbathing_h093mm_L460mm_T10s_meanon062p5_meanoff187p5_fps4000',
              r'piv_4pumps_stackedParallel_sunbathing_h093mm_L460mm_T10s_meanon075_meanoff175_fps4000',
              r'piv_4pumps_stackedParallel_sunbathing_h093mm_L460mm_T10s_meanon087p5_meanoff162p5_fps4000',
              r'piv_4pumps_stackedParallel_sunbathing_h093mm_L460mm_T10s_meanon100_meanoff150_fps4000',
              r'piv_4pumps_stackedParallel_sunbathing_h093mm_L460mm_T10s_meanon112p5_meanoff137p5_fps4000',
              r'piv_4pumps_stackedParallel_sunbathing_h093mm_L460mm_T10s_meanon125_meanoff125_fps4000']
offsets = [(-550,-500)]*len(case_names)
parent_folders = [r'D:\data_comp3_D\180122\\']*5 + [r'D:\data_comp3_D\180125\\'] + [r'D:\data_comp3_D\180124\\'] + [r'D:\data_comp3_D\180125\\'] + [r'C:\Users\Luc Deike\data_comp3_C\180124\\']
diffusers = [False] * len(case_names)
periods = [250]*len(case_names)
on_portions = [.1,.15,.2,.25,.3,.35,.4,.45,.5]

meta3['case_name'] = case_names
meta3['offset'] = offsets
meta3['parent_folder'] = parent_folders
meta3['diffuser'] = diffusers
meta3['period'] = periods
meta3['on_portion'] = on_portions
meta3['need2rotate'] = True
meta3['grid'] = False
meta3['description'] = None


meta4 = pd.DataFrame()

case_names = [r'piv_4pumps_stackedParallel_sunbathing_h093mm_L460mm_T10s_meanon025_meanoff475_fps4000',
              r'piv_4pumps_stackedParallel_sunbathing_h093mm_L460mm_T10s_meanon050_meanoff450_fps4000',
              r'piv_4pumps_stackedParallel_sunbathing_h093mm_L460mm_T10s_meanon075_meanoff425_fps4000',
              r'piv_4pumps_stackedParallel_sunbathing_h093mm_L460mm_T10s_meanon100_meanoff400_fps4000',
              r'piv_4pumps_stackedParallel_sunbathing_h093mm_L460mm_T10s_meanon125_meanoff375_fps4000',
              r'piv_4pumps_stackedParallel_sunbathing_h093mm_L460mm_T10s_meanon150_meanoff350_fps4000',
              r'piv_4pumps_stackedParallel_sunbathing_h093mm_L460mm_T10s_meanon175_meanoff325_fps4000',
              r'piv_4pumps_stackedParallel_sunbathing_h093mm_L460mm_T10s_meanon200_meanoff300_fps4000',
              r'piv_4pumps_stackedParallel_sunbathing_h093mm_L460mm_T10s_meanon225_meanoff275_fps4000',
              r'piv_4pumps_stackedParallel_sunbathing_h093mm_L460mm_T10s_meanon250_meanoff250_fps4000']
offsets = [(-550,-500)]*len(case_names)
parent_folders = [r'C:\Users\Luc Deike\data_comp3_C\180123\\']*6 + [r'C:\Users\Luc Deike\data_comp3_C\180124\\']*4
diffusers = [False] * len(case_names)
periods = [500]*len(case_names)
on_portions = [.05,.1,.15,.2,.25,.3,.35,.4,.45,.5]

meta4['case_name'] = case_names
meta4['offset'] = offsets
meta4['parent_folder'] = parent_folders
meta4['diffuser'] = diffusers
meta4['period'] = periods
meta4['on_portion'] = on_portions
meta4['need2rotate'] = True
meta4['grid'] = False
meta4['description'] = None


#meta5 = pd.DataFrame()
#
#case_names = [r'piv_4pumps_showerhead_stackedParallel_sunbathing_h110mm_L425mm_T10s_meanon050_meanoff200_fps4000',
#              r'piv_4pumps_showerhead_stackedParallel_sunbathing_h110mm_L425mm_T10s_meanon100_meanoff400_fps4000',
#              r'piv_4pumps_showerhead_stackedParallel_sunbathing_h110mm_L425mm_T10s_meanon100_meanoff150_fps4000',              
#              r'piv_4pumps_showerhead_stackedParallel_sunbathing_h110mm_L425mm_T10s_meanon200_meanoff300_fps4000',
#              r'piv_4pumps_grid5cm_stackedParallel_sunbathing_h080mm_L450mm_T10s_meanon100_meanoff400_fps4000',
#              r'piv_4pumps_grid5cm_stackedParallel_sunbathing_h080mm_L450mm_T10s_meanon200_meanoff300_fps4000',
#              r'piv_4pumps_grid10cm_stackedParallel_sunbathing_h080mm_L450mm_T10s_meanon100_meanoff400_fps4000',
#              r'piv_4pumps_grid10cm_stackedParallel_sunbathing_h080mm_L450mm_T10s_meanon200_meanoff300_fps4000']
#offsets =  [(-400,-500)]*4 +  [(-500,-500)]*4
#parent_folders = [r'C:\Users\Luc Deike\data_comp3_C\180201\\']*4 + [r'D:\data_comp3_D\180202\\']*4
#diffusers = [True] * 4 + [False]*4
#periods = [250,500,250,500,500,500,500,500]
#on_portions = [.2,.2,.4,.4,.2,.4,.2,.4]
#grids = [False]*4 + [5]*2 + [10]*2
#
#meta5['case_name'] = case_names
#meta5['offset'] = offsets
#meta5['parent_folder'] = parent_folders
#meta5['diffuser'] = diffusers
#meta5['period'] = periods
#meta5['on_portion'] = on_portions
#meta5['need2rotate'] = False
#meta5['grid'] = grids
#meta5['description'] = ['showerhead']*4 + ['grids 5cm']*2 + ['grids 10cm']*2

meta = pd.concat([meta3,meta4])

meta.index = range(len(meta))
#skip_ax = [20,21,22,25,26,]

#meta = meta[meta['on_portion'].isin([.2,.25,.4,.45])]
#meta.index = range(len(meta))



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
Figs = piv.PIVComparisonsFigures(2,10,figsize=(14,4.5),max_speed=0.3,vmin=-.3,vmax=0.3,legend_axi=0)
fig_Lint0,axs_Lint0 = plt.subplots(2,10,figsize=(14,4.5),sharex=True,sharey=True); axs_Lint0=axs_Lint0.flatten()
#skip_ax = [9,14,19]
skip_ax=[]

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
    
    color = c_onportions[meta.loc[i,'on_portion']]
    ls = ls_periods[meta.loc[i,'period']]
    
    label = 'T = '+str(meta.loc[i,'period'])+' ms\n$\phi$ = '+str(int(100*meta.loc[i,'on_portion']))+'/100'
    if meta.loc[i,'description'] is not None:
        label = label+'\n'+meta.loc[i,'description']

    p = pickle.load(open(parent_folder+case_name+'.pkl'))
    p.parent_folder = parent_folder
    p.name_for_save = case_name
    p.associate_flowfield()   
    
    '''
    Check the concatenation
    '''
    print(parent_folder)    
    print(case_name)        
    try: 
        first_job_name = case_name+'_job0'
        p_0 = pickle.load(open(parent_folder+first_job_name+'.pkl'))
        p_0.parent_folder = parent_folder
        p_0.name_for_save = first_job_name
        p_0.associate_flowfield()           
        print((p.data.ff[0,0,0,0],p_0.data.ff[0,0,0,0]))        
    except:
        print('no job file found, correcting the scaling now')
        p.data.ff = p.data.ff / p.dx * p.dt_ab
    
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
    Figs.update_limits(ai,[-0.1,0.05],[-0.12,0.1])
    #Figs.update_limits(ai,[-0.1,0.15],[-0.12,0.1])
    Figs.add_text(ai,0.,.09,label)
    Figs.add_rect(ai,-.03,-.05,0.06,0.1,color=color,ls=ls,)
    
#    '''
#    FFTs
#    '''
#    #smaller_rows,smaller_cols = g.get_coords(np.array([[0.01,-.01],[-.01,0.01]]))
#    A = spectra.AutocorrResults(g,parent_folder=parent_folder,case_name=case_name)
#    
#    if os.path.isfile(A.filepath):
#        print('loading the existing file')
#        A_old = pickle.load(open(A.filepath,'rb'))
#        A = A_old
#    else:
#        A.run_autocorr(ff,time,2000*8,coords_used=[center_rows[0],center_rows[1],center_cols[0],center_cols[1]])
#    
#    C_dict[case_name] = A.C_avg
#    lags = A.lags
    
    '''
    Some scalar data
    '''
    meta.loc[i,'u_rms'] = np.nanmean(u_rms[center_rows[0]:center_rows[1],center_cols[0]:center_cols[1]])
    meta.loc[i,'mean_speed'] = np.nanmean(meanflow_speed[center_rows[0]:center_rows[1],center_cols[0]:center_cols[1]])
    meta.loc[i,'epsilon_0'] = 1e-6 * 15 * np.nanmean((dudx/g.dx)**2)
    meta.loc[i,'epsilon_1'] = 1e-6 * 15 * np.nanmean((dvdy/g.dx)**2)
    
    
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
    
    data_hist = ff[:,center_rows[0]:center_rows[1]+1,center_cols[0]:center_cols[1]+1,0].flatten()
    data_hist = data_hist[~np.isnan(data_hist)]
    
    ax_hist.hist(data_hist,bins=np.linspace(-0.5,0.5,1001),alpha=0.5)
    
#    int_scales = []
#    for d in [0,1]:
#        zero_crossing = np.argwhere(A.C_avg[:,d]<0)[0]
#        int_scales.append(np.cumsum(A.C_avg[:,d])[zero_crossing]*(time[1]-time[0]))
#        
#    meta.loc[i,'t_int_0'] = int_scales[0]
#    meta.loc[i,'t_int_1'] = int_scales[1]
    
Figs.tight_layout()
Figs.add_legends()
[Figs.remove_axes(a) for a in skip_ax]




#fig = plt.figure(figsize=(8,4))
#ax = fig.add_subplot(111)
#
#ax.plot([np.nan,np.nan],[np.nan,np.nan],'o-',color='k',label='$\epsilon_0$')
#ax.plot([np.nan,np.nan],[np.nan,np.nan],'x--',color='k',label='$\epsilon_1$')
#for key in np.sort(list(c_onportions.keys())):
#    color = c_onportions[key]
#    ax.plot([np.nan,np.nan],[np.nan,np.nan],'-',color=color,label='$\phi$ = 0.'+str(int(100.*key)))
#ax.legend()
#
#for o in meta['on_portion'].unique():
#    meta_on = meta[meta['on_portion']==o]
#    ax.plot(meta_on['period'],meta_on['t_int_0'],c=c_onportions[o],ls='-')
#    ax.plot(meta_on['period'],meta_on['t_int_1'],c=c_onportions[o],ls='--')
#    
#ax.scatter(meta['period'],meta['t_int_0'],c=[c_onportions[o] for o in meta['on_portion']],marker='o',s=20)
#ax.scatter(meta['period'],meta['t_int_1'],c=[c_onportions[o] for o in meta['on_portion']],marker='x',s=20)
#
#ax.set_ylabel('$\epsilon$ [W/kg]')
#ax.set_xlabel('T [ms]')

fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(111)

'''
Legend
'''
ax.plot([np.nan,np.nan],[np.nan,np.nan],'o--',color='k',label='$u_\mathrm{rms}$')
ax.plot([np.nan,np.nan],[np.nan,np.nan],'^-.',color='k',label='$|\overline{\mathbf{U}}|$')
for key in np.sort(list(c_periods.keys())):
    color = c_periods[key]
    #ax.plot([np.nan,np.nan],[np.nan,np.nan],'-',color=color,label='$\phi$ = '+str(int(100.*key))+'/100')
    ax.plot([np.nan,np.nan],[np.nan,np.nan],'-',color=color,label='$T$ = '+str(key)+' ms')
ax.legend()


for key in np.sort(list(c_periods.keys())):
    color = c_periods[key]
    x = meta[meta['period']==key]['on_portion']
    y = meta[meta['period']==key]['u_rms']
    ax.plot(x,y,'--',color=color,alpha=0.5)
    x = meta[meta['period']==key]['on_portion']
    y = meta[meta['period']==key]['mean_speed']
    ax.plot(x,y,'-.',color=color,alpha=0.5)
    
meta_diff = meta[meta['diffuser']==True]
ax.scatter(meta_diff['on_portion'],meta_diff['u_rms'],c='k',marker='s',s=50)
ax.scatter(meta_diff['on_portion'],meta_diff['mean_speed'],c='k',marker='s',s=50)

meta_grid = meta[meta['grid']==5]
ax.scatter(meta_grid['on_portion'],meta_grid['u_rms'],c='pink',marker='s',s=50)
ax.scatter(meta_grid['on_portion'],meta_grid['mean_speed'],c='pink',marker='s',s=50)

meta_grid = meta[meta['grid']==10]
ax.scatter(meta_grid['on_portion'],meta_grid['u_rms'],c='green',marker='s',s=50)
ax.scatter(meta_grid['on_portion'],meta_grid['mean_speed'],c='green',marker='s',s=50)

ax.scatter(meta['on_portion'],meta['u_rms'],c=[c_periods[o] for o in meta['period']],marker='o',s=20,edgecolor='k')
ax.scatter(meta['on_portion'],meta['mean_speed'],c=[c_periods[o] for o in meta['period']],marker='^',s=20,edgecolor='k')
#for i in meta.index:
    #ax.plot([meta.loc[i,'on_portion'],meta.loc[i,'on_portion']-(meta.loc[i,'u_rms']-meta.loc[i,'mean_speed'])/3,meta.loc[i,'on_portion']],[meta.loc[i,'u_rms'],(meta.loc[i,'u_rms']+meta.loc[i,'mean_speed'])/2,meta.loc[i,'mean_speed']],color='gray',lw=.5,alpha=0.5)

ax.set_xlabel('$\phi$ [-]')
ax.set_ylabel('speed [m/s]')
ax.set_ylim([0,0.5])

axt = ax.twinx()
for key in np.sort(list(c_periods.keys())):
    color = c_periods[key]
    x = meta[meta['period']==key]['on_portion']
    y = meta[meta['period']==key]['u_rms'] / meta[meta['period']==key]['mean_speed']
    axt.plot(x,y,'-',color=color)

axt.set_ylabel('$u_\mathrm{rms} / |\overline{\mathbf{U}}|$')
axt.set_ylim([0,4])

fig.tight_layout()
fig.savefig(figfolder+'sunbathing_sweep_center_flow_comparisons.pdf')

'''
Plot the correlations
'''

fig_B00,axs_B00 = plt.subplots(2,10,figsize=(14,4.5),sharex=True,sharey=True); axs_B00=axs_B00.flatten()
fig_B11,axs_B11 = plt.subplots(2,10,figsize=(14,4.5),sharex=True,sharey=True); axs_B11=axs_B11.flatten()
for i in meta.index:
    ai = i
    for si in [Figs.legend_axi] + skip_ax:
        if ai>=si:
            ai = ai+1            
    axs_B00[ai].imshow(radial_vals[i][0][:,:,0,0])
    axs_B11[ai].imshow(radial_vals[i][0][:,:,1,1])



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