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

#figfolder = r'E:\Stephane\20180108\figures\\'
figfolder = r'C:\Users\Luc Deike\Documents\dan_turbulence_project\figures\\'

meta = pd.DataFrame()

case_names = [r'piv_4pumps_back_sideView_sched-tenthOn_T0100ms_fps4000_dur4s',
              r'piv_4pumps_back_sideView_sched-tenthOn_T0250ms_fps4000_dur4s',
              r'piv_4pumps_back_sideView_sched-tenthOn_T0500ms_fps4000_dur4s',
              r'piv_4pumps_back_sideView_sched-fifthOn_T0100ms_fps4000_dur4s',
              r'piv_4pumps_back_sideView_sched-fifthOn_T0250ms_fps4000_dur4s',
              r'piv_4pumps_back_sideView_sched-fifthOn_T0500ms_fps4000_dur4s',]
offsets = [(-450,-700)]*len(case_names)
parent_folders = [r'C:\Users\Luc Deike\highspeed_data\180108\\']*len(case_names)
diffusers = [False] * len(case_names)
periods = [100,250,500,100,250,500,]
on_portions = [0.1,0.1,.1,0.2,0.2,.2,]

c_onportions = {0.1:'b',0.2:'g'} #,1:'cyan'
ls_periods = {100:':',250:'-.',500:'--',1000:'-'}


meta['case_name'] = case_names
meta['offset'] = offsets
meta['parent_folder'] = parent_folders
meta['diffuser'] = diffusers
meta['period'] = periods
meta['on_portion'] = on_portions


need2rotate = False

vmin = -.2
vmax = .2

'''
Initialize the figures
'''

Figs = piv.PIVComparisonsFigures(2,4,figsize=(11,5),max_speed=0.5,vmin=-.5,vmax=0.5,legend_axi=3)


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
    if i>2:
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
    
    '''
    Add to the figures
    '''
    Figs.add_case(ai,ff,g,time)
    Figs.add_text(ai,.07,.04,label)
    Figs.add_rect(ai,.1+-1*lim,-1*lim,lim*2,lim*2,color=color,ls=ls,)

    
    '''
    Some scalar data
    '''
    meta.loc[i,'u_rms'] = np.nanmean(u_rms[center_rows[0]:center_rows[1],center_cols[0]:center_cols[1]])
    meta.loc[i,'mean_speed'] = np.nanmean(meanflow_speed[center_rows[0]:center_rows[1],center_cols[0]:center_cols[1]])
    
    
Figs.tight_layout()
Figs.add_legends()
Figs.remove_axes(7)

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
fig.savefig(figfolder+'backSide_turb_avgs.pdf')