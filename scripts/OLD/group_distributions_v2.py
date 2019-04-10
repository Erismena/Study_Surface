# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 12:50:16 2017

@author: danjr
"""

import numpy as np
import matplotlib.pyplot as plt
import fluids2d.piv as piv
import pickle
import fluids2d.geometry
import pandas as pd

csv_filepath = r'C:\Users\user\Documents\2d-fluids-analysis\piv_data_organizer.csv'
all_cases = pd.read_csv(csv_filepath)

group_name = '170918-170921_3x3x10_withCap_midAndTopRegions'
ylims = [.15,.42]
all_cases = all_cases.loc[all_cases['use_now']==1]

vmin = -0.2
vmax = 0.2

amplitudes = all_cases['A'].unique()
freqs = all_cases['freq'].unique()

center_x = 0.12


fig=plt.figure(figsize=(13.5,8))
ax1=fig.add_subplot(111)
#ax2=fig.add_subplot(142)
#ax3=fig.add_subplot(143)
#ax4=fig.add_subplot(144)
plt.tight_layout()

colors = ['r','g','b','y','orange','cyan','purple','k','gray']

    
for ai,amp in enumerate(amplitudes):
    for fi,f in enumerate(freqs):
        
        

        fig_name = group_name + '_f'+str(int(f))+'Hz_A'+str(int(amp))+'mm'
        cases = all_cases.copy()[(all_cases['A']==amp) & (all_cases['freq']==f)]
        
        c = colors[int(ai*fi)]
        
        #ax5=fig.add_subplot(155,sharey=ax1)
        for case in cases.index:

            parent_folder = cases.loc[case,'parent_folder']+'\\'
            case_name = cases.loc[case,'file_name']
            origin_y = cases.loc[case,'origin_y']
            need2rotate=bool(cases.loc[case,'need2rotate'])
            
            p = pickle.load(open(parent_folder+case_name+'.pkl'))
            p.parent_folder = parent_folder
            p.associate_flowfield()
            
            if need2rotate:
                p.data.ff=piv.rotate_data_90(p.data.ff)    
            ff = p.data.ff
            
            ff = piv.clip_flowfield(p.data.ff,0.5) 
            g_orig = fluids2d.geometry.GeometryScaler(dx=p.dx,im_shape=(1,1),origin_pos=(origin_y,-0.06),origin_units='m')
            g = fluids2d.geometry.create_piv_scaler(p,g_orig)
            
            [_,col_lims] = g.get_coords([[0,0],[center_x-0.005,center_x+0.005]])
            
            mean_flow = np.nanmean(ff,axis=0)
            fluc = ff-mean_flow
            u_rms = np.sqrt( np.nanmean( (fluc[:,:,:,0])**2,axis=0) + np.nanmean( (fluc[:,:,:,1])**2,axis=0) )    
            inst_speed = np.linalg.norm(ff,ord=2,axis=3)
            
            ax1.plot(np.nanmean(u_rms[:,col_lims[0]:col_lims[1]],axis=1),g.y,color=c,alpha=1,label=case)
            
            #ax5.imshow(u_rms/np.nanmean(inst_speed,axis=0),vmin=0,vmax=3,extent=g[k].im_extent)
            
            '''
            [_,col_lims] = g[k].get_coords([[0,0],[-0.005,0.005]])
            other_percentiles = [10,25,75,90]
            ax5.plot(np.nanmean(np.nanpercentile(ff[:,:,col_lims[0]:col_lims[1],0],50,axis=0),axis=1),g[k].y,color='k',alpha=1)
            for other_percentile in other_percentiles:
                ax5.plot(np.nanmean(np.nanpercentile(ff[:,:,col_lims[0]:col_lims[1],0],other_percentile,axis=0),axis=1),g[k].y,color='k',alpha=0.5)
            '''

ax1.legend()