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

group_name = ''
ylims = [.05,.3]
all_cases = all_cases.loc[all_cases['use_now']==1]

vmin = -0.05
vmax = 0.05

amplitudes = all_cases['A'].unique()
freqs = all_cases['freq'].unique()

fig=plt.figure(figsize=(13.5,8))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122,sharey=ax1)


ls_dict = {5:'--',10:'-'}
import matplotlib
norm = matplotlib.colors.Normalize(vmin=0,vmax=10)
c_dict = {A:matplotlib.cm.get_cmap('viridis')(norm(A)) for A in [1,2,3,4,5,6,7,8,9,10]}

left_lim = -.02
right_lim = 0.02

for amp in amplitudes:
    for f in freqs:
        
        cases = all_cases.copy()[(all_cases['A']==amp) & (all_cases['freq']==f)]
        
        print('-------------')
        print(amp,f)
        print(cases)

        u_rms_list = []
        v_mean_list = []
        g_list = []
        
        #ax5=fig.add_subplot(155,sharey=ax1)
        for case in cases.index:
            
            '''
            Load this case
            '''

            parent_folder = cases.loc[case,'parent_folder']+'\\'
            case_name = cases.loc[case,'file_name']
            origin_y = cases.loc[case,'origin_y']
            origin_x = cases.loc[case,'origin_x']
            need2rotate=bool(cases.loc[case,'need2rotate'])
            
            p = pickle.load(open(parent_folder+case_name+'.pkl'))
            p.parent_folder = parent_folder
            p.associate_flowfield()
            
            if need2rotate:
                p.data.ff=piv.rotate_data_90(p.data.ff)    
            ff = p.data.ff
            
            ff = piv.clip_flowfield(p.data.ff,0.5) 
            
            '''
            Create the object for scaling the geometry
            '''
            
            g_orig = fluids2d.geometry.GeometryScaler(dx=p.dx,im_shape=(1,1),origin_pos=(origin_y,origin_x),origin_units='m')
            g = fluids2d.geometry.create_piv_scaler(p,g_orig)
            g_list.append(g)
            
            # find the columns corresponding to the region of interest
            [_,left] = g.get_coords([np.nan,left_lim])
            [_,right] = g.get_coords([np.nan,right_lim])
            
            '''
            Extract and plot the desired data
            '''
            
            mean_flow = np.nanmean(ff,axis=0)
            fluc = ff-mean_flow
            u_rms = np.sqrt( np.nanmean( (fluc[:,:,:,0])**2,axis=0) + np.nanmean( (fluc[:,:,:,1])**2,axis=0) )    
            u_rms_list.append(np.nanmean(u_rms[:,left:right],axis=1))
            v_mean_list.append(np.nanmean(mean_flow[:,left:right,1],axis=1))            
            
            ax1.plot(pd.Series(np.nanmean(u_rms[:,left:right],axis=1)).rolling(window=3,center=True,min_periods=0).mean(),g.y,ls_dict[f],color=c_dict[amp],alpha=0.3)
            ax2.plot(pd.Series(np.nanmean(mean_flow[:,left:right,1],axis=1)).rolling(window=3,center=True,min_periods=0).mean(),g.y,ls_dict[f],color=c_dict[amp],alpha=0.3)
            
        if len(g_list)>0:
            int_y,int_urms = piv.combine_linear_data(u_rms_list,g_list)
            ax1.plot(int_urms,int_y,ls_dict[f],color=c_dict[amp],label='f = '+str(amp)+', A = '+str(f))
            
            int_y,int_vmean = piv.combine_linear_data(v_mean_list,g_list)
            ax2.plot(int_vmean,int_y,ls_dict[f],color=c_dict[amp])
        
ax1.legend()
ax1.set_ylabel('vertical position [m]')
ax1.set_xlabel('u_rms [m/s]')
ax2.set_xlabel('v mean [m/s]')        
        
'''
Make an image of the velocity fields for each case
'''
    
for amp in amplitudes:
    for f in freqs:
        
        fig=plt.figure(figsize=(13.5,8))
        ax1=fig.add_subplot(141)
        ax2=fig.add_subplot(142)
        ax3=fig.add_subplot(143)
        ax4=fig.add_subplot(144)
        plt.tight_layout()

        fig_name = group_name + '_f'+str(int(f))+'Hz_A'+str(int(amp))+'mm'
        cases = all_cases.copy()[(all_cases['A']==amp) & (all_cases['freq']==f)]
        
        #ax5=fig.add_subplot(155,sharey=ax1)
        for case in cases.index:

            parent_folder = cases.loc[case,'parent_folder']+'\\'
            case_name = cases.loc[case,'file_name']
            origin_y = cases.loc[case,'origin_y']
            origin_x = cases.loc[case,'origin_x']
            need2rotate=bool(cases.loc[case,'need2rotate'])
            
            p = pickle.load(open(parent_folder+case_name+'.pkl'))
            p.parent_folder = parent_folder
            p.associate_flowfield()
            
            if need2rotate:
                p.data.ff=piv.rotate_data_90(p.data.ff)    
            ff = p.data.ff
            
            ff = piv.clip_flowfield(p.data.ff,0.5) 
            g_orig = fluids2d.geometry.GeometryScaler(dx=p.dx,im_shape=(1,1),origin_pos=(origin_y,origin_x),origin_units='m')
            g = fluids2d.geometry.create_piv_scaler(p,g_orig)
            
            mean_flow = np.nanmean(ff,axis=0)
            fluc = ff-mean_flow
            u_rms = np.sqrt( np.nanmean( (fluc[:,:,:,0])**2,axis=0) + np.nanmean( (fluc[:,:,:,1])**2,axis=0) )    
            inst_speed = np.linalg.norm(ff,ord=2,axis=3)
            
            piv.add_fieldimg_to_ax(mean_flow[:,:,0],g,ax1,vel_dir='horizontal',vmin=vmin,vmax=vmax)
            piv.add_fieldimg_to_ax(mean_flow[:,:,1],g,ax2,vel_dir='vertical',vmin=vmin,vmax=vmax)
            piv.add_fieldimg_to_ax(np.log10(np.nanmean(inst_speed,axis=0)),g,ax3,vmin=-2,vmax=0)
                
            skip=(slice(None,None,4),slice(None,None,4))
            ax3.quiver(g.X[skip],g.Y[skip],mean_flow[:,:,0][skip],mean_flow[:,:,1][skip],color='white')
            
            ax4.imshow(np.log10(u_rms),vmin=-2,vmax=0,extent=g.im_extent)
            
            #ax5.imshow(u_rms/np.nanmean(inst_speed,axis=0),vmin=0,vmax=3,extent=g[k].im_extent)
            
            '''
            [_,col_lims] = g[k].get_coords([[0,0],[-0.005,0.005]])
            other_percentiles = [10,25,75,90]
            ax5.plot(np.nanmean(np.nanpercentile(ff[:,:,col_lims[0]:col_lims[1],0],50,axis=0),axis=1),g[k].y,color='k',alpha=1)
            for other_percentile in other_percentiles:
                ax5.plot(np.nanmean(np.nanpercentile(ff[:,:,col_lims[0]:col_lims[1],0],other_percentile,axis=0),axis=1),g[k].y,color='k',alpha=0.5)
            '''


        [a.set_ylim(ylims) for a in [ax1,ax2,ax3,ax4]]
        [a.yaxis.set_ticklabels([]) for a in [ax2,ax3,ax4]]
        #[a.set_xlim([-0.07,0.07]) for a in [ax1,ax2,ax3,ax4]]
        
        fig.suptitle(fig_name,y=0.98,fontsize=16)
        ax1.set_title(r'''$\overline{u}$''')
        ax2.set_title(r'''$\overline{v}$''')
        ax3.set_title(r'''$\overline{ \sqrt{u^2 + v^2}}$''')
        ax4.set_title(r'''$ \sqrt{\overline{u'^2} + \overline{v'^2}}$''')
        plt.subplots_adjust(top=0.9)
        
        #if cases.empty==False:
            #fig.savefig(r'E:\Experiments_Stephane\Grid column\PIV_measurements\summary_figures\\'+fig_name+r'_images.png')