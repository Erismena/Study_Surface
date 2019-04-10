# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:15:16 2017

@author: danjr
"""

import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

figfolder = r'C:\Users\danjr\Documents\Fluids Research\Writing\grid_characterization\figures\\'

traj_dir = r'C:\Users\danjr\Documents\Fluids Research\Data\misc\trajectories_lower23s\\'

# calibration varies between the sets of videos
dx_dict = {'lower_23s':0.00016556200626314 * 1000} # pixel width [mm]
dt = 0.001
#dx_dict = {'center':1, 'lower':1}
#dt=1

trajs = []
metadata = []

'''
Read in each bubble, filter out bad ones, store metadata
'''
for region_dir in os.listdir(traj_dir):
    for filename in os.listdir(traj_dir+'\\'+region_dir):
        print(filename)
        filepath = traj_dir + region_dir + '\\' + filename
        
        #f = open(filepath,'rb')
        #object_file = pickle.load(f)
    
        object_file = pd.read_pickle(filepath)
        
        for o in list(object_file.keys()):
            new_x = object_file[o]['y'].copy()
            new_y = object_file[o]['x'].copy()            
            object_file[o]['x'] = new_x.copy()
            object_file[o]['y'] = 800-new_y.copy()
        
        # filter out the bubble trajectories that are too short
        object_file = {i:object_file[i] for i in list(object_file.keys()) if ( (len(object_file[i])>100) and (object_file[i]['filled_area'].mean()>100) and (abs(object_file[i]['x'].diff()).max()<20) and (abs(object_file[i]['y'].diff()).max()<20))}
        
        if True:
            trajs.append(object_file)
            m = {}
            
            
            
            if region_dir == 'center_third':
                m['region'] = 'center'
            elif region_dir == 'lower23s':
                m['region'] = 'lower_23s'
            
            if 'f00Hz' in filename:
                m['f'] = 0
            elif 'f05Hz' in filename:
                m['f'] = 5
                
            if 'A00mm' in filename:
                m['A'] = 0
            elif 'A05mm' in filename:
                m['A'] = 5
            elif 'A10mm' in filename:
                m['A'] = 10
                
            metadata.append(m)
    
'''
Create a list similar in structure to trajs, called diffs, that stores the 
time derivatives of the values in trajs
'''
diffs = []
for ri,run in enumerate(trajs):
    run_diffs = {}
    for bubble_num in list(run.keys()):
        bubble = run[bubble_num]
        idx = bubble.index
        idx_diff = idx[1:] - idx[:-1]
        d = bubble.diff()
        for col in d.columns:
            d[col].iloc[1:] = d[col].iloc[1:] / idx_diff
        run_diffs[bubble_num] = d
        
        for col in d.columns:
            col_diff_name = 'diff_'+col
            bubble[col_diff_name] = d[col]
        
        run[bubble_num] = bubble
    diffs.append(run_diffs)
    trajs[ri] = run
            
'''
Get general bubble statistics
'''
bubble_stats = pd.DataFrame()
ix=0
c_mapping = {}
ls_mapping = {}

# map amplitude to color
cmap = {0:'k',5:'b',10:'r'}

# map region to linestyle
lsmap = {'lower_23s':'-'}

# map region to marker style
msmap = {'lower_23s':'o'}

for ri,run in enumerate(trajs):
    run_diffs = diffs[ri]
    
    c_mapping[ri] = cmap[metadata[ri]['A']]
    ls_mapping[ri] = lsmap[metadata[ri]['region']]
    
    # now have two dicts: run and run_diff
    for bubble_num in list(run.keys()):
        
        bubble = run[bubble_num]
        bubble_diff = run_diffs[bubble_num]
        
        bubble_stats.loc[ix,'run_num'] = ri
        bubble_stats.loc[ix,'bubble_num'] = bubble_num
        
        bubble_stats.loc[ix,'f'] = metadata[ri]['f']
        bubble_stats.loc[ix,'A'] = metadata[ri]['A']
        bubble_stats.loc[ix,'region'] = metadata[ri]['region']
        
        dx = dx_dict[metadata[ri]['region']]
        
        bubble_stats.loc[ix,'x_start'] = bubble.loc[bubble['y'].argmax(),'x']  * dx
        bubble_stats.loc[ix,'y_start'] = bubble.loc[bubble['y'].argmax(),'y']  * dx
        bubble_stats.loc[ix,'mean_area'] = bubble['filled_area'].mean()  * dx**2
        bubble_stats.loc[ix,'median_vertical_speed'] = bubble_diff['y'].median() * dx / dt
        bubble_stats.loc[ix,'median_horz_speed'] = abs(bubble_diff['x']).median() * dx / dt
        bubble_stats.loc[ix,'mean_vertical_speed'] = bubble_diff['y'].mean() * dx / dt
        bubble_stats.loc[ix,'mean_horz_speed'] = abs(bubble_diff['x']).mean() * dx / dt
        bubble_stats.loc[ix,'std_horz_speed'] = (bubble_diff['x']).std() * dx / dt
        bubble_stats.loc[ix,'std_vertical_speed'] = (bubble_diff['y']).std() * dx / dt
        bubble_stats.loc[ix,'std_minor_axis_length'] = bubble['minor_axis_length'].std() * dx
        bubble_stats.loc[ix,'std_major_axis_length'] = bubble['major_axis_length'].std() * dx
        bubble_stats.loc[ix,'std_area'] = bubble['filled_area'].std() * dx**2
        bubble_stats.loc[ix,'mean_ecc'] = np.mean ( (bubble['major_axis_length']-bubble['minor_axis_length']) / ((bubble['major_axis_length']+bubble['minor_axis_length'])))
        bubble_stats.loc[ix,'std_ecc'] = np.std ( (bubble['major_axis_length']-bubble['minor_axis_length']) / ((bubble['major_axis_length']+bubble['minor_axis_length'])))
        
        # time to pass through interrogation window
        if (bubble['y'].max() >= 580) & (bubble['y'].min() <= 225):        
            frame_enter = bubble['y'][bubble['y']>225].argmin()
            frame_exit = bubble['y'][bubble['y']>580].argmax()
            bubble_stats.loc[ix,'residence_time'] = frame_enter-frame_exit
        else:
            bubble_stats.loc[ix,'residence_time'] = None
            
        # horizontal extent of the bubble
        bubble_stats.loc[ix,'horizontal_extent'] = (bubble['x'].max() - bubble['x'].min())  * dx
        
        ix = ix+1
        
'''
Scatter matrix
'''

cols_to_scatter = ['x_start','mean_area','median_vertical_speed','median_horz_speed','horizontal_extent','mean_ecc','std_ecc']
from pandas.plotting import scatter_matrix
scatter_matrix(bubble_stats[cols_to_scatter],color=[cmap[i] for i in bubble_stats['A']],alpha=0.8,diagonal='kde')
fig = plt.gcf()
fig.set_size_inches(13,11)
plt.tight_layout()
fig.savefig(figfolder+'lower23s_scatter_matrix.pdf')

'''
probability density for each condition - by bubble
'''

cols_to_kde = ['x_start','mean_area','std_area','median_horz_speed','std_horz_speed','horizontal_extent','median_vertical_speed','std_vertical_speed','std_minor_axis_length','mean_ecc','std_ecc','std_major_axis_length']
fig,ax = plt.subplots(4,3,figsize=(10,8))
for ci,col in enumerate(cols_to_kde):
    '''
    Work on each subplot separately.
    
    For each subplot, work on each line separately.
    '''
    for A in list(cmap.keys()):
        for region in list(lsmap.keys()):
            series = bubble_stats[(bubble_stats['A']==A)&(bubble_stats['region']==region)][col]
            if series.empty==False:
                series.plot.kde(ax=ax.flatten()[ci],color=cmap[A],style=lsmap[region],label='A = '+format(A,'02')+' mm')
    ax.flatten()[ci].set_title(col)
    ax.flatten()[ci].get_yaxis().set_visible(False)
#ax.flatten()[0].legend()
plt.tight_layout()
fig.savefig(figfolder+'lower23s_kde_plots.pdf')
  
'''
Plots of the trajectories
'''

fig,ax = plt.subplots(1,2,sharex=True,sharey=True,figsize=(10,7))
c_to_subplotcol_mapping = {'b':0,'r':1}
for ri,run in enumerate(trajs):
    for bubble_num in list(run.keys()):
        bubble = run[bubble_num]
        ax[c_to_subplotcol_mapping[c_mapping[ri]]].plot(bubble['x']*dx,bubble['y']*dx,color=c_mapping[ri],linestyle=ls_mapping[ri],alpha=0.5)
        ax[c_to_subplotcol_mapping[c_mapping[ri]]].invert_yaxis()
        
[a.set(aspect=1) for a in ax.flatten()]
[a.set(adjustable='box-forced') for a in ax.flatten()]

ax[0].set_ylabel('y [mm]')
[a.set_xlabel('x [mm]') for a in ax]
ax[0].set_title('A = 5 mm, f = 5 Hz')
ax[1].set_title('A = 10 mm, f = 5 Hz')

plt.show()
plt.pause(0.1)
fig.savefig(figfolder+'lower23s_trajectories.pdf')

#'''
#random kde
#'''
#
#fig=plt.figure()
#ax=fig.add_subplot(111)
#for ri,run in enumerate(trajs):
#    run_diff = diffs[ri]
#    for bubble_num in list(run.keys()):
#        bubble = run[bubble_num][run[bubble_num]['minor_axis_length']>0]
#        #bubble_diff = run_diff[bubble_num]#.rolling(window=3,center=True,min_periods=0).mean()
#        (bubble['major_axis_length']/bubble['minor_axis_length']).plot.kde(ax=ax,alpha=0.2,color=c_mapping[ri],linestyle=ls_mapping[ri])

'''
Time series of an individual bubble
'''
fig,ax = plt.subplots(4,1,sharex=True,figsize=(5,9))
run_num=8
bubble_num=0
bubble = trajs[run_num][list(trajs[run_num].keys())[bubble_num]].iloc[10:-10]
bubble_diff = diffs[run_num][list(trajs[run_num].keys())[bubble_num]].iloc[10:-10]

ax[0].plot(bubble.index*dt,bubble['x']*dx)
ax[0].set_ylabel('x position [mm]')
ax[1].plot(bubble_diff.index*dt,np.sqrt(bubble_diff['x']**2+bubble_diff['y']**2)*dx/dt)
ax[1].plot(bubble_diff.index*dt,bubble_diff['x']*dx/dt)
ax[1].plot(bubble_diff.index*dt,bubble_diff['y']*dx/dt)
ax[1].set_ylabel('velocity [mm/s]')
ax[2].plot(bubble.index*dt,bubble['orientation'])
ax[2].set_ylabel('orientation [rad]')
ax[3].plot(bubble.index*dt,bubble['major_axis_length']*dx)
ax[3].plot(bubble.index*dt,bubble['minor_axis_length']*dx)
ax[3].set_ylabel('axis lengths [mm]')
plt.tight_layout()

'''
Spectral analysis
'''

fig=plt.figure(); ax=fig.add_subplot(111); ax2=ax.twinx()
freq=np.fft.fftfreq(len(bubble),dt)
f = np.fft.fft(bubble['minor_axis_length'].rolling(window=3,center=True,min_periods=0).mean()*dx/dt)

ax.semilogy(freq[freq>0],np.absolute(f[freq>0]),color='k')
ax2.plot(freq[freq>0],np.cos(np.angle(f[freq>0])),color='r',alpha=0.3)

'''
KDE by timepoint
'''
cols_to_kde = ['filled_area','major_axis_length','x','y','diff_y','diff_x','orientation',]
fig,ax = plt.subplots(4,3,figsize=(10,8))
for A in list(cmap.keys()):
    for region in list(lsmap.keys()):
        
        '''
        Concatenate all the bubble timepoints at this condition
        '''
        bubbles_to_use = bubble_stats.copy()[(bubble_stats['A']==A)&(bubble_stats['region']==region)]
        df = pd.DataFrame(columns=cols_to_kde)
        for ix in bubbles_to_use.index:
            run_num = int(bubbles_to_use.loc[ix,'run_num'])
            bubble_num = int(bubbles_to_use.loc[ix,'bubble_num'])        
            df = pd.concat([df,trajs[run_num][bubble_num]])
            
        # remove outliers
        #from scipy import stats
        #df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
        if df.empty==False:
            df = df[np.abs(df-df.mean())<=(3*df.std())]
        
            
        '''
        Plot the pdf of all of these time points
        '''
        for ci,col in enumerate(cols_to_kde):
            series=df[col]
            if series.empty == False:
                series.plot.kde(ax=ax.flatten()[ci],color=cmap[A],style=lsmap[region])            
            ax.flatten()[ci].set_title(col)
            ax.flatten()[ci].get_yaxis().set_visible(False)
#ax.flatten()[0].legend()
plt.tight_layout()
#fig.savefig(figfolder+'lower23s_kde_plots.pdf')