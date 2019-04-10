# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:15:16 2017

@author: danjr
"""

import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

traj_dir = r'C:\Users\danjr\Documents\Fluids Research\Data\misc\trajectories\\'

dx = 0.0000485712959867987 * 1000 # pixel width [mm]
dt = 0.001

trajs = []
metadata = []
for filename in os.listdir(traj_dir):
    print(filename)
    filepath = traj_dir + filename
    
    #f = open(filepath,'rb')
    #object_file = pickle.load(f)

    object_file = pd.read_pickle(filepath)
    
    # filter out the bubble trajectories that are too short
    object_file = {i:object_file[i] for i in list(object_file.keys()) if ( (len(object_file[i])>100) and (object_file[i]['filled_area'].mean()>100) and (abs(object_file[i]['x'].diff()).max()<50))}
    
    if True:
        trajs.append(object_file)
        m = {}
        
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
    
diffs = []
for run in trajs:
    run_diffs = {}
    for bubble_num in list(run.keys()):
        bubble = run[bubble_num]
        idx = bubble.index
        idx_diff = idx[1:] - idx[:-1]
        d = bubble.diff()
        for col in d.columns:
            d[col].iloc[1:] = d[col].iloc[1:] / idx_diff
        run_diffs[bubble_num] = d
    diffs.append(run_diffs)
            
'''
Get general bubble statistics
'''

bubble_stats = pd.DataFrame()
ix=0
c_mapping = {}

cmap = {0:'k',5:'b',10:'r'}

for ri,run in enumerate(trajs):
    run_diffs = diffs[ri]
    
    c_mapping[ri] = cmap[metadata[ri]['A']]
    
    # now have two dicts: run and run_diff
    for bubble_num in list(run.keys()):
        bubble = run[bubble_num]
        bubble_diff = run_diffs[bubble_num]
        
        bubble_stats.loc[ix,'f'] = metadata[ri]['f']
        bubble_stats.loc[ix,'A'] = metadata[ri]['A']
        
        bubble_stats.loc[ix,'x_start'] = bubble.loc[bubble['y'].argmax(),'x']  * dx
        bubble_stats.loc[ix,'mean_area'] = bubble['filled_area'].mean()  * dx**2
        bubble_stats.loc[ix,'median_vertical_speed'] = bubble_diff['y'].median() * dx / dt
        bubble_stats.loc[ix,'median_horz_speed'] = abs(bubble_diff['x']).median() * dx / dt
        bubble_stats.loc[ix,'mean_vertical_speed'] = bubble_diff['y'].mean() * dx / dt
        bubble_stats.loc[ix,'mean_horz_speed'] = abs(bubble_diff['x']).mean() * dx / dt
        bubble_stats.loc[ix,'std_horz_speed'] = (bubble_diff['x']).std() * dx / dt
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

'''
probability density for each condition
'''

cols_to_kde = ['x_start','mean_area','std_area','median_vertical_speed','median_horz_speed','horizontal_extent','mean_ecc','std_ecc','std_horz_speed']
fig,ax = plt.subplots(3,3)
for ci,col in enumerate(cols_to_kde):
    for A in list(cmap.keys()):
        series = bubble_stats[(bubble_stats['A']==A)][col]
        series.plot.density(ax=ax.flatten()[ci],color=cmap[A],label='A = '+format(A,'02')+' mm')
    ax.flatten()[ci].set_xlabel(col)
    ax.flatten()[ci].get_yaxis().set_visible(False)
ax.flatten()[0].legend()
  
'''
Plots of the trajectories
'''      

fig,ax = plt.subplots(1,3,sharex=True,sharey=True,figsize=(10,3))
c_to_subplot_mapping = {'k':0,'b':1,'r':2}
for ri,run in enumerate(trajs):
    for bubble_num in list(run.keys()):
        bubble = run[bubble_num]
        print(len(bubble))
        ax[c_to_subplot_mapping[c_mapping[ri]]].plot(bubble['x']*dx,bubble['y']*dx,color=c_mapping[ri],alpha=0.5)
        ax[c_to_subplot_mapping[c_mapping[ri]]].invert_yaxis()
        
[a.set(aspect=1) for a in ax]
[a.set(adjustable='box-forced') for a in ax]

ax[0].set_ylabel('y [mm]')
[a.set_xlabel('x [mm]',rotation=45) for a in ax]
ax[0].set_title('No induced turbulence')
ax[1].set_title('A = 5 mm, f = 5 Hz')
ax[2].set_title('A = 10 mm, f = 5 Hz')

plt.show()
plt.pause(0.1)