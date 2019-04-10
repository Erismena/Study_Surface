# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:02:30 2018

@author: Luc Deike
"""

import numpy as np
import matplotlib.pyplot as plt
import pims
import fluids2d.backlight as backlight
import trackpy as tp
import pandas as pd
import scipy.ndimage.filters

parent_folder = r'D:\data_comp3_D\180129\\'
cine_name = r'dual_view_Cam_20861_Cine3'

c = pims.open(parent_folder+cine_name+'.cine')

bg = c[0].astype(int)

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

#top_lims = [250,430,290,455]
#side_lims = [755,1213,141,606]

top_lims = [165,342,305,480]
side_lims = [755,1213,160,625]

all_top = pd.DataFrame()
all_side = pd.DataFrame()

frames = range(2000,6226,2)

for fi,f in enumerate(frames):
    
    print(f)
    
    im = np.abs(c[f].astype(int) - bg)
    
    im = scipy.ndimage.filters.median_filter(im,size=3)
    im = im>50
        
    top = im[top_lims[2]:top_lims[3],top_lims[0]:top_lims[1]]
    side = im[side_lims[2]:side_lims[3],side_lims[0]:side_lims[1]]

    df_top = tp.locate(top,21)
    df_side = tp.locate(side,21)
    
    df_top['frame'] = f
    df_side['frame'] = f
    
    all_top = pd.concat([all_top,df_top])
    all_side = pd.concat([all_side,df_side])
    
#    im = np.abs(c[f].astype(int) - bg)
#    
#    im = scipy.ndimage.filters.median_filter(im,size=3)
#        
#    top = im[top_lims[2]:top_lims[3],top_lims[0]:top_lims[1]]
#    side = im[side_lims[2]:side_lims[3],side_lims[0]:side_lims[1]]
##
#    ax1.clear()
#    ax1.imshow(top)
#    ax2.clear()
#    ax2.imshow(side)
#    for ix in df_top.index:
#        ax1.plot(df_top.loc[ix,'x'],df_top.loc[ix,'y'],'o',alpha=0.5)
#        
#    for ix in df_side.index:
#        ax2.plot(df_side.loc[ix,'x'],df_side.loc[ix,'y'],'o',alpha=0.5)
#        
#    plt.show()
#    plt.pause(0.5)
        
'''
Perform the tracking in each view, separately
'''
t_top = tp.link_df(all_top,11,memory=8)
t_side = tp.link_df(all_side,11,memory=8)

top_traj_x = pd.DataFrame(index=frames)
top_traj_y = pd.DataFrame(index=frames)
side_traj_x = pd.DataFrame(index=frames)
side_traj_y = pd.DataFrame(index=frames)

meta_df = pd.DataFrame()
cdict = {'top':'r','side':'b'}

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

[a.set_aspect('equal') for a in [ax1,ax2]]

for p in t_top['particle'].unique():
    
    t = t_top[t_top['particle']==p].set_index(['frame'])
    if len(t)>3:
        #ax1.plot(t['x'],t['y'],c='gray',lw=1,alpha=0.5)
        #ax1.scatter(t['x'],t['y'],c=[[0.5,0.5,(f-700.)/2000.] for f in t.index])
        
        top_traj_x[p] = t['x']
        top_traj_y[p] = t['y']
    
for p in t_side['particle'].unique():
    t = t_side[t_side['particle']==p].set_index(['frame'])
    if len(t)>3:
        #ax2.plot(t['x'],t['y'],c='gray',lw=1,alpha=0.5)
        #ax2.scatter(t['x'],t['y'],c=[[0.5,0.5,(f-700.)/2000.] for f in t.index])
        
        side_traj_x[p] = t['x']
        side_traj_y[p] = t['y']
        


fig,axs = plt.subplots(2,1,figsize=(5,6),sharex=True); axs=axs.flatten()

[axs[0].plot(top_traj_y[t],label=t) for t in top_traj_y.columns]
[axs[1].plot(side_traj_y[s],label=s) for s in side_traj_y.columns]

axs[0].set_ylabel('y (top view)')
axs[1].set_ylabel('y (side view)')

#axs[0].legend()
#axs[1].legend()

'''
Correlate the y positions of trajectories in each view
'''

df_corr = pd.DataFrame(index=top_traj_y.columns,columns=side_traj_y.columns)
for t in top_traj_y.columns:    
    for s in side_traj_y.columns:
        df_corr.loc[t,s] = top_traj_y[t].corr(side_traj_y[s])
df_corr[df_corr<0.85] = np.nan

'''
Decide which top view trajs are associated with each side view traj at each  
point in time
'''
top_for_side = pd.DataFrame(index=side_traj_y.index,columns=side_traj_y.columns)
for s in top_for_side.columns:
    for i in top_for_side.index[pd.notnull(side_traj_y[s])]:
        
        '''
        available top view trajectories are those which are not null at this 
        point in time
        '''
        available_tops = top_traj_y.columns[pd.notnull(top_traj_y.loc[i,:])]
        
        '''
        which top view trajectory should be used at this point for this side
        view trajecotry
        '''
        if len(df_corr.loc[available_tops,s]) > 0:
            top_for_side.loc[i,s] = df_corr.loc[available_tops,s].argmax()
            
'''
Draw on the 2d y trajectories the corresponding regions
'''
from matplotlib.patches import ConnectionPatch
for s in top_for_side.columns:
    '''
    each column in top_for_side corresponds to a side-view y trajectory
    '''
    
    #top_for_side[~pd.notnull(top_for_side)] = -1
    top_traj_diff = top_for_side.copy().loc[:,s].fillna(-1).diff()
    #top_for_side[top_for_side==-1] = None
    
    top_traj_diff = top_traj_diff[pd.notnull(top_traj_diff)]
    switch_points = top_traj_diff.index[top_traj_diff!=0]
    switch_points_before = switch_points-2
    #switch_points = list(switch_points) + [sp-2 for sp in list(switch_points)]
    print(switch_points)
    
    for ix in switch_points:
        try:
            xy_top = (ix,top_traj_y.loc[ix,top_for_side.loc[ix,s]])
            xy_side = (ix,side_traj_y.loc[ix,s])
            print(xy_top)
            print(xy_side)
            con = ConnectionPatch(xyA=xy_side, xyB=xy_top, coordsA="data", coordsB="data",
                                  axesA=axs[1], axesB=axs[0], color='g',alpha=0.5)
            axs[1].add_artist(con)
        except:
            pass
        
    for ix in switch_points_before:
        try:
            xy_top = (ix,top_traj_y.loc[ix,top_for_side.loc[ix,s]])
            xy_side = (ix,side_traj_y.loc[ix,s])
            print(xy_top)
            print(xy_side)
            con = ConnectionPatch(xyA=xy_side, xyB=xy_top, coordsA="data", coordsB="data",
                                  axesA=axs[1], axesB=axs[0], color='r',alpha=0.5)
            axs[1].add_artist(con)
        except:
            pass
        
        #axs[0].plot(xy_top[0],xy_top[1],'x')
        #axs[1].plot(xy_side[0],xy_side[1],'x')

'''
Create the 3d trajectories
'''
t3d_dict = {}
for s in top_for_side.columns:
    if len(top_for_side[s][pd.notnull(top_for_side[s])]) > 10:
        df = pd.DataFrame()
        df['y'] = side_traj_y.loc[ top_traj_x.index[pd.notnull(top_for_side[s])], s ]
        df['z'] = side_traj_x.loc[ top_traj_x.index[pd.notnull(top_for_side[s])], s ]
        df['x'] = [top_traj_x.loc[i,top_for_side.loc[i,s]] for i in top_traj_x.index[pd.notnull(top_for_side[s])]]
        
        t3d_dict[s] = df
        


fig = plt.figure()
from mpl_toolkits.mplot3d import Axes3D
ax = fig.add_subplot(111,projection='3d')

import matplotlib
cmap = plt.cm.viridis
norm = matplotlib.colors.Normalize(vmin=min(frames), vmax=max(frames))

for df in list(t3d_dict.values()):
    ax.plot(df['x'],df['y'],zs=df['z']*-1,lw=1,color='gray',alpha=0.5)
    ax.scatter(df['x'],df['y'],zs=df['z']*-1, color=cmap(norm(df.index)))