# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 10:09:51 2018
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pims
import scipy.ndimage
import pandas as pd
from scipy.ndimage.morphology import binary_fill_holes
import skimage.measure
import skimage.filters
import skimage.morphology
from matplotlib import cm
import trackpy as tp
import skimage
import cv2
from skimage.filters import threshold_local
import fluids2d.backlight as backlight
import fluids2d.geometry

thresh=480

folder = r'E:\Stephane\171114\\'
cine_names = [r'balloon_breakup_pumps_fps10000_backlight_D800minch_d20mm',
             r'balloon_breakup_pumps_fps10000_backlight_D400minch_d28mm']

methods = ['standard','watershed']
colors = [[1,0,0,0.5],[0,0,1,0.5]]
ls = ['-','--']

dx = 0.000112656993755
bins = np.geomspace(0.0001,0.02,41)
db = bins[1]-bins[0]
num_to_combine = 300

colors = ['r','b','g','orange','cyan','magenta']
'''
Make plots of the bubbes size distributions
'''
fig_dists,axs_dists = plt.subplots(1,2,sharex=True,sharey=True,figsize=(13,7)); axs_dists=np.atleast_2d(axs_dists)

for ci,cine_name in enumerate(cine_names):
    
    c = pims.open(folder+cine_name+'.cine')
    im = c[0]
    g = fluids2d.geometry.GeometryScaler(dx=dx,im_shape=np.shape(im),origin_pos=(0,0),origin_units='m')

    for mi,method in enumerate(methods):

        df_all = pd.read_pickle(folder+cine_name+r'_bubbles_'+method+'.pkl')
        frames = df_all['frame'].unique()
        
        fff = 0
        for fi in np.arange(0,len(frames),num_to_combine):
    
            df_frame_list = [ df_all[df_all['frame']==f] for f in frames[fi:fi+num_to_combine] ]
            df_frame = pd.concat(df_frame_list)
            
            df_frame = df_frame[(df_frame['x']>0.01)&(df_frame['x']<0.08)&(df_frame['y']<0.035)]
            
            counts =  np.zeros(len(bins)-1)
            for bi in np.arange(len(bins)-1):
                d = [r for r in df_frame['radius'] if (r>=bins[bi]) and (r<bins[bi+1])]
                counts[bi] = len(d)/float(num_to_combine)
            axs_dists[0,ci].loglog(bins[:-1],counts/(bins[1:]-bins[:-1]),ls=ls[mi],color=colors[fff])
            fff = fff+1

[axs_dists[0,ci].set_xlabel('bubble radius [m]') for ci in [0,1]]
[axs_dists[0,ci].set_title('Balloon '+str(ci)) for ci in [0,1]]
axs_dists[0,0].set_ylabel('bubble count [1/m]')
#axs_dists[1,0].set_ylabel('watershed method \n bubble count [1/m]')

axs_dists[0,0].plot(np.nan,np.nan,color='k',ls='-',label='standard method')
axs_dists[0,0].plot(np.nan,np.nan,color='k',ls='--',label='watershed method')
axs_dists[0,0].legend()

fig_dists.tight_layout()
fig_dists.savefig(folder+'distribution_comparisons.pdf')


stophere
'''
Make frames comparing the deteced bubbles
'''

for ci,cine_name in enumerate(cine_names):
    
    c = pims.open(folder+cine_name+'.cine')
    im = c[0]
    g = fluids2d.geometry.GeometryScaler(dx=dx,im_shape=np.shape(im),origin_pos=(0,0),origin_units='m')
    
    df_all = pd.read_pickle(folder+cine_name+r'_bubbles_'+method+'.pkl')
    frames = df_all['frame'].unique()
    
    figfolder = folder+'method_comparison_'+cine_name+'\\'
    
    for fi in np.arange(0,len(frames),num_to_combine):
        
        f = frames[fi]
        
        fig = plt.figure(figsize=(14,10))
        ax = fig.add_subplot(111)
        
        ax.imshow(c[f],extent=g.im_extent,vmin=0,vmax=600,cmap='gray')

        for mi,method in enumerate(methods):
    
            df_all = pd.read_pickle(folder+cine_name+r'_bubbles_'+method+'.pkl')
            df = df_all[df_all['frame']==f]
            
            for ix in df.index:
                e = Ellipse([df.loc[ix,'x'],df.loc[ix,'y']],width=df.loc[ix,'major_axis_length'],height=df.loc[ix,'minor_axis_length'],angle=df.loc[ix,'orientation'])
                ax.add_artist(e)
                e.set_facecolor('None')
                e.set_edgecolor(colors[mi])
            
        fig.tight_layout()
        fig.savefig(figfolder+'frame_'+str(int(f))+'.png')
        plt.close(fig)

stophere



ax.set_xlabel('bubble radius [m]')
ax.set_ylabel('bubble count per radius meter [1/m]')
figfolder = folder+'watershed_frames\\'

    
#    f = frames[fi]
#    fig_im = plt.figure(figsize=(12,8))
#    ax_im = fig_im.add_subplot(111)
#    ax_im.clear()
#    backlight.show_and_annotate(c[f],g,df_all[df_all['frame']==f],ax=ax_im,vmin=0,vmax=600)
#    fig_im.tight_layout()
#    fig_im.savefig(figfolder+'frame_'+str(int(f))+'.png')
#    plt.close(fig_im)