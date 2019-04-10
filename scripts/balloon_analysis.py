# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 19:03:44 2017

@author: danjr
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import fluids2d.geometry
import fluids2d.backlight
import pims
import scipy.ndimage
import pandas as pd
from scipy.ndimage.morphology import binary_fill_holes
import skimage.measure
import skimage.filters
from matplotlib import cm

if False:
    folder = r'E:\Stephane\171114\\'
    
    cine_names = [#r'balloon_breakup_nopumps_fps10000_backlight_D400minch_d30mm',
                  #r'balloon_breakup_nopumps_fps10000_backlight_D800minch_d20mm',]
                  r'balloon_breakup_pumps_fps10000_backlight_D400minch_d28mm',
                  r'balloon_breakup_pumps_fps10000_backlight_D800minch_d20mm']
    
    cine_list = [pims.open(folder+cine_name+'.cine') for cine_name in cine_names]
    
    titles = [r'D = 0.4 in, d = 28 mm, 4 pumps',
              r'D = 0.8 in, d = 20 mm, 4 pumps']
    
    frame_start = [0,0,0]
    cmaps = ['Greens','Purples','Oranges']
    cs = ['g','purple','orange']
    threshs = [500,500,400]
    
    dt = 1./10000
    #dx = 0.000124996484523 # from 17/11/09
    dx = 0.000113764988658 # from 17/11/15
    g = fluids2d.geometry.GeometryScaler(dx=dx,im_shape=(1024,800),origin_pos=(0,0),origin_units='m')
    
if False:
    
    '''
    Grid turbulence data
    '''
    
    folder = r'E:\Stephane\20171028\bubble_cloud\\'
    cine_names = [r'backlight_sv_grid3tx2x20_fps2000_cloud_25mL_fast_noTurbulence_a',
                  r'backlight_sv_grid3tx2x20_fps2000_cloud_25mL_fast_f10Hz_A4mm_a',
                  r'backlight_sv_grid3tx2x20_fps2000_cloud_25mL_fast_f5Hz_A10mm_a']
    
    cine_list = [pims.open(folder+cine_name+'.cine') for cine_name in cine_names]

    titles = ['no turbulence','10 Hz, 4 mm','5 Hz, 10 mm']
    
    frame_start = [0,0,0]   
    cmaps = ['Greens','Purples','Oranges']
    cs = ['g','purple','orange']
    threshs = [-50,-50,-50] 
    
    dt = 1./2000    
    dx=0.000126336189127
    g = fluids2d.geometry.GeometryScaler(dx=dx,im_shape=(1280,800),origin_pos=(0.096,0),origin_units='m')
    
if False:
    
    folder = r'E:\Stephane\20171109\\'
    cine_names = [r'balloon_breakup_4pumps_fps10000_backlight_D400minch',
                  r'balloon_breakup_nopumps_fps10000_backlight_D400minch',
                  r'balloon_breakup_nopumps_fps10000_backlight_D800minch']
    
    cine_list = [pims.open(folder+cine_name+'.cine') for cine_name in cine_names]

    titles = ['4 pumps, .4 in','no pumps, .4 in','no pumps, .8 in']
    
    frame_start = [0,0,0]   
    cmaps = ['Greens','Purples','Oranges']
    cs = ['g','purple','orange']
    threshs = [500,500,425] 
    
    dt = 1./10000    
    dx= 0.000124996484523
    g = fluids2d.geometry.GeometryScaler(dx=dx,im_shape=(1024,800),origin_pos=(0,0),origin_units='m')
    
if True:
    
    folder = r'D:\high_speed_data\171103\\'
    cine_names = [r'Balloon_backlight_fps2000_4pump12V_ceiling_2_breaking_500ms_end',]
    
    cine_list = [pims.open(folder+cine_name+'.cine') for cine_name in cine_names]

    titles = ['Balloon_backlight_fps2000_4pump12V_ceiling_2_breaking_500ms_end']
    
    frame_start = [0,0,0]   
    cmaps = ['Greens','Purples','Oranges']
    cs = ['g','purple','orange']
    threshs = [450] 
    
    dt = 1./2000    
    dx= 0.000130717383739
    g = fluids2d.geometry.GeometryScaler(dx=dx,im_shape=(800,1280),origin_pos=(0,0),origin_units='m')
    
    def mask(im):
        im = im.astype(float)
        im[:,0:350] = 1000
        im[:,1060:] = 1000
        im[675:,:] = 1000
        #im = skimage.filters.median(im,3)
        #im = np.rot90(im,k=1)
        return im

#def mask(im):
#    im = im.astype(float)
#    #im = np.rot90(im,k=3)
#    im[0:20,:] = 1000
#    im[:,0:15] = 1000
#    #im = np.rot90(im,k=1)
#    return im

def get_filled(im,thresh):
    im_filt = scipy.ndimage.filters.median_filter(im,size=2)
    im_filt = fluids2d.backlight.binarize(im_filt,thresh,large_true=False)
    filled = binary_fill_holes(im_filt)
    return filled

def filled_props(filled,g):
    
    objects, num_objects = scipy.ndimage.label(filled)
    props = skimage.measure.regionprops(objects)
    
    df = pd.DataFrame(columns=['frame','radius','x','y'])
    if (len(props)==0)==False:
        for ri,region in enumerate(props):
            df.loc[ri,'y'],df.loc[ri,'x'] = g.get_loc([region.centroid[0],region.centroid[1]])
            df.loc[ri,'radius'] = np.sqrt(region.filled_area * dx**2 / np.pi)
            df.loc[ri,'orientation'] = region.orientation / (2*np.pi) * 360
            df.loc[ri,'major_axis_length'] = region.major_axis_length* dx
            df.loc[ri,'minor_axis_length'] = region.minor_axis_length* dx
            
    return df




frames = np.arange(0,23000,2)

len_cine = len(frames)
#time = np.arange(len_cine) * 1./10000.

#composite_img0 = [np.zeros((np.shape(mask(c[0]))[0],len_cine)) for c in cine_list]
#composite_img1 = [np.zeros((np.shape(mask(c[0]))[1],len_cine)) for c in cine_list]

#bgs = [mask(c[0]) for c in cine_list]

df_all = [pd.DataFrame() for _ in cine_list]

bins = np.logspace(-4,-2,41)
frame_sep = 100
for i,f0 in enumerate(frames):
    
    print(f0)
    
    if True:
        
        fig,axes=plt.subplots(2,len(cine_list),figsize=(12,9))
        axes = axes.flatten()
        ax_hist = fig.add_subplot(2,1,2)    
        #[axes[ai].set_visible(False) for ai in (3,4,5)]
        [axes[ai].set_visible(False) for ai in [1]]
        #[axes[ai].set_visible(False) for ai in (2,3)]
    
    for ci,c in enumerate(cine_list):
        
        f = f0+frame_start[ci]
        im = mask(c[f]) #- bgs[ci]
        
        if False:
            
            composite_img0[ci][:,i] = np.mean(np.abs(im),axis=1)
            composite_img1[ci][:,i] = np.mean(np.abs(im),axis=0)
        
        if True:
            
            filled = get_filled(im,threshs[ci])
            
            df = filled_props(filled,g)
            df['frame'] = f0
            df = df[df['radius']>0.0001]
            df = df[df['radius']<0.05] 
            #df = df[df['y']>0.025]
            #df = df[df['y']<0.112]
            #df = df[df['y']<0.245]
            df_all[ci] = pd.concat([df_all[ci],df])
            
        if True:
            ax = axes[ci]
    
            ax.clear()
            ax.imshow(im,cmap='gray',vmin=0,vmax=600,extent=g.im_extent)
            ax.imshow(fluids2d.backlight.alpha_binary_cmap(filled.astype(float),cm.get_cmap(cmaps[ci])),extent=g.im_extent)
            
            for ix in df.index:
                e = Ellipse([df.loc[ix,'x'],df.loc[ix,'y']],width=df.loc[ix,'major_axis_length'],height=df.loc[ix,'minor_axis_length'],angle=df.loc[ix,'orientation'])
                ax.add_artist(e)
                e.set_facecolor('None')
                e.set_edgecolor([1,0,0,0.5])
            
            ax.set_title(titles[ci])
            #ax.scatter(df['x'],df['y'],alpha=0.5,color='red')
            
        if True:
            d = df_all[ci].copy()
            d = d[d['frame']>=f0-frame_sep]
            d = d[d['frame']<=f0]
            #ax.hist(d['radius'],bins=np.logspace(1, 4, 50),alpha=0.3)
            #d['radius'].plot(ax=ax,kind='kde')
            
            counts = [len(d[(d['radius']>=bins[ii])&(d['radius']<bins[ii+1])]) for ii in range(len(bins)-1)]
            counts = np.array(counts).astype(float)
            
            counts_norm = counts / (bins[1:] - bins[0:-1])
            
            ax_hist.plot(bins[0:-1],counts_norm/float(len(np.unique(d['frame']))),'.-',color=cs[ci])
            ax_hist.set_xscale('log')
            ax_hist.set_yscale('log')
            ax_hist.set_ylim([10,10**7])
            ax_hist.set_xlim([min(bins),max(bins)])
            ax_hist.set_ylabel('counts/m')
            ax_hist.set_xlabel('bubble radius [m]')
            ax_hist.set_title('size distribution, averaged over past 10 ms')
       
    #plt.show()
    #plt.pause(3)
    
    if True:
        
        fig.suptitle('t = '+str(float(f0)*dt*1000)+' ms')
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        fig.savefig(folder+r'\Balloon_backlight_fps2000_4pump12V_ceiling_2_breaking_500ms_end_frames\\frame_'+str(f0)+'.png')
        plt.close('all')
        
import pickle
for ci,c in enumerate(cine_list):
    
    d = df_all[ci]
    d['time'] = d['frame']*dt
    df_all[ci].to_pickle(folder+cine_names[ci]+r'_bubbles.pkl')
    with open(folder+cine_names[ci]+r'_bubbles.pkl', 'wb') as handle:
        pickle.dump(df_all[ci], handle)

        
df_filtered = df_all[df_all['radius']>5]
df_filtered = df_filtered[df_filtered['radius']<10000]
fig = plt.figure()
ax = fig.add_subplot(111)


frame_sep = 1000
for fi in range(min(df_filtered['frame']),max(df_filtered['frame']),frame_sep):
    d = df_filtered
    d = d[d['frame']>=fi]
    d = d[d['frame']<fi+frame_sep]
    #ax.hist(d['radius'],bins=np.logspace(1, 4, 50),alpha=0.3)
    #d['radius'].plot(ax=ax,kind='kde')
    
    counts = [len(d[(d['radius']>=bins[i])&(d['radius']<bins[i+1])]) for i in range(len(bins)-1)]
    counts = np.array(counts).astype(float)
    
    ax.plot(bins[0:-1],counts/np.sum(counts),'.-',color=[0.5+0.5*float(fi)/max(df_filtered['frame']),0,float(fi)/max(df_filtered['frame'])])
    
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([min(bins),max(bins)])
    #ax.set_ylim([0.0001,1])
    
    #plt.show()
    #plt.pause(0.5)
    #ax.clear()

#ax.set_ylim([0.0000001,0.01])
#ax.set_yscale('log')

fig=plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.imshow(composite_img0,vmin=200,vmax=700,cmap='gray',aspect='auto',extent=[time[0],time[-1],g.im_extent[2],g.im_extent[3]])

ax2.imshow(np.flipud(composite_img1.T),vmin=200,vmax=700,cmap='gray',aspect='auto',extent=[g.im_extent[0],g.im_extent[1],time[0],time[-1]])

tip_loc= [_ for _ in range(len(cine_list))]
for ci in range(len(cine_list)):

    vim = composite_img0[ci]
    firstcol = vim[:,0].copy()
    firstcol = firstcol[:,np.newaxis]
    vim = vim-firstcol
    
    plt.figure()
    plt.imshow(vim,aspect='auto')
    
    plt.figure()
    for t in np.arange(0,1500,200):
        plt.plot(vim[:,t],color=cs[ci])
        
    cutoff = -40
    tip_loc[ci] = np.zeros(np.shape(vim)[1])
    for i in range(np.shape(vim)[1]):
        v = np.argwhere(vim[:,i]<cutoff)
        if len(v)>0:
            tip_loc[ci][i] = g(v[0])
        else:
            tip_loc[ci][i]= np.nan
        
plt.figure()
for ci in range(len(cine_list)):
    plt.plot(tip_loc[ci] , color=cs[ci])

#[axs[ci].set_title(title_list[ci]) for ci in range(3)]
[axs[ci].set_xlabel('time [s]') for ci in [4,5,6,7]]
[ax.grid(True,linestyle='--',color='k',alpha=0.5) for ax in axs]
[ax.set_xlim([0,0.95]) for ax in axs]
[axs[ci].xaxis.set_ticklabels([]) for ci in [0,1,2,3]]
[axs[ci].yaxis.set_ticklabels([]) for ci in [1,2,3,5,6,7]]
[axs[ci].set_ylabel('vertical position [m]') for ci in [0,4]]
plt.tight_layout()
fig.savefig(folder+'comparisons.png')


fig= plt.figure()
ax = fig.add_subplot(111)

import matplotlib
cmap = matplotlib.cm.get_cmap('inferno')
rgba = cmap(0.5)

#base_colors = [cmap(i)[0:3] for i in np.linspace(0,1,len(file_list))]
base_colors = [[1,0,0],[0,1,0],[0,0,1]]

for ci in np.arange(len(file_list)):
    comp = composite_img_list[ci]
    colors4 = np.zeros((np.shape(comp)[0],np.shape(comp)[1],4))
    
    for i in [0,1,2]:
        #im = (base_colors[fi][i] * (crop(c[f]) + 500)) / scale
        im = base_colors[ci][i] * np.ones(np.shape(comp))
        colors4[:,:,i] = im
        
    alpha = (comp -10) / 70
    alpha[alpha<0] = 0
    alpha[alpha>0.8] = 0.8
    colors4[:,:,3] = alpha
    
    ax.imshow(colors4)