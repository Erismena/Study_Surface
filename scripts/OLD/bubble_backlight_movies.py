# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 14:24:22 2017

@author: danjr
"""

import numpy as np
import pims
import matplotlib.pyplot as plt

parent_folder = r'C:\Users\danjr\Documents\Fluids Research\Data\misc\170808_cines\\'
cine_filename = r'Backlight_bubble_sv_makrozeiss100mm_grid3_fps1000_A15mm_f10Hz'

cine_filepath = parent_folder+cine_filename+'.cine'
c = pims.open(cine_filepath)

ci = 20

def crop(im,l=30,r=1150):
    return im

fig,ax = plt.subplots(1,3,sharex=True,sharey=True)

im = crop(c[ci])
ax[0].imshow(im)

top_region = im[0:20,:]
intns = np.mean(top_region,0)
thresh = 600
bad_region = np.argwhere(intns>thresh)

mask = np.zeros(np.shape(im),dtype=bool)
mask[:,bad_region] = 1

from skimage import feature
edges1 = feature.canny(im,low_threshold=400,high_threshold=1000,mask=mask)

ax[1].imshow(edges1)

from scipy.ndimage.morphology import binary_fill_holes
import scipy.ndimage.filters

filled = binary_fill_holes(edges1)

ax[2].imshow(filled)

f = []
orig = []
for ci in np.arange(500,550,2):
    
    print(ci)
    
    im = crop(c[ci])
    
    top_region = im[0:20,:]
    intns = np.mean(top_region,0)
    thresh = 600
    bad_region = np.argwhere(intns>thresh)
    
    mask = np.zeros(np.shape(im),dtype=bool)
    mask[:,bad_region] = 1
    
    im_filt = scipy.ndimage.filters.median_filter(im,size=3)
    
    edges1 = feature.canny(im_filt,low_threshold=200,high_threshold=1000,mask=mask,sigma=0.2) # sigma=3,    
    
    filled = binary_fill_holes(edges1)
    
    f.append(filled)
    orig.append(im)
    
import trackpy as tp
features = tp.batch(f,diameter=(91,55),threshold=0.5,invert=False,minmass=0,separation=(40,20),noise_size=5,smoothing_size=11)

t = tp.link_df(features, 50, memory=7,adaptive_stop=30)

plt.figure()
tp.plot_traj(t)

fig = plt.figure(figsize=(14,7))
ax = fig.add_subplot(111)

particle_nums = t['particle'].unique()


for fi,frame in enumerate(f):
    
    ax.clear()
    ax.imshow(orig[fi],cmap='gray')
    
    ax.imshow(frame,alpha=0.3)
    
    '''
    Show the paths of the particles
    '''
    for part in particle_nums:
        
        part_path = t.copy()[(t['particle']==part)&(t['frame']<=fi)].set_index('frame')
        
        part_path_all_vols = t.copy()[(t['particle']==part)].set_index('frame')
        
        if (len(part_path_all_vols) > -1) & (part_path_all_vols.index.max()>=fi):
            ax.plot(part_path['x'],part_path['y'],linestyle='-',alpha=0.8,color='r',markersize=5)
            ax.plot(part_path.loc[part_path.index==fi,'x'],part_path.loc[part_path.index==fi,'y'],marker='o',alpha=0.8,color='r',markersize=5)
            
    plt.show()
    plt.pause(.1)
    
    fig.savefig(parent_folder+cine_filename+r'\\frame_'+format(fi,'04')+'.png')