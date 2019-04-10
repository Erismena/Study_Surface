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
import scipy.signal
import pandas as pd
import skimage.measure
import skimage.feature
#import fluids2d.bubble_fit_ga as ga
import skimage.morphology

plt.close('all')

folder = r'E:\Stephane\171114\\'
cine_name = r'balloon_breakup_pumps_fps10000_backlight_D400minch_d28mm'


c = pims.open(folder+cine_name+'.cine')
dt = 1./1000
dx= 0.000211798121201

f = 11500

im = c[f].astype(float)
im_shape = np.shape(im)

plt.figure()
plt.imshow(im)

g = fluids2d.geometry.GeometryScaler(dx=dx,im_shape=np.shape(c[0]),origin_pos=(0,0),origin_units='m')

def divergence(im):
    dI2dx2 = np.gradient(np.gradient(im,axis=0),axis=0)
    dI2dy2 = np.gradient(np.gradient(im,axis=1),axis=1)
    return dI2dx2 + dI2dy2

#edges = skimage.feature.canny(im,sigma=5)

range_im = 200
scale_im = 2
lims = float(range_im)/float(scale_im)**2
#im_smoothed = scipy.signal.medfilt2d(im,kernel_size=3)



sigma_vals = np.linspace(0.5,1.5,11)
d_sigma = sigma_vals[1]-sigma_vals[0]

smoothed = np.zeros((im_shape[0],im_shape[1],len(sigma_vals)))
div = smoothed.copy()

for si,sigma in enumerate(sigma_vals):
    smoothed[:,:,si] = skimage.filters.gaussian(im,sigma=sigma)
    div[:,:,si] = divergence(smoothed[:,:,si])
    
'''
Get the local rate of increase in the divergence wrt gaussian blur amount
'''
div_diff = np.gradient(div,axis=2).mean(axis=2) / d_sigma
div_diff = skimage.filters.gaussian(div_diff,sigma=1)

fig,axs = plt.subplots(1,2,sharex=True,sharey=True,figsize=(10,7))
axs[0].imshow(im)
axs[1].imshow(div_diff,vmin=0,vmax=20,cmap='gist_ncar')

'''
Get the local rate of increase in the image wrt gaussian blur amount
'''
smoothed_diff = np.gradient(smoothed,axis=2).mean(axis=2) / d_sigma
smoothed_diff = skimage.filters.gaussian(smoothed_diff,sigma=1)

fig,axs = plt.subplots(1,2,sharex=True,sharey=True,figsize=(10,7))
axs[0].imshow(im)
axs[1].imshow(smoothed_diff,vmin=0,vmax=20,cmap='gist_ncar')



'''
Hopefully each bubble has a semi-constant value of div_diff around its border,
so pick various ranges of these values by which to skeletonize the image to 
extract the borders. Different bubbles should then show up in different 
skeletonized images.
'''

skel_bins = np.linspace(5,20,10)
skel_window=3
skel = np.zeros((im_shape[0],im_shape[1],len(skel_bins)))
for ski in range(len(skel_bins)):
    min_idx = max(0,ski-skel_window)
    max_idx = min(len(skel_bins)-1,ski+skel_window)
    binary = (div_diff>=skel_bins[min_idx]) * (div_diff<=skel_bins[max_idx])
    skel[:,:,ski] = skimage.morphology.skeletonize(binary)

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111)
for ski in range(len(skel_bins)):
    ax.clear()
    ax.imshow(skel[:,:,ski])
    plt.show()
    plt.pause(1)

plt.figure()
plt.imshow(div_diff**2,vmin=0,vmax=400)

'''

div = divergence(im_smoothed)
plt.figure()
plt.imshow(div,vmin=0,vmax=lims,cmap='gist_ncar')

edge_regions = div>lims

plt.figure()
plt.imshow(edge_regions)

edges_skeletonized = skimage.morphology.skeletonize(edge_regions)
plt.figure()
plt.imshow(edges_skeletonized)

edges_labeled = skimage.measure.label(edges_skeletonized)
plt.figure()
plt.imshow(edges_labeled)

edges_regions = skimage.measure.regionprops(edges_labeled)
edges_regions = [r for r in edges_regions if r.area>10]

areas = [r.area for r in edges_regions]
plt.figure()
plt.hist(areas,bins=100)
'''