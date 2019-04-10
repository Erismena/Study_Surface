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
import skimage.measure
import skimage.feature
import fluids2d.bubble_fit_ga as ga
import skimage.morphology



folder = r'E:\Stephane\171114\\'
cine_name = r'balloon_breakup_pumps_fps10000_backlight_D400minch_d28mm'

#folder = r'\\Mae-deike-lab3\d\data_comp3_D\180210\\'
#cine_name = r'backlight_bubbles_4pumps_stackedParallel__sunbathing_meanon100_meanoff400_L450mm_h080mm_fps1000'

c = pims.open(folder+cine_name+'.cine')
dt = 1./1000
dx= 0.000211798121201

f = 12700

im = c[f].astype(float)

frames = np.arange(f,f+500,2)
c_frames = np.array(c[frames])

g = fluids2d.geometry.GeometryScaler(dx=dx,im_shape=np.shape(c[0]),origin_pos=(0,0),origin_units='m')

#'''
#Get the image
#'''
#im = np.array(c[f].astype(float))
#im = scipy.ndimage.filters.gaussian_filter(im,3)
#
#def divergence(im):
#    dI2dx2 = np.gradient(np.gradient(im,axis=0),axis=0)
#    dI2dy2 = np.gradient(np.gradient(im,axis=1),axis=1)
#    return dI2dx2 + dI2dy2
#
#div = divergence(im)
#divdiv = divergence(div)
#
#
#is_border = np.zeros(np.shape(div))
#is_border[(div>-5) & (div<5)] = 1
#
#
#print(np.shape(div))
#
#thresh= 10
#
#fig = plt.figure(figsize=(10,4));
#ax_im=fig.add_subplot(121)
#ax_edges = fig.add_subplot(122,sharex=ax_im,sharey=ax_im)
#ax_im.imshow(im,cmap='gray')
##ax_edges.imshow(div,vmin=-thresh,vmax=thresh,cmap='PuOr')
#
#ax_edges.imshow(div,cmap='gray')

#div_contours = skimage.measure.find_contours(div,0)
#
#for dc in div_contours:
#    if np.shape(dc)[0]>5:
#        ax_im.plot(dc[:,1],dc[:,0])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(im)

y = 675
x = 400

ax.plot(x,y,'x',color='r')

dx = 100
dy = 2

c_region = c_frames[:,y-dy:y+dy,x-dx:x+dx]

#fig,axs = plt.subplots(1,2,figsize=(12,6)); axs=axs.flatten()
#[axs[i].imshow(np.nanmean(c_region,axis=i+1),aspect='auto') for i in [0,1]]

fig = plt.figure()
ax_im = fig.add_subplot(2,2,1)
ax_x = fig.add_subplot(2,2,3,sharex=ax_im)
ax_y = fig.add_subplot(2,2,2,sharey=ax_im)

ax_im.imshow(np.nanmean(c_region,axis=0))

ax_x.imshow(np.nanmean(c_region,axis=1),aspect='auto')
ax_y.imshow(np.nanmean(c_region,axis=2).T,aspect='auto')
