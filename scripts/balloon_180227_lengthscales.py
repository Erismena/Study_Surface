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
import scipy.signal



folder = r'E:\Stephane\171114\\'
cine_name = r'balloon_breakup_pumps_fps10000_backlight_D400minch_d28mm'

#folder = r'\\Mae-deike-lab3\d\data_comp3_D\180210\\'
#cine_name = r'backlight_bubbles_4pumps_stackedParallel__sunbathing_meanon100_meanoff400_L450mm_h080mm_fps1000'

c = pims.open(folder+cine_name+'.cine')
dt = 1./1000
dx= 0.000211798121201
pad = 5

frames = np.arange(5000,30000,2500)

fig= plt.figure()
ax = fig.add_subplot(111)

for fi,f in enumerate(frames):

    color = [0.5,0.5,float(fi)/len(frames)]

    im = c[f].astype(float)
    im = im<500
    im=im.astype(float)
    im[im==0]=-1
    
    
    im_smaller = im[pad:-pad,pad:-pad]
    shape_smaller = np.shape(im_smaller)
    
    res = np.zeros((pad*2,pad*2))
    
    for x in np.arange(2*pad):
        print(x)
        for y in np.arange(2*pad):
            res[y,x] = np.sum(im_smaller * im[y:y+shape_smaller[0],x:x+shape_smaller[1]])

#conv = scipy.signal.convolve(im,im_smaller,mode='valid',method='direct')

    plt.figure()
    plt.imshow(im)
    
    plt.figure()
    plt.imshow(res)
    
    #plt.figure()
    ax.plot(res[:,pad],color=color)
    ax.plot(res[pad,:],color=color)