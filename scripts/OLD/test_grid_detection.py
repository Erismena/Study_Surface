# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 09:42:16 2017

@author: danjr
"""

import numpy as np
import pims
import matplotlib.pyplot as plt
import pandas as pd
import scipy.ndimage.filters
import scipy.signal

parent_folder = r'C:\Users\danjr\Documents\Fluids Research\Data\misc\cines_backlight_copied170818\\'
cine_filename = r'Backlight_bubble_fps1000_svLower23s_mz100mm_A10mm_f05Hz_largeNeedle_grid3x4x6_Cam_20861_Cine7'

cine_filepath = parent_folder+cine_filename+'.cine'
c = pims.open(cine_filepath)

frames = np.arange(500,3000,3)

im_shape = c[0].shape

'''

ims = np.zeros([len(frames),im_shape[0],im_shape[1]])

for fi,f in enumerate(frames):
    im = c[f]
    ims[fi,:,:] = im
    
plt.figure()
plt.imshow(np.mean(ims,axis=1),aspect='auto')

f = np.fft.fft(np.mean(np.mean(ims,axis=2),axis=1))

plt.figure()
plt.plot(f.real)
plt.plot(f.imag)
'''

def crop(im):
    return im[:,390:1175]

fig = plt.figure(figsize=(6,9))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2,sharex=ax1)

for f in frames:
    
    ax1.clear()
    ax2.clear()
    
    im = crop(c[f])
    
    #kern = np.ones((11, 3))
    #kern /= kern.sum()
    #im = scipy.signal.convolve(im, kern,'same')
    
    #top_region = im[110:130,:]
    #intns = scipy.signal.medfilt(np.mean(top_region,0),3)
    
    grad = np.gradient(im,axis=0)**2
    #grad = grad[2::-2,2::-2]
    grad = grad[5:750,:]
    
    intns = np.mean(grad,axis=0)
    
    thresh = 700
    bad_region = np.argwhere(intns>thresh)
    
    mask = np.zeros(np.shape(im),dtype=bool)
    mask[:,bad_region] = 1
    
    ax1.imshow(im,cmap='gray')
    ax2.plot(intns)
    ax1.imshow(mask,cmap='jet',alpha=0.3)
    
    plt.show()
    plt.pause(.1)