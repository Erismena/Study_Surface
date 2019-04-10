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
import trackpy as tp
import multiprocessing as mp


    
folder = r'\\Mae-deike-lab3\c\Users\Luc Deike\data_comp3_C\180319\\'
cine_name = r'backlight_bubblesRising_tinyNeedle_2x2x2_on100_off400_fps1000_withCap'

c = pims.open(folder+cine_name+'.cine')

thresh = 300

dt = 1./1000
dx= 9.8467328387E-05
g = fluids2d.geometry.GeometryScaler(dx=dx,im_shape=(2048,2048),origin_pos=(-1024,-1024),origin_units='pix')

def mask(im):
    im = im.astype(float)
    im[:,0:100] = 1000
    im[:,1960:] = 1000
    im[0:150:,:] = 1000
    im[2000::,:] = 1000
    #im = skimage.filters.median(im,3)
    #im = np.rot90(im,k=1)
    return im

def get_filled(im,thresh):
    im_filt = scipy.ndimage.filters.median_filter(im,size=3)
    im_filt = fluids2d.backlight.binarize(im_filt,thresh,large_true=False)
    filled = binary_fill_holes(im_filt)
    return filled

def filled_props(filled,g):
    
    objects, num_objects = scipy.ndimage.label(filled)
    props = skimage.measure.regionprops(objects)
    
    df = pd.DataFrame(columns=['frame','radius','x','y'])
    if (len(props)==0)==False:
        for ri,region in enumerate(props):
            y,x = g.get_loc([region.centroid[0],region.centroid[1]])
            df.loc[ri,'y']=y[0]
            df.loc[ri,'x']=x[0]
            df.loc[ri,'radius'] = np.sqrt(region.filled_area * dx**2 / np.pi)
            df.loc[ri,'orientation'] = region.orientation / (2*np.pi) * 360
            df.loc[ri,'major_axis_length'] = region.major_axis_length* dx
            df.loc[ri,'minor_axis_length'] = region.minor_axis_length* dx
            
    return df

frames = np.arange(0,12111,1)

len_cine = len(frames)

frames_list = frames = [np.arange(i*1000,i*1000+1000,1) for i in np.arange(0,12)]

def process_frames(frames_list):
    
    df_all = pd.DataFrame()
 
    for i,f in enumerate(frames):
        
        print('frame '+str(f))
        
            
        im = mask(c[f]) #- bgs[ci]
        
        print('. get_filled')
        filled = get_filled(im,thresh)
        
        print('. filled_props')
        df = filled_props(filled,g)
        df['frame'] = f
        df = df[df['radius']>0.0001]
        df = df[df['radius']<0.05] 
        print('... found '+str(len(df))+' objects.')
        #df = df[df['y']>0.025]
        #df = df[df['y']<0.112]
        #df = df[df['y']<0.245]
        df_all = pd.concat([df_all,df])
        
    return df_all


df_all['time'] = df_all['frame']*dt        
df_filtered = df_all[df_all['radius']>0.0005]
df_filtered = df_filtered[df_filtered['radius']<0.02]

if __name__ == '__main__':
        
    jobs = []
    for frames in frames_list:
        p = mp.Process(target=frames_list,args=(frames,))
        jobs.append(p)
        p.start()
    
        