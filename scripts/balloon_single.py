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
import skimage.feature
import fluids2d.bubble_fit_ga as ga


folder = r'\\DESKTOP-TDOAU0M\PIV_Data\Stephane\171114\\'
cine_name = r'balloon_breakup_pumps_fps10000_backlight_D400minch_d28mm'

#folder = r'\\Mae-deike-lab3\d\data_comp3_D\180210\\'
#cine_name = r'backlight_bubbles_4pumps_stackedParallel__sunbathing_meanon100_meanoff400_L450mm_h080mm_fps1000'

c = pims.open(folder+cine_name+'.cine')
dt = 1./1000
dx= 0.000211798121201

f = 15600

frame_start = 0
cmaps = ['Greens','Purples','Oranges']
cs = ['g','purple','orange']
thresh = -100

g = fluids2d.geometry.GeometryScaler(dx=dx,im_shape=np.shape(c[0]),origin_pos=(0,0),origin_units='m')

import matplotlib as mpl
import matplotlib.cm as cm

norm = mpl.colors.Normalize(vmin=0.5, vmax=1.5)
cmap = cm.spectral
m = cm.ScalarMappable(norm=norm, cmap=cmap)

def mask(im):
    im = im.astype(float)
    #im = np.rot90(im,k=3)
    #im[:,724:1100] = 1000
    return im[440:500,590:640]
    #return im[:,724:1100]

def get_filled(im,thresh):
    im_filt = scipy.ndimage.filters.median_filter(im,size=2)
    #im_filt = im.copy()
    im_filt = fluids2d.backlight.binarize(im_filt,thresh,large_true=False)
    filled = binary_fill_holes(im_filt)
    return filled

def filled_props(filled,g,intensity_image=None):
    
    objects, num_objects = scipy.ndimage.label(filled)
    props = skimage.measure.regionprops(objects,intensity_image=intensity_image)
    
    df = pd.DataFrame(columns=['frame','radius','x','y'])
    if (len(props)==0)==False:
        for ri,region in enumerate(props):
            print(region.centroid)
            
            df.loc[ri,'y'],df.loc[ri,'x'] = g.get_loc([int(region.centroid[0]),int(region.centroid[1])])
            df.loc[ri,'filled_area'] = region.filled_area * g.dx**2
            df.loc[ri,'radius'] = np.sqrt(region.filled_area * g.dx**2 / np.pi)
            df.loc[ri,'orientation'] = region.orientation / (2*np.pi) * 360
            df.loc[ri,'major_axis_length'] = region.major_axis_length* g.dx
            df.loc[ri,'minor_axis_length'] = region.minor_axis_length* g.dx
            df.loc[ri,'e'] = df.loc[ri,'major_axis_length'] / df.loc[ri,'minor_axis_length']
            df.loc[ri,'perimiter_bubble'] = region.perimeter * g.dx
            df.loc[ri,'perimeter_ellipse'] = np.pi * ( 3 * (df.loc[ri,'major_axis_length']/2. + df.loc[ri,'minor_axis_length']/2.) - np.sqrt((3*df.loc[ri,'major_axis_length']/2. + df.loc[ri,'minor_axis_length']/2.) * (df.loc[ri,'major_axis_length']/2. + 3*df.loc[ri,'minor_axis_length']/2.)) )
            df.loc[ri,'area_ellipse'] = np.pi * df.loc[ri,'major_axis_length'] * df.loc[ri,'minor_axis_length'] / 4.
            df.loc[ri,'bounding_portion'] = df.loc[ri,'area_ellipse'] / df.loc[ri,'filled_area']
            df.loc[ri,'perimeter_ratio'] = df.loc[ri,'perimeter_ellipse'] / df.loc[ri,'perimiter_bubble']
            
    return df,props


'''
Get the image
'''
bg = mask(c[0])
im = mask(c[f]) - np.median(mask(c[f]))#-bg

'''
Fill it and get its properties
'''
    
filled = get_filled(im,thresh)

df,props = filled_props(filled,g,intensity_image=im)
df = df[df['radius']>0.001]
df = df[df['radius']<0.05] 

'''
Decide whether each bubble needs more attention, based on comparisons between
the original image and the fitted ellipse
'''

df['bad'] = False
#df['bad'][df['e']>2.] = True
df['bad'][df['bounding_portion']>1.1] = True
df['bad'][df['bounding_portion']<0.9] = True
df['bad'][df['perimeter_ratio']>1.1] = True
df['bad'][df['perimeter_ratio']<0.9] = True


'''
Show various manipulations of the image
'''

#fig,axs = plt.subplots(2,2,sharex=True,sharey=True); axs=axs.flatten()
fig,axs = plt.subplots(1,1,sharex=True,sharey=True); axs=[axs,]

axs[0].imshow(im,cmap='gray',vmin=-600,vmax=100,extent=g.im_extent,origin='upper')
#axs[1].imshow(filled,cmap='gray',extent=g.im_extent,origin='upper')

#edges = skimage.feature.canny(im,sigma=4)
#axs[2].imshow(edges,cmap='gray',extent=g.im_extent,origin='upper')

#lpn = np.gradient(np.gradient(im,axis=0),axis=0) + np.gradient(np.gradient(im,axis=1),axis=1)
#axs[3].imshow(np.abs(lpn),cmap='gray',extent=g.im_extent,origin='upper')
#ax.imshow(fluids2d.backlight.alpha_binary_cmap(filled.astype(float),cm.get_cmap(cmaps[ci])),extent=g.im_extent)

color_dict = {True:'r',False:'g'}

for ax in axs:
    for ix in df.index:
        e = Ellipse([df.loc[ix,'x'],df.loc[ix,'y']],width=df.loc[ix,'major_axis_length'],height=df.loc[ix,'minor_axis_length'],angle=df.loc[ix,'orientation'])
        ax.add_artist(e)
        e.set_facecolor('None')
        #e.set_edgecolor(m.to_rgba(df.loc[ix,'perimeter_ratio']))
        e.set_edgecolor(color_dict[df.loc[ix,'bad']])
        
df_bad = df.copy()[df['bad']==True]

for ri in df_bad.index:
    
    
    
    r = props[ri]
    
    best_params = ga.fit_subimage_with_ga(r.image.astype(int).copy(),r.intensity_image.astype(int).copy(),n_bubbles=2)
    
    for ax in axs:
        for p in best_params:
            yi,xi=g.get_loc([p['y']+r.bbox[0],p['x']+r.bbox[1]])
            e = Ellipse((xi,yi),width=2*p['major_axis_length']*g.dx,height=2*p['minor_axis_length']*g.dx,angle=p['orientation']/np.pi*180.)
            ax.add_artist(e)
            e.set_facecolor('None')
            #e.set_edgecolor(m.to_rgba(df.loc[ix,'perimeter_ratio']))
            e.set_edgecolor('cyan')

im_small = props[11].image.astype(int).copy()
im_small_orig = props[11].intensity_image.astype(int).copy()


#frames_to_avg = np.arange(2500,3500,1)
#ri = 300
#ci = 125
#rowval = np.zeros((len(frames_to_avg),np.shape(im)[0]))
#colval = np.zeros((np.shape(im)[0],len(frames_to_avg)))
#for fi,f in enumerate(frames_to_avg):
#    rowval[fi,:] = np.nanmean(c[f][ri:ri+1,:],axis=0)
#    colval[:,fi] = np.nanmean(c[f][:,ci:ci+1],axis=1)
#    
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.imshow(rowval,aspect='auto')
#
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.imshow(colval,aspect='auto')