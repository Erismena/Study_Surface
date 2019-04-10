# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 10:25:49 2017

@author: danjr
"""

import numpy as np
import pims
import matplotlib.pyplot as plt
import pandas as pd
import scipy.ndimage.filters
import fluids2d.backlight

parent_folder = r'C:\Users\danjr\Documents\Fluids Research\Data\misc\cines_backlight_copied170818\\'
cine_filename = r'Backlight_bubble_fps1000_svLower23s_mz100mm_A10mm_f05Hz_largeNeedle_grid3x4x6_Cam_20861_Cine7'

cine_filepath = parent_folder+cine_filename+'.cine'
c = pims.open(cine_filepath)

df_all = pd.DataFrame()

frames = np.arange(600,len(c),5)

flow_dir = 'horizontal'

def crop(im):
    return im[10:790,390:1175]

masker = fluids2d.backlight.Masker(np.shape(c[0]),'horizontal',np.arange(104,200,1),800,min_filter_dist=0)

for ci in frames:   
    
    print('Reading frame '+str(ci)+' of '+str(max(frames))+'.')
        
    im = crop(c[ci])

    mask = masker.create_mask(im)

    filled = fluids2d.backlight.raw2binaryfilled(im,mask,800)
    
    df = fluids2d.backlight.filled2regionpropsdf(filled,min_area=40,frame=ci)            
    df_all = pd.concat([df_all,df])
    
    
'''
Track the particles
'''
import trackpy as tp
t = tp.link_df(df_all,50,memory=5,adaptive_stop=2)

'''
Split up the dataframe into separate ones for each particle
'''
particle_nums = t['particle'].unique()

particles = {}
for pi in particle_nums:
    particles[pi] = t[t['particle']==pi]
    particles[pi] = particles[pi].set_index('frame')
    
    if (particles[pi]['x'].max() < 610) & (particles[pi]['x'].min() > 300):
        del particles[pi]
    
# filter out the spurious particles
particles = {pi:particles[pi] for pi in list(particles.keys()) if len(particles[pi])>4}

# Get the derivatives
diffs = {}
for pi in list(particles.keys()):
    d = particles[pi].diff() 
    d_index = d.index[1:] - d.index[:-1]
    
    for col in d.columns:
        d[col].iloc[1:] = d[col].iloc[1:] / d_index
    diffs[pi] = d

#diffs = {pi:particles[pi].diff() for pi in list(particles.keys())}

'''
Reconstruct one big df
'''

plt.figure()
for pi in list(particles.keys()):
    plt.scatter(particles[pi]['x'].rolling(window=3,center=True).median(),particles[pi]['filled_area'].rolling(window=3,center=True).median(),alpha=0.2)

'''
Animate the annotated video
'''

def f_n(n,sigma,rho,d):
    return float(1) / (2*np.pi) * np.sqrt( 8*(n-1)*(n+1)*(n+2)*(sigma / (rho*d**3)))


for fi in frames:
    
    print('Working on frame '+str(fi)+'.')
    
    fig = plt.figure(figsize=(14,7))
    ax = fig.add_subplot(111)
    ax.clear()
    
    im = c[fi]
    imshape = np.shape(im)
    ax.imshow(c[fi],cmap='gray',origin='lower')
    
    '''
    Show the paths of the particles
    '''
    for part in list(particles.keys()):
        
        #part_path = t.copy()[(t['particle']==part)&(t['frame']<=fi)].set_index('frame')
        
        #part_path_all_vols = t.copy()[(t['particle']==part)].set_index('frame')
        part_path_all_vols = particles[part]
        part_path = part_path_all_vols[part_path_all_vols.index<=fi]
        

        if (part_path_all_vols.index.max()>=fi) & (part_path_all_vols.index.min()<=fi):
            ax.plot(part_path['x'],part_path['y'],linestyle='-',alpha=0.8,color='r',markersize=5)
            
            centroid = [part_path.loc[part_path.index==fi,'x'],part_path.loc[part_path.index==fi,'y']]
            if len(centroid[0]>0):
                ax.plot(part_path.loc[part_path.index==fi,'x'],part_path.loc[part_path.index==fi,'y'],marker='o',alpha=0.8,color='r',markersize=5)            
                from matplotlib.patches import Ellipse
                ellipse = Ellipse(xy=centroid, width=part_path.loc[part_path.index==fi,'major_axis_length'], height=part_path.loc[part_path.index==fi,'minor_axis_length'], angle = np.pi - part_path.loc[part_path.index==fi,'orientation']*360/(2*np.pi),alpha=0.8,facecolor='None', edgecolor='blue',linewidth=3)
                ax.add_patch(ellipse)
                
    ax.set_ylim([0,imshape[0]])
    ax.set_xlim([0,imshape[1]])
            
    ax.set_title('t = {:01.4f} s'.format(c.get_time(fi)))
    plt.show()
    plt.pause(.1)
    
    fig.savefig(parent_folder+cine_filename+r'\\frame_'+format(fi,'04')+'.png')
    plt.close(fig)
    
'''
Plot area of bubbles
'''

plt.figure()
for pi in list(particles.keys()):
    plt.plot(particles[pi].index,particles[pi]['filled_area'].rolling(window=3,center=True).median())
