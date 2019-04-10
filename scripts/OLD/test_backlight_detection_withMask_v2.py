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

parent_folder = r'E:\Experiments_Stephane\Grid column\Backlight_bubbles\20170817\Backlight_bubble_fps1000_svLower23s_mz100mm_A10mm_f05Hz_largeNeedle_grid3x4x6\\'
#cine_filename = r'Backlight_bubble_fps1000_svCenterThird_mz100mm_A00mm_f00Hz_largeNeedle_grid3x4_Cam_20861_Cine1'

cine_filenames = [r'Backlight_bubble_fps1000_svLower23s_mz100mm_A10mm_f05Hz_largeNeedle_grid3x4x6_Cam_20861_Cine1',
                  r'Backlight_bubble_fps1000_svLower23s_mz100mm_A10mm_f05Hz_largeNeedle_grid3x4x6_Cam_20861_Cine2',
                  r'Backlight_bubble_fps1000_svLower23s_mz100mm_A10mm_f05Hz_largeNeedle_grid3x4x6_Cam_20861_Cine3',
                  r'Backlight_bubble_fps1000_svLower23s_mz100mm_A10mm_f05Hz_largeNeedle_grid3x4x6_Cam_20861_Cine4',
                  r'Backlight_bubble_fps1000_svLower23s_mz100mm_A10mm_f05Hz_largeNeedle_grid3x4x6_Cam_20861_Cine5',
                  r'Backlight_bubble_fps1000_svLower23s_mz100mm_A10mm_f05Hz_largeNeedle_grid3x4x6_Cam_20861_Cine6',
                  r'Backlight_bubble_fps1000_svLower23s_mz100mm_A10mm_f05Hz_largeNeedle_grid3x4x6_Cam_20861_Cine7',
                  r'Backlight_bubble_fps1000_svLower23s_mz100mm_A10mm_f05Hz_largeNeedle_grid3x4x6_Cam_20861_Cine8',
                  r'Backlight_bubble_fps1000_svLower23s_mz100mm_A10mm_f05Hz_largeNeedle_grid3x4x6_Cam_20861_Cine9',
                  r'Backlight_bubble_fps1000_svLower23s_mz100mm_A10mm_f05Hz_largeNeedle_grid3x4x6_Cam_20861_Cine10']

def crop(im):
    return im[10:790,390:1175]

for cine_filename in cine_filenames:
    
    cine_filepath = parent_folder+cine_filename+'.cine'
    c = pims.open(cine_filepath)
    
    df_all = pd.DataFrame()
    
    
    flow_dir = 'horizontal'
    
    frames = np.arange(0,len(c),1)
    
    fig,ax = plt.subplots(2,2,figsize=(15,8),sharex=True,sharey=True)
    
    for ci in frames:   
        
        print('Reading frame '+str(ci)+' of '+str(max(frames))+'.')
                
        im = crop(c[ci])
        
        '''
        Mask
        '''
        
        if flow_dir == 'horizontal':
                top_region = im[104:200,:]
                intns = scipy.ndimage.filters.minimum_filter(np.mean(top_region,0),20)
                thresh = 800
                good_region = np.argwhere(intns>thresh).flatten()
                
                
                
                good_diff = np.gradient(good_region)
                good_diff_jump_locs = good_region[np.argwhere(good_diff>2).flatten()]
                
                good_diff_jump_locs_diff = np.gradient(good_diff_jump_locs)
                fill_locs = good_diff_jump_locs[np.argwhere(good_diff_jump_locs_diff>1).flatten()]
                
                mask = np.zeros(np.shape(im),dtype=int)
                mask[:,good_region] = 1

    

        #ax[0][0].imshow(im,cmap='gray')
        #ax[0][0].imshow(mask,cmap='jet',alpha=0.3)
        
        '''
        Median filter
        '''
        #import scipy.ndimage.filters
        #im_filt = scipy.ndimage.filters.median_filter(im,size=1)    
        im_filt = im < 800
        #im_filt = 1-np.array(im_filt,dtype=int)
        im_filt = np.array(im_filt,dtype=int) * mask
        #ax[0,1].imshow(im_filt,cmap='gray')
          
        
        '''
        Detect the edges
        '''
        #from skimage import feature
        #edges = feature.canny(im_filt,low_threshold=50,high_threshold=800,sigma=.1,mask=mask)
        #ax[1,0].imshow(edges)
        
        '''
        Fill the holes
        '''
        from scipy.ndimage.morphology import binary_fill_holes
        #filled = binary_fill_holes(scipy.ndimage.filters.median_filter(im_filt,size=1))
        #filled = binary_fill_holes(scipy.ndimage.filters.median_filter(filled,size=3))
        filled = binary_fill_holes(im_filt)
        #ax[1,1].imshow(filled)
        
        '''
        Find the regions and get their properties
        '''
        import skimage.measure
        objects, num_objects = scipy.ndimage.label(filled)
        props = skimage.measure.regionprops(objects)
        
        df = pd.DataFrame()
        if (len(props)==0)==False:
            for ri,region in enumerate(props):
                df.loc[ri,'x'] = region.centroid[1]
                df.loc[ri,'y'] = region.centroid[0]
                df.loc[ri,'filled_area'] = region.filled_area
                df.loc[ri,'orientation'] = region.orientation
                df.loc[ri,'minor_axis_length'] = region.minor_axis_length
                df.loc[ri,'major_axis_length'] = region.major_axis_length
                df.loc[ri,'perimeter'] = region.perimeter
                df['frame'] = ci
            #[axi.scatter(df['x'],df['y'],color='r',s=20,alpha=0.5) for axi in ax.reshape(-1)]
            df = df[df['filled_area']>40]
                
        df_all = pd.concat([df_all,df])
        
        #plt.show()
        #plt.pause(0.1)
        
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
        
    # filter out the spurious particles
    particles = {pi:particles[pi] for pi in particle_nums if len(particles[pi])>4}
    
    import pickle
    pickle.dump(particles, open(parent_folder+cine_filename+'_trackedParticles.pkl', "wb"))