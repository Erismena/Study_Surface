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

parent_folder = r'E:\Experiments_Stephane\Grid column\Backlight_bubbles\20170817\Backlight_bubble_fps1000_svCenterThird_mz100mm_A10mm_f05Hz_largeNeedle_grid3x4\\'
#cine_filename = r'Backlight_bubble_fps1000_svCenterThird_mz100mm_A00mm_f00Hz_largeNeedle_grid3x4_Cam_20861_Cine1'

cine_filenames = [r'Backlight_bubble_fps1000_svCenterThird_mz100mm_A10mm_f05Hz_largeNeedle_grid3x4_Cam_20861_Cine1',
                  r'Backlight_bubble_fps1000_svCenterThird_mz100mm_A10mm_f05Hz_largeNeedle_grid3x4_Cam_20861_Cine2',
                  r'Backlight_bubble_fps1000_svCenterThird_mz100mm_A10mm_f05Hz_largeNeedle_grid3x4_Cam_20861_Cine3',
                  r'Backlight_bubble_fps1000_svCenterThird_mz100mm_A10mm_f05Hz_largeNeedle_grid3x4_Cam_20861_Cine4',
                  r'Backlight_bubble_fps1000_svCenterThird_mz100mm_A10mm_f05Hz_largeNeedle_grid3x4_Cam_20861_Cine5',
                  r'Backlight_bubble_fps1000_svCenterThird_mz100mm_A10mm_f05Hz_largeNeedle_grid3x4_Cam_20861_Cine6',
                  r'Backlight_bubble_fps1000_svCenterThird_mz100mm_A10mm_f05Hz_largeNeedle_grid3x4_Cam_20861_Cine7',
                  r'Backlight_bubble_fps1000_svCenterThird_mz100mm_A10mm_f05Hz_largeNeedle_grid3x4_Cam_20861_Cine8',
                  r'Backlight_bubble_fps1000_svCenterThird_mz100mm_A10mm_f05Hz_largeNeedle_grid3x4_Cam_20861_Cine9',
                  r'Backlight_bubble_fps1000_svCenterThird_mz100mm_A10mm_f05Hz_largeNeedle_grid3x4_Cam_20861_Cine10']

for cine_filename in cine_filenames:
    
    cine_filepath = parent_folder+cine_filename+'.cine'
    c = pims.open(cine_filepath)
    
    df_all = pd.DataFrame()
    
    
    flow_dir = 'horizontal'
    
    frames = np.arange(0,len(c),1)
    
    fig,ax = plt.subplots(2,2,figsize=(15,8),sharex=True,sharey=True)
    
    for ci in frames:   
        
        print('Reading frame '+str(ci)+' of '+str(max(frames))+'.')
        
        #[axi.clear() for axi in ax.reshape(-1)]
        
        im = c[ci]
        #im = im[150:620,90:900]    
        im[0:200,:] = 1024
        im[585:,:] = 1024
        im[:,0:90] = 1024
        im[:,900:] = 1024
        
        '''
        Mask
        '''
        
    #    if flow_dir == 'horizontal':
    #        top_region = im[0:20,:]
    #        intns = np.mean(top_region,0)
    #        thresh = 600
    #        bad_region = np.argwhere(intns>thresh)    
    #        
    #        mask = np.zeros(np.shape(im),dtype=bool)
    #        mask[:,bad_region] = 1
    #    
    #    if flow_dir == 'vertical':
    #        
    #        side_region = im[:,100:150]
    #        intns = np.mean(side_region,1)
    #        thresh = 10000
    #        bad_region = np.argwhere(intns>thresh)    
    #        
    #        mask = np.zeros(np.shape(im),dtype=bool)
    #        mask[:,bad_region] = 1
        
        #ax[0][0].imshow(im,cmap='gray')
        top_region = im[:,90:120]
        intns = np.mean(top_region,1)
        thresh = 0
        good_region = np.argwhere(intns>thresh)
    #    
    ##    fig=plt.figure()
    ##    ax1=fig.add_subplot(3,1,1)
    ##    ax2=fig.add_subplot(3,1,2,sharex=ax1)
    ##    ax3=fig.add_subplot(3,1,3,sharex=ax1)
    ##    
    ##    ax1.imshow(im,cmap='gray')
    ##    ax2.plot(intns)
    ##    
    ##    im_smoothed = scipy.ndimage.uniform_filter1d(im, 3,0)
    ##    
    ##    ax3.plot(np.mean(np.abs(np.gradient(im_smoothed,axis=0)),axis=0))
    
        mask = np.zeros(np.shape(im),dtype=int)
        mask[good_region,:] = 1
        mask[0:5,:] = 1
        
        #ax[0][0].imshow(im,cmap='gray')
        #ax[0][0].imshow(mask,cmap='jet',alpha=0.3)
        
        '''
        Median filter
        '''
        #import scipy.ndimage.filters
        #im_filt = scipy.ndimage.filters.median_filter(im,size=1)    
        im_filt=im
        im_filt = im_filt>800
        im_filt = 1-np.array(im_filt,dtype=int)
        #im_filt = im_filt * mask
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
                
        df_all = pd.concat([df_all,df])
        
        #plt.show()
        #plt.pause(0.1)
        
    '''
    Track the particles
    '''
    import trackpy as tp
    t = tp.link_df(df_all,100,memory=5,adaptive_stop=100)
    
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
    
#    # Get the derivatives
#    diffs = {}
#    for pi in list(particles.keys()):
#        d = particles[pi].diff() 
#        d_index = d.index[1:] - d.index[:-1]
#        
#        for col in d.columns:
#            d[col].iloc[1:] = d[col].iloc[1:] / d_index
#        diffs[pi] = d
#    
#    #diffs = {pi:particles[pi].diff() for pi in list(particles.keys())}
#    
#    '''
#    Reconstruct one big df
#    '''
#    
#    plt.figure()
#    for pi in list(particles.keys()):
#        plt.scatter(particles[pi]['y'].rolling(window=3,center=True).median(),particles[pi]['orientation'].rolling(window=3,center=True).median(),alpha=0.2)
#    
#    '''
#    Animate the annotated video
#    '''
#    
#    imgs_dir = parent_folder+cine_filename+r'\\'
#    
#    import os.path
#    if not os.path.exists(imgs_dir):
#        os.makedirs(imgs_dir)
#    
#    for fi in frames:
#        
#        print('Working on frame '+str(fi)+'.')
#        
#        fig = plt.figure(figsize=(14,7))
#        ax = fig.add_subplot(111)
#        ax.clear()
#        
#        im = c[fi]
#        imshape = np.shape(im)
#        ax.imshow(c[fi],cmap='gray',origin='lower')
#        
#        '''
#        Show the paths of the particles
#        '''
#        for part in list(particles.keys()):
#            
#            #part_path = t.copy()[(t['particle']==part)&(t['frame']<=fi)].set_index('frame')
#            
#            #part_path_all_vols = t.copy()[(t['particle']==part)].set_index('frame')
#            part_path_all_vols = particles[part]
#            part_path = part_path_all_vols[part_path_all_vols.index<=fi]
#            
#            if (part_path_all_vols.index.max()>=fi) & (part_path_all_vols.index.min()<=fi):
#                ax.plot(part_path['x'],part_path['y'],linestyle='-',alpha=0.8,color='r',markersize=5)
#                
#                centroid = [part_path.loc[part_path.index==fi,'x'],part_path.loc[part_path.index==fi,'y']]
#                if len(centroid[0]>0):
#                    ax.plot(part_path.loc[part_path.index==fi,'x'],part_path.loc[part_path.index==fi,'y'],marker='o',alpha=0.8,color='r',markersize=5)            
#                    from matplotlib.patches import Ellipse
#                    ellipse = Ellipse(xy=centroid, width=part_path.loc[part_path.index==fi,'major_axis_length'], height=part_path.loc[part_path.index==fi,'minor_axis_length'], angle = np.pi - part_path.loc[part_path.index==fi,'orientation']*360/(2*np.pi),alpha=0.8,facecolor='None', edgecolor='blue',linewidth=3)
#                    ax.add_patch(ellipse)
#                    
#        ax.set_ylim([0,imshape[0]])
#        ax.set_xlim([0,imshape[1]])
#                
#        ax.set_title('t = {:01.4f} s'.format(c.get_time(fi)))
#        plt.show()
#        plt.pause(.1)
#        
#        fig.savefig(imgs_dir+r'frame_'+format(fi,'04')+'.png')
#        plt.close(fig)
#        
#    '''
#    Plot area of bubbles
#    '''
#    
#    plt.figure()
#    for pi in list(particles.keys()):
#        plt.plot(particles[pi].index,particles[pi]['perimeter'].rolling(window=3,center=True).median())
