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
import fluids2d.masking
import os.path

parent_folder = r'C:\Users\user\Documents\little_camera_data\20170831\Backlight_bubble_sv_wholeColumn_grid3_3x10_gridposLow_fps25_littleCamera_noTurbulence_f00Hz_A00mm\\'
#cine_filename = r'Backlight_bubble_fps1000_svCenterThird_mz100mm_A00mm_f00Hz_largeNeedle_grid3x4_Cam_20861_Cine1'

dt = 1./25 # [s]
dx = 0.29/ 1461. #[m]

filenames = []
import os
for f in os.listdir(parent_folder):
    if f.endswith(".avi"):
        filenames.append(f[:-4]) # get rid of the ".avi"

flow_dir = 'vertical'

def crop(im):
    #return im[:,525:1200,0]
    return im[:,300:970,0]

# create a masker for each file
for cine_filename in filenames:
    
    mask_filepath = parent_folder+cine_filename+'.masker'    
    
    if os.path.isfile(mask_filepath)==False:
        masker = fluids2d.masking.MovingRectMasks(1,motion_direction='vertical')
        cine = pims.open(parent_folder+cine_filename+'.avi')
        masker.calibrate_from_images('1',cine,frames_to_calibrate=1,crop_func=crop)
        masker.calibrate_from_images('2',cine,frames_to_calibrate=1,crop_func=crop)
        masker.calibrate_from_images('3',cine,frames_to_calibrate=1,crop_func=crop)
        #masker.calibrate_from_images('4',cine,frames_to_calibrate=6,crop_func=crop)
        masker.save_self(parent_folder,cine_filename)        
        plt.close('all')



for cine_filename in filenames:
    
    #if os.path.isfile(parent_folder+cine_filename+'_trackedParticles.pkl')==True:
    #    break
    
    cine_filepath = parent_folder+cine_filename+'.avi'
    c = pims.open(cine_filepath)
    
    #masker=fluids2d.masking.load_masker(parent_folder+cine_filename+'.masker')
    masker = fluids2d.masking.load_masker(parent_folder+r'Basler acA2040-90um (22332433)_20170831_173052910'+'.masker')
    
    '''
    Get the background by median filtering along the frame axis
    '''
    num_first_frames = 10
    frame_shape = np.shape(crop(c[0]))
    first_frames = np.zeros([num_first_frames,frame_shape[0],frame_shape[1]])
    for n in range(num_first_frames):
        first_frames[n,:,:] = masker.mask_frame(n,crop(c[n]))
        
    first_frames[first_frames==0] = np.nan
    bg_image = np.nanmedian(first_frames,axis=0)
    bg_image[bg_image==np.nan] = 0
    
    #fig,ax = plt.subplots(1,4,figsize=(15,8),sharex=True,sharey=True)
    
    df_all = pd.DataFrame()
    frames = np.arange(0,len(c),1)
    for ci in frames:   
        
        #[a.clear() for a in ax.flatten()]
        
        print('Reading frame '+str(ci)+' of '+str(max(frames))+'.')
        
        '''
        Load the image
        '''
        im = crop(c[ci])
        #ax[0].imshow(im,cmap='gray')
        #ax[0].set_title('Raw image')

        '''
        Background subtract
        '''
        im = im - bg_image
        #ax[1].imshow(im,cmap='gray')
        #ax[1].set_title('Background subtracted')
        
        '''
        Median filter
        '''
        im_filt = scipy.ndimage.filters.median_filter(im,size=1)    
        #im_filt = im

        '''
        Binarize and flip so strong signal is 1
        '''
        im_filt = np.array(im_filt<-20,dtype=int) # Bubble points are now 1
        
        '''
        Apply the mask
        '''
        im_filt = masker.mask_frame(ci,im_filt)
        #ax[2].imshow(im_filt,cmap='gray')
        #ax[2].set_title('Binarized and masked')
          
        
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
        filled = binary_fill_holes(scipy.ndimage.filters.median_filter(im_filt,size=1))
        #filled = binary_fill_holes(scipy.ndimage.filters.median_filter(filled,size=3))
        filled = binary_fill_holes(im_filt)
        #ax[3].imshow(filled)
        #ax[3].set_title('Holes filled in')
        
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
        df = df[df['filled_area']>20]
                
        #if df.empty==False:
        #    [axi.scatter(df['x'],df['y'],color='r',s=20,alpha=0.5) for axi in ax.reshape(-1)]
                
        df_all = pd.concat([df_all,df])
        
        #plt.show()
        #plt.pause(1)
        
    '''
    Track the particles
    '''
    import trackpy as tp
    t = tp.link_df(df_all,150,memory=5,adaptive_stop=10)
    
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
        plt.scatter(particles[pi]['y'].rolling(window=3,center=True).median(),particles[pi]['orientation'].rolling(window=3,center=True).median(),alpha=0.2)
    
    '''
    Animate the annotated video
    '''
    
    imgs_dir = parent_folder+cine_filename+r'\\'
    
    import os.path
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir)
    
    
    for fi in frames:
        
        print('Working on frame '+str(fi)+'.')
        
        fig = plt.figure(figsize=(5,12))
        ax = fig.add_subplot(111)
        ax.clear()
        
        im = crop(c[fi])
        imshape = np.shape(im)
        extent = (0,imshape[1]*dx,0,imshape[0]*dx)
        ax.imshow(im,cmap='gray',origin='lower',extent=extent)
        
        # show the mask
        mask = masker.create_mask(fi)
        mask[mask==1] = np.nan
        
        my_cmap = plt.cm.get_cmap('hsv') # get a copy of the gray color map
        my_cmap.set_bad(alpha=1) # set how the colormap handles 'bad' values
        ax.imshow(mask,alpha=.2,origin='lower',cmap=my_cmap,extent=extent)
        
        '''
        Show the paths of the particles
        '''
        for part in list(particles.keys()):
            
            #part_path = t.copy()[(t['particle']==part)&(t['frame']<=fi)].set_index('frame')
            
            #part_path_all_vols = t.copy()[(t['particle']==part)].set_index('frame')
            part_path_all_vols = particles[part]
            part_path = part_path_all_vols[part_path_all_vols.index<=fi]
            
            if (part_path_all_vols.index.max()>=fi) & (part_path_all_vols.index.min()<=fi):
                ax.plot(part_path['x']*dx,part_path['y']*dx,linestyle='-',alpha=0.5,color='r',markersize=5)
                
                centroid = [part_path.loc[part_path.index==fi,'x']*dx,part_path.loc[part_path.index==fi,'y']*dx]
                if len(centroid[0]>0):
                    ax.plot(part_path.loc[part_path.index==fi,'x']*dx,part_path.loc[part_path.index==fi,'y']*dx,marker='o',alpha=0.5,color='r',markersize=5)            
                    from matplotlib.patches import Ellipse
                    ellipse = Ellipse(xy=centroid, width=part_path.loc[part_path.index==fi,'major_axis_length']*dx, height=part_path.loc[part_path.index==fi,'minor_axis_length']*dx, angle = np.pi - part_path.loc[part_path.index==fi,'orientation']*360/(2*np.pi),alpha=0.5,facecolor='None', edgecolor='blue',linewidth=3)
                    ax.add_patch(ellipse)
                    
        ax.set_ylim([imshape[0]*dx,0])
        ax.set_xlim([0,imshape[1]*dx])
                
        #ax.set_title('t = {:01.4f} s'.format(c.get_time(fi)))
        ax.set_title('frame '+str(fi)+' | t = {:04.2f}'.format(dt*fi)+' s')
        plt.show()
        plt.pause(.1)
        
        fig.savefig(imgs_dir+r'frame_'+format(fi,'04')+'.png')
        plt.close(fig)
#        
#    '''
#    Plot area of bubbles
#    '''
#    
#    plt.figure()
#    for pi in list(particles.keys()):
#        plt.plot(particles[pi].index,particles[pi]['perimeter'].rolling(window=3,center=True).median())
