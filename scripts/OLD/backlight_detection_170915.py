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
import fluids2d.geometry
import fluids2d.backlight as bl
import fluids2d.piv as piv
import os.path
import pickle
from fluids2d.masking import MovingRectMasks

parent_folder = r'E:\Experiments_Stephane\Grid column\Backlight_bubbles\20170915\Backlight_bubble_fps1000_sv_A05mm_f10Hz_grid3x3x10_withCap\\'
#cine_filename = r'Backlight_bubble_fps1000_svCenterThird_mz100mm_A00mm_f00Hz_largeNeedle_grid3x4_Cam_20861_Cine1'


p_top = pickle.load(open(r'E:\Experiments_Stephane\Grid column\PIV_measurements\Hollow_grid_3\20170918\PIV_sv_fps4k_grid3x3x10_withCap_topRegion_Cam_20861_Cine6.pkl'))
#p_top.parent_folder = parent_folder
p_top.associate_flowfield()
ff = p_top.data.ff
cine = pims.open(p_top.cine_filepath)
im_shape = np.shape(cine[0])

g_orig = fluids2d.geometry.GeometryScaler(dx=p_top.dx,im_shape=np.shape(cine[0]),origin_pos=(0.28,-0.018),origin_units='m')
g_top = fluids2d.geometry.create_piv_scaler(p_top,g_orig)

p_mid = pickle.load(open(r'E:\Experiments_Stephane\Grid column\PIV_measurements\Hollow_grid_3\20170921\PIV_sv_fps4k_grid3x3x10_withCap_midRegion_Cam_20861_Cine6.pkl'))
#p_mid.parent_folder = parent_folder
p_mid.associate_flowfield()
ff = p_top.data.ff

g_orig = fluids2d.geometry.GeometryScaler(dx=p_mid.dx,im_shape=np.shape(cine[0]),origin_pos=(0.18,-0.018),origin_units='m')
g_mid = fluids2d.geometry.create_piv_scaler(p_mid,g_orig)

filenames = []
import os
for f in os.listdir(parent_folder):
    if f.endswith(".cine"):
        filenames.append(f[:-5]) # get rid of the ".avi"

flow_dir = 'horizontal'

dx_back = 0.28 / 685.7
dx_front = 0.28 / 950.2

dx = (dx_back+dx_front) / 2.


dt = 1./1000.

#for cine_filename in filenames:
for cine_filename in [r'Backlight_bubble_fps1000_sv_A05mm_f10Hz_grid3x3x10_withCap_Cam_20861_Cine10']:
    
    #if os.path.isfile(parent_folder+cine_filename+'_trackedParticles.pkl')==True:
    #    break
    
    cine_filepath = parent_folder+cine_filename+'.cine'
    c = pims.open(cine_filepath)
    
    g = fluids2d.geometry.GeometryScaler(dx=dx,origin_pos=[0,0],im_shape=np.shape(np.rot90(c[0])))
    
    #g.calibrate_image(crop(c[0]))
    
    #masker=fluids2d.masking.load_masker(parent_folder+cine_filename+'.masker')
    masker = fluids2d.masking.load_masker(parent_folder+r'Backlight_bubble_fps1000_sv_A05mm_f10Hz_grid3x3x10_withCap_Cam_20861_Cine1'+'.masker')
    
    '''
    Get the background by median filtering along the frame axis
    '''
    frames_for_bg = np.arange(0,300,8)
    bg_image = bl.construct_bg_image(c,masker,frames_for_bg)    
    
    fig,ax = plt.subplots(2,2,figsize=(8,8),sharex=True,sharey=True)
    ax=ax.flatten()
    plt.tight_layout()
    
    df_all = pd.DataFrame()
    frames = np.arange(0,len(c),10)
    for ci in frames:
        
        
        
        print('Reading frame '+str(ci)+' of '+str(max(frames))+'.')
        
        '''
        Load the image
        '''
        im_orig = c[ci]        

        '''
        Background subtract
        '''
        im = (im_orig - bg_image)
        
        '''
        Median filter the bg subtracted image.
        '''
        im_filt = scipy.ndimage.filters.median_filter(im,size=3)    

        '''
        Binarize and flip so strong signal is 1
        '''
        im_filt = np.array(im_filt<-20,dtype=int) # Bubble points are now 1
        
        '''
        Apply the mask
        '''
        im_filt = masker.mask_frame(ci,im_filt)
        im_filt = fluids2d.masking.mask_borders(im_filt,0,1100,120,480)
        
        '''
        Fill the holes
        '''
        from scipy.ndimage.morphology import binary_fill_holes
        filled = binary_fill_holes(im_filt)
        
        '''
        Find the regions and get their properties
        '''
        df = bl.filled2regionpropsdf(filled,min_area=20,frame=ci)                
        df_all = pd.concat([df_all,df])
        
        if np.random.uniform() < 9:
            
            [a.clear() for a in ax]
            
            ax[0].imshow(np.rot90(im_orig),cmap='gray',extent=g.im_extent)
            ax[0].set_title('Raw image')
            
            ax[1].imshow(np.rot90(im),cmap='gray',extent=g.im_extent)
            ax[1].set_title('Background subtracted')
            
            ax[2].imshow(np.rot90(im_filt),cmap='gray',extent=g.im_extent)
            ax[2].set_title('Binarized and masked')
            
            ax[3].imshow(np.rot90(filled),extent=g.im_extent)
            ax[3].set_title('Holes filled in')
            
            ax[0].imshow(np.linalg.norm(np.nanmean(p_top.data.ff,axis=0),ord=2,axis=2),vmin=0.0,vmax=0.2,alpha=0.2,extent=g_top.im_extent,cmap='jet')
            #ax[0].imshow(np.linalg.norm(np.nanmean(p_mid.data.ff,axis=0),ord=2,axis=2),vmin=0.0,vmax=0.2,alpha=0.2,extent=g_mid.im_extent,cmap='jet')
            
            [a.set_ylim([0,.5]) for a in ax]
            [a.set_xlim([0,.25]) for a in ax]
            
            #if df.empty==False:
            #    [axi.scatter(df['x'],df['y'],color='r',s=20,alpha=0.5) for axi in ax.reshape(-1)]
        
        plt.show()
        plt.pause(1)
        
    '''
    Track the particles
    '''
    import trackpy as tp
    t = tp.link_df(df_all,10,memory=11,adaptive_stop=2)
    
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
    
    #import pickle
    #pickle.dump(particles, open(parent_folder+cine_filename+'_trackedParticles.pkl', "wb"))
    
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
    
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir)
    
    
    for fi in frames:
        
        print('Working on frame '+str(fi)+'.')
        
        figsize_pix = np.shape(im_orig)
        figwidth_inches = 12.
        figheight_inches = figwidth_inches * float(figsize_pix[0])/float(figsize_pix[1])
        dpi = float(figsize_pix[1]) / figwidth_inches
        fig = plt.figure(figsize=(figwidth_inches,figheight_inches),dpi=dpi)
        ax = fig.add_axes([0., 0., 1., 1.])
        ax.clear()
        #ax.set_axis_off()
        
        im = c[fi]
        imshape = np.shape(im)
        extent = (0,imshape[1]*dx,0,imshape[0]*dx)
        ax.imshow(im,cmap='gray',origin='lower',extent=extent)
        ax.set_yticks([])
        ax.get_xaxis().set_tick_params(direction='in',pad=-20,colors='b')
        
        
        
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
                    ax.text(part_path.loc[part_path.index==fi,'x']*dx,part_path.loc[part_path.index==fi,'y']*dx,str(int(part)))
                    #ax.plot(part_path.loc[part_path.index==fi,'x']*dx,part_path.loc[part_path.index==fi,'y']*dx,marker='o',alpha=0.5,color='r',markersize=5)            
                    from matplotlib.patches import Ellipse
                    ellipse = Ellipse(xy=centroid, width=part_path.loc[part_path.index==fi,'major_axis_length']*dx, height=part_path.loc[part_path.index==fi,'minor_axis_length']*dx, angle = np.pi - part_path.loc[part_path.index==fi,'orientation']*360/(2*np.pi),alpha=0.5,facecolor='None', edgecolor='blue',linewidth=3)
                    ax.add_patch(ellipse)
                    
        ax.set_ylim([imshape[0]*dx,0])
        ax.set_xlim([0,imshape[1]*dx])
        
        ax.text(.35,.02,'t = '+'{:04.3f}'.format(fi*dt)+' s',color='b',fontsize=20)
        ax.text(.05,.02,'A = 5 mm | f = 5 Hz',color='b',fontsize=20)
        ax.text(.05,.01,cine_filename,color='b',fontsize=8)
                
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
