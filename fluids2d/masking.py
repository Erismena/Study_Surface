# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 11:39:20 2017

@author: danjr
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os

class DetectedRectMasks:
    '''
    Class to detect where the mask over an image should be based on some region
    of the image.
    '''
    
    def __init__(self,motion_direction='vertical'):
        self.motion_direction=motion_direction
        return
    
    def create_mask_from_images(self,mask_name,cine):
        
        '''
        Select a region of the image: its boundaries are parallel with motion
        '''
        fig=plt.figure()
        ax=fig.add_subplot(111)
        
        ax.imshow(cine[0])
        plt.show()
        plt.pause(0.1)
        
        if self.motion_direction=='vertical':
            axis_of_thickness=0
            axis_of_dir=1
        elif self.motion_direction=='horizontal':
            axis_of_thickness=1
            axis_of_dir=0
            
        first_lim = plt.ginput(1)[0][axis_of_thickness]
        second_lim = plt.ginput(1)[0][axis_of_thickness]
        
        print(first_lim)
        
        indices=np.arange(first_lim,second_lim+1,dtype=int)
        print(indices)
        
        im_region = np.take(cine[0][:,:,0],indices=indices,axis=axis_of_dir)
        
        plt.figure()
        plt.imshow(im_region)
        
        return im_region
    
class MovingRectMasks:
    '''
    Class for a mask that moves, and potentially changes width, periodically.
    
    Methods to create a moving mask and apply it to a frame.
    '''
    
    def __init__(self,period_frames,motion_direction='vertical'):
        self.period_frames = period_frames
        self.motion_direction=motion_direction
        self.boundary_locs = dict()
        return
    
    def calibrate_from_images(self,mask_name,cine,frames_to_calibrate=25,crop_func=None):
        '''
        Let the user click on points on the mask over various frames to create
        a single mask.
        '''
        
        if crop_func is None:
            def crop_func(im): return im
        
        self.frame_shape = np.shape(crop_func(cine[0]))
        
        boundary_locs = pd.DataFrame(columns=['inner','outer'])
        
        if self.motion_direction=='vertical':
            axis_of_interest=1
        elif self.motion_direction=='horizontal':
            axis_of_interest=0
            
        frames_for_points = np.linspace(0,self.period_frames-1,num=frames_to_calibrate,dtype=int)
        
        fig=plt.figure()
        ax=fig.add_subplot(111)        
        for frame_to_show in frames_for_points:
 
            ax.clear()
            ax.imshow(crop_func(cine[frame_to_show]))
            ax.set_title('Frame '+str(frame_to_show))
            plt.show()
            plt.pause(0.1)
            
            outer_boundary = plt.ginput(1,timeout=-1)[0][axis_of_interest]
            inner_boundary = plt.ginput(1,timeout=-1)[0][axis_of_interest]
            
            boundary_locs.loc[frame_to_show,'outer'] = min([outer_boundary,inner_boundary])
            boundary_locs.loc[frame_to_show,'inner'] = max([outer_boundary,inner_boundary])
            
        boundary_locs.index = boundary_locs.index % int(self.period_frames)
        
        reindexed = boundary_locs[~boundary_locs.index.duplicated(keep='first')].copy().reindex(index=np.arange(0,self.period_frames)).astype(float)
        reindexed.loc[self.period_frames] = reindexed.loc[0,:]
        resampled = reindexed.interpolate(method='linear')
        
        self.boundary_locs[mask_name] = resampled
        
    def create_mask(self,fi):
        '''
        Determine the mask for a given frame number.
        '''
        f = fi % self.period_frames
        
        import numpy as np
        
        mask = np.ones(self.frame_shape)
        
        for mask_name in list(self.boundary_locs.keys()):
            
            lims = self.boundary_locs[mask_name].loc[f]
            
            if self.motion_direction=='horizontal':
                mask[:,lims.min():lims.max()] = 0
            elif self.motion_direction=='vertical':
                mask[lims.min():lims.max(),:] = 0
                
        return mask
        
    def mask_frame(self,fi,frame,method='zero'):        
        '''
        Given a frame and its number, construct the correct mask and use it to
        mask the frame.
        '''
        mask = self.create_mask(fi)                
        if method=='zero':
            masked_image = frame * mask
        elif method=='nan':
            masked_image = frame.copy()
            masked_image[mask==0] = np.nan
        return masked_image
    
    def get_num_masks(self):
        '''
        See how many masks are to be created with this masker (ie how many 
        grids there are to mask in each image).
        '''
        return len(self.boundary_locs)
    
    def save_self(self,directory,masker_name):
        pickle.dump(self,open(directory+masker_name+'.masker','wb'))
        
def load_masker(filename):
    return pickle.load(open(filename,'rb'))

def create_moving_rect_masker(parent_folder,cine_name,crop_func=None):
    '''
    UI to create a masker
    '''
    
    masker_filepath = parent_folder+cine_name+'.masker'
    if os.path.isfile(masker_filepath):
        print('Masker file already exists at '+str(masker_filepath)+'.')
        overwrite= input('Overwrite the above file? 1/0: ')
        if overwrite==0:
            print(type(overwrite))
            return
    
    cine= pims.open(parent_folder+cine_name+'.cine')
    
    period_frames=input(' -- Enter the periodicity in frames: ')
    num_frames=input(' -- Enter the number of frames with which to calibrate each mask: ')
    motion_direction=raw_input(' -- Enter the direction of motion: ')
    num_masks=input(' -- Enter the number of masks to create: ')
    mask_names = [str(i) for i in range(num_masks)]
    
    masker = MovingRectMasks(period_frames,motion_direction=motion_direction)
    
    for mask_name in mask_names:
        print('Calibrating mask '+str(mask_name)+'.')
        masker.calibrate_from_images(mask_name,cine,frames_to_calibrate=num_frames,crop_func=crop_func)
        
    masker.save_self(parent_folder,cine_name)
    
    
def create_masker_given_offset(original_masker,offset_frames):
    '''
    Create a new masker for a video identical to another for which there is 
    already a masker, except that the timing of the new video is offset.
    '''
    
    import copy
    new_masker = copy.deepcopy(original_masker)
    
    for mi in np.arange(new_masker.get_num_masks()):
        df = new_masker.boundary_locs[mi]
        df.index = df.index + offset_frames
        df.index = df.index % new_masker.period_frames
        new_masker.boundary_locs[mi] = df
        
    return new_masker

def mask_borders(im,left,right,bottom,top):
    masked = im.copy()
    masked[0:bottom,:] = 0
    masked[top:-1,:] = 0
    masked[:,0:left] = 0
    masked[:,right:-1] = 0
    return masked
    
if __name__ == '__main__':
    
    parent_folder = r'E:\Experiments_Stephane\Grid column\PIV_measurements\Hollow_grid_3\20170921\\'
    cine_name = r'PIV_sv_fps4k_grid3x3x10_withCap_midRegion_Cam_20861_Cine10'

    import pims
    cine = pims.open(parent_folder+cine_name+'.cine')
    
    create_moving_rect_masker(parent_folder,cine_name)
    
    masker = load_masker(parent_folder+cine_name+'.masker')
    
    masker.save_self(parent_folder,cine_name)
    
    fig = plt.figure()
    ax=fig.add_subplot(111)
    for i in np.arange(2000,3000,6):
        ax.clear()
        ax.imshow(masker.mask_frame(i,cine[i]))
        plt.show()
        plt.pause(0.2)