# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 12:18:54 2017

@author: danjr
"""

import pims
import numpy as np
import openpiv.process
import openpiv.validation
import openpiv.filters
import matplotlib.pyplot as plt
import matplotlib.patches
import pickle
import pandas as pd

class PIVDataProcessing:
    '''
    Class for PIV computations for a cine.
    '''
    
    def __init__(self,parent_folder,cine_name,window_size=32,overlap=16,search_area_size=64,frame_diff=1,name_for_save=None,maskers=None,crop_lims=None,dx=1,dt_orig=1):
        
        # Metadata
        self.cine_filepath = parent_folder+cine_name+'.cine'
        self.parent_folder=parent_folder
        self.cine_name=cine_name
        #self.num_frames_orig = len(pims.open(self.cine_filepath))
        self.maskers=maskers
        self.crop_lims = crop_lims
        self.cine_frame_shape=None
        
        # PIV Parameters
        self.window_size=window_size
        self.overlap=overlap
        self.search_area_size=search_area_size
        self.frame_diff = frame_diff # pairs of frames separated by how many frames
        
        # Scaling, for reference -- results are still stored in [pixels / frame] !
        self.dx=dx
        self.dt_orig=dt_orig # "original" dt - between frames in the cine
        self.origin_pos=None
        
        # For saving the results
        self.flow_field_res_filepath = None # will store a path to the numpy array with the flow field results        
        if name_for_save is None:
            self.name_for_save = self.cine_name
        else:
            self.name_for_save = name_for_save
            
    def save(self):
        self.ff = None # So the actual flowfield isn't stored in this pickled object
        pickle.dump(self,open(self.parent_folder+self.name_for_save+'.pkl','wb'))

    def process_frame(self,frame_a,frame_b,s2n_thresh=1.3):
        frame_a = frame_a.astype(np.int32)
        frame_b = frame_b.astype(np.int32)
        
        u,v,sig2noise = openpiv.process.extended_search_area_piv( frame_a, frame_b, window_size=self.window_size, overlap=self.overlap, dt=1, search_area_size=self.search_area_size,sig2noise_method='peak2peak' )
        u, v, mask = openpiv.validation.sig2noise_val( u, v, sig2noise, threshold = s2n_thresh )
        
        return u,v
        
    def run_analysis(self,a_frames=None,save=True,s2n_thresh=1.3):
        
        c=pims.open(self.cine_filepath)
        self.cine_frame_shape = c.frame_shape
        
        if a_frames is None:
            a_frames = np.arange(0,len(c)-self.frame_diff)
        
        '''
        Define function to crop the image, based on the limits in crop_lims
        ''' 
        crop_lims = self.crop_lims
        if crop_lims is not None:
#            def crop(im):
#                im[0:crop_lims[2],:] = 0
#                im[crop_lims[3]:-1,:] = 0
#                im[:,0:crop_lims[0]] = 0
#                im[:,crop_lims[1]:-1] = 0
#                
#                plt.figure()
#                plt.imshow(im)
#                plt.show()
#                
#                return im
            def crop(im): return im[crop_lims[0]:crop_lims[1],crop_lims[2]:crop_lims[3]]                
        else:
            def crop(im): return im
        
        flow_field = init_flowfield(len(a_frames),np.shape(crop(c[0])),self.window_size,self.overlap)        
        
        '''
        Store some processing parameters
        '''
        self.window_coordinates = openpiv.process.get_coordinates(np.shape(crop(c[0])),self.window_size,self.overlap)
        self.s2n_thresh=s2n_thresh # threshold used in PIV analysis
        self.a_frames=a_frames # which original frames are the "a" frames
        self.dt_ab = self.dt_orig*self.frame_diff # dt between frames a and b
        self.dt_frames = self.dt_orig * np.median(np.diff(self.a_frames)) # ASSUMES THAT ALL A FRAMES ARE EVENLY SPACED!
        try:
            self.a_frame_times = [c.frame_time_stamps[i] for i in a_frames] # time of the a frames
        except:
            print('frame time stamps not available')
            self.a_frame_times = a_frames*self.dt_frames
        
        for aii,ai in enumerate(a_frames):
            print('file '+str(self.name_for_save)+', frame a: '+str(ai))
            
            # get the two frames
            #frame_a = crop(c[ai].astype(np.int32))
            #frame_b = crop(c[ai+self.frame_diff].astype(np.int32))
            
            frame_a = c[ai].astype(np.int32)
            frame_b = c[ai+self.frame_diff].astype(np.int32)
            
            if self.maskers is not None:
                for masker in self.maskers:
                    frame_a = masker.mask_frame(ai,frame_a).astype(np.int32)
                    frame_b = masker.mask_frame(ai,frame_b).astype(np.int32)
                    
            frame_a = crop(frame_a)
            frame_b = crop(frame_b)
            
            # run the PIV analysis
            u,v,sig2noise = openpiv.process.extended_search_area_piv( frame_a, frame_b, window_size=self.window_size, overlap=self.overlap, dt=1, search_area_size=self.search_area_size,sig2noise_method='peak2peak' )
            u, v, mask = openpiv.validation.sig2noise_val( u, v, sig2noise, threshold = s2n_thresh )
            
            # store the velocity fields
            flow_field[aii,:,:,0] = u
            flow_field[aii,:,:,1] = v
            
            if (save==True)&(ai%5000==0): # save every some frames, just in case
                np.save(self.parent_folder+self.name_for_save+'_flowfield.npy',flow_field)
                self.flow_field_res_filepath = self.parent_folder+self.name_for_save+'_flowfield.npy'
                self.save()
            
        if save==True:
            np.save(self.parent_folder+self.name_for_save+'_flowfield.npy',flow_field)
            self.flow_field_res_filepath = self.parent_folder+self.name_for_save+'_flowfield.npy'
            self.save()
            
        return flow_field
    
    def load_flowfield(self):
        '''
        Return an instance of the PIVDataAnalysis class containing the SCALED
        dataset.
        '''
        flow_field = PIVDataAnalysis(load_scaledflowfield(self.parent_folder+self.name_for_save+'_flowfield.npy',self.dx,self.dt_ab))
        print('dx : ')
        print(self.dx)
        print('dt_ab: ')
        print(self.dt_ab)
        return flow_field
    
    def associate_flowfield(self):
        '''
        So the flowfield can be accessed through the .data attribute.
        
        If this object is going to be saved, the .data attribute will first be
        cleared so as to not save the actual flowfield with the metadata 
        contained in this object.
        '''
        self.data = self.load_flowfield()
    
#    def indx2time(self,idx):
#        c=pims.open(self.cine_filepath)
#        time = [c]
    
    
class PIVDataAnalysis:
    '''
    Class to work with processed PIV data. Data is ALREADY SCALED when 
    initialized!
    
    The main attribute is "ff" (flowfield), a 4D numpy array:
        local_inst_velocity_component = ff[frame,row,column,direction]
    '''
    def __init__(self,ff):
        '''
        ff here is ALREADY SCALED given dx and dt!
        '''
        self.ff = ff
        
    def get_frame(self,i):
        '''
        Return one frame from the flowfield array (4D -> 3D)
        '''
        return self.ff[i,:,:,:]
    
    def show_frame(self,i,bg='speed',ax=None):
        frame = self.get_frame(i)
        ax=show_frame(frame[:,:,0],frame[:,:,1],bg=bg,ax=ax)
        return ax
        
    def average_frames(self,frames):
        return np.nanmean(self.ff[frames,:,:,:],axis=0)
    
class TurbulenceParams:
    '''
    Class for mapping forcing conditions to turbulence parameters
    '''
    
    def __init__(self,metadata,data):
        '''
        metadata is a dataframe indexed with the dict data. it has the forcing
        conditions associated with each entry in data
        '''

###############################################################################

'''
FUNCTIONS FOR DATA I/O
'''

def read_control_csv(csv_filepath):
    '''
    Read the .csv file that controls which cases are to be analyzed.
    '''
    
    all_cases = pd.read_csv(csv_filepath)
    all_cases = all_cases.loc[all_cases['use_now']==1]
    
    amplitudes = all_cases['A'].unique()
    freqs = all_cases['freq'].unique()

def combine_linear_data(data,geometries,dy=0.001):
    '''
    List of numpy arrays and list of corresponding geometries. The arrays will
    be stacked vertically with some interpolation.
    '''
    
    from scipy.interpolate import interp1d
    
    '''
    First find the vertical extent of all the geometries.
    '''
    min_y = np.nan
    max_y = np.nan
    for g in geometries:
        min_y = np.nanmin([min_y,np.min(g.y)])
        max_y = np.nanmax([max_y,np.max(g.y)])
        
    print(min_y)
    print(max_y)
    new_y = np.arange(min_y,max_y,dy)
    new_f = np.nan * new_y
    
    for di in np.arange(len(data)):
        d = data[di]
        g = geometries[di]
        
        print(np.shape(d))
        print(np.shape(g.y))
        
        min_y = np.min(g.y)
        max_y = np.max(g.y)
        
        interpolated_y = new_y[(new_y>=min_y)&(new_y<=max_y)]
        interpolator = interp1d(g.y,d)
        interpolated_f = interpolator(interpolated_y)
        
        #new_f[(new_y>=min_y)&(new_y<=max_y)] = np.nanmean([new_f[(new_y>=min_y)&(new_y<=max_y)],interpolated_f])
        new_f[(new_y>=min_y)&(new_y<=max_y)] = interpolated_f
        
    return new_y,new_f

def rotate_data_90(ff):
    '''
    Rotate the images in a 4d matrix 90 deg ccw.
    
    Can't use the 'axes' parameter in numpy.rot90 since that version of numpy
    is not compatible with trackpy.
    '''
    
    # first move the time dimension out of the way
    ff = np.swapaxes(ff,0,1)
    ff = np.swapaxes(ff,1,2)
    
    # rotate about the plane of the first two axes
    ff = np.rot90(ff)
    
    # put back the time axis at the front
    ff = np.swapaxes(ff,2,1)
    ff = np.swapaxes(ff,1,0)
    
    # flip the velocity components
    ff = ff[:,:,:,[1,0]]
    
    # correct the direction of the u velocity
    ff[:,:,:,0] = -1*ff[:,:,:,0]
    
    return ff

def init_flowfield(num_frames,frame_shape,window_size,overlap):
    '''
    Initialize the 4d numpy array to store the flow field
    '''
    field_shape = openpiv.process.get_field_shape(frame_shape,window_size,overlap)
    flow_field = np.zeros([num_frames,field_shape[0],field_shape[1],2]) # [frame, row, column, velocity component]
    return flow_field

def scale_flowfield(ff,dx,dt_ab):
    '''
    Convert [pixels/frame] to [m/s]
    '''
    return ff * dx / dt_ab

def load_scaledflowfield(filepath,dx,dt):
    '''
    Load the flowfield stored at filepath, and scale it given dx and dt.
    
    Returns a numpy array.
    '''
    ff=np.load(filepath)
    return scale_flowfield(ff,dx,dt)

def load_processed_PIV_data(parent_folder,name):
    '''
    Load the pickled metadata object and call the .associate_flowfield method
    so the stored flowfield is in the .ff attribute.
    '''
    p = pickle.load(open(parent_folder+name+'.pkl','rb'))
    p.associate_flowfield() # the numpy data can now be accessed with p.data.ff[...]
    return p

def clip_flowfield(ff,thresh):
    ff2 = ff.copy()
    #ff2[ff>thresh] = thresh
    #ff2[ff<-1*thresh] = -1*thresh
    ff2[ff>thresh] = np.nan
    ff2[ff<-1*thresh] = np.nan
    return ff2

def fill_nans(x):
    '''
    Use pandas to forward fill then backfill nan values in a series
    '''
    return pd.Series(x).fillna(method='ffill').fillna(method='bfill').values

def fill_nans_3d(x):
    '''
    Forward/backward fill nans in a 3d array along the first axis
    '''
    num_rows = np.shape(x)[1]
    num_cols = np.shape(x)[2]
    
    for yi in range(num_rows):
        for xi in range(num_cols):            
            x[:,yi,xi] = fill_nans(x[:,yi,xi])
            
    return x

def fill_nans_nd(x):
    s = np.shape(x)
    
    to_it = s[1:]
    idxs = [range(i) for i in to_it]
    
    for i in zip(idxs):
        print(i)
        print(np.shape(x[:,i]))
        x[:,i] = fill_nans(x[:,i])
        
def check_concatenation(p):
    
    first_job_name = p.cine_name+'_job0'
    p_0 = pickle.load(open(p.parent_folder+first_job_name+'.pkl'))
    p_0.parent_folder = p.parent_folder
    p_0.name_for_save = first_job_name
    p_0.associate_flowfield()
    
    is_good = p.data.ff[0,0,0,0]==p_0.data.ff[0,0,0,0]
    
    return is_good
    
    

###############################################################################

'''
FUNCTIONS FOR FLOWFIELD ANALYSIS
'''

def compute_frame_shear(u,v):
    uxx = np.gradient(np.gradient(u,axis=1),axis=1)
    vyy = np.gradient(np.gradient(v,axis=0),axis=0)
    shear = 0.5 * ( uxx**2 + vyy**2 )
    return shear

def compute_gradients(ff):
    '''
    Given a (4D) 3D flowfield, return a (5D) 4D gradient field indexed by
    [(time,)row,column,velocity_dir,gradient_dir]
    '''
    
    ff_gradients = np.zeros(np.shape(ff)+(2,))
    
    for i in [0,1]:
        for j in [0,1]:
            ff_gradients[...,i,j] = np.gradient(ff[...,i],axis=j)
    
    return ff_gradients

def compute_turbulent_fluctuations(ff):
    mean_flow = np.nanmean(ff,axis=0)
    return ff - mean_flow

def basic_field_calcs(ff):    
    '''
    Return a dict of common fields (mean flow, fluctuations, etc)
    '''
    
    results = {}
    
    results['mean_flow'] =np.nanmean(ff,axis=0)
    results['fluc'] = ff-mean_flow
    results['u_rms'] = np.sqrt( np.nanmean( (results['fluc'][:,:,:,0])**2,axis=0) + np.nanmean( (results['fluc'][:,:,:,1])**2,axis=0) )
    results['inst_speed'] = np.linalg.norm(ff,ord=2,axis=3)
    results['meanflow_speed'] = np.sqrt((results['mean_flow'][:,:,0])**2 + (results['mean_flow'][:,:,1])**2)    
    results['dudx'] = np.gradient(results['fluc'][:,:,:,0],axis=1)
    results['dvdy'] = np.gradient(results['fluc'][:,:,:,1],axis=2) 

    return results


###############################################################################

'''
FUNCTIONS FOR VISUALIZATION
'''

def add_fieldimg_to_ax(ff_reduced,g,ax,time=None,slice_dir=None,vel_dir=None,cmap_other=None,vmin=-0.2,vmax=0.2):
    '''
    Add a 2d image to an axes with the appropriate scaling.
    
    Use for either a snapshot or time average, or spatially-reduced composite 
    image with appropriate slice_dir defined
    '''
    
    if vel_dir=='horizontal':
        cmap = 'PuOr'
    elif vel_dir=='vertical':
        cmap = 'seismic'
    elif vel_dir==None:
        cmap = 'viridis'
        
    if cmap_other is not None:
        cmap = cmap_other
    
    if slice_dir=='vertical':
        ax.imshow(ff_reduced.transpose(),origin='upper',vmin=vmin,vmax=vmax,aspect='auto',cmap=cmap,extent=[0,max(time),g.im_extent[2],g.im_extent[3]])
    elif slice_dir=='horizontal':
        ax.imshow(ff_reduced,vmin=vmin,vmax=vmax,aspect='auto',cmap=cmap,extent=[g.im_extent[0],g.im_extent[1],0,max(time)],origin='lower')
    elif slice_dir==None:
        ax.imshow(ff_reduced,vmin=vmin,vmax=vmax,cmap=cmap,extent=[g.im_extent[0],g.im_extent[1],g.im_extent[2],g.im_extent[3]],origin='upper')

def composite_image_plots(ff,g,time,row_lims,col_lims,vmin=-0.2,vmax=0.2):
    
    '''
    Velocity along center column
    '''    
    
    fig_column = plt.figure(figsize=(16,9))
    ax1=fig_column.add_subplot(1,2,1)
    ax2=fig_column.add_subplot(1,2,2) 
    
    #ff = ff - np.nanmean(ff,axis=0)

    add_fieldimg_to_ax(np.nanmean(ff[:,:,col_lims[0]:col_lims[1]+1,0],axis=2),g,ax1,time=time,slice_dir='vertical',vel_dir='horizontal',vmin=vmin,vmax=vmax)
    add_fieldimg_to_ax(np.nanmean(ff[:,:,col_lims[0]:col_lims[1]+1,1],axis=2),g,ax2,time=time,slice_dir='vertical',vel_dir='vertical',vmin=vmin,vmax=vmax)
    
    [a.set_xlabel('Time [s]') for a in [ax1,ax2]]
    ax1.set_ylabel('Position along vertical column [m]')
    ax1.set_title('Horizontal velocity')
    ax2.set_title('Vertical velocity')    
        
    '''
    Velocity along center span
    '''
        
    fig_row = plt.figure(figsize=(16,9))
    ax1=fig_row.add_subplot(1,2,1)
    ax2=fig_row.add_subplot(1,2,2)
    
    add_fieldimg_to_ax(np.nanmean(ff[:,row_lims[0]:row_lims[1]+1,:,0],axis=1),g,ax1,time=time,slice_dir='horizontal',vel_dir='horizontal',vmin=vmin,vmax=vmax)
    add_fieldimg_to_ax(np.nanmean(ff[:,row_lims[0]:row_lims[1]+1,:,1],axis=1),g,ax2,time=time,slice_dir='horizontal',vel_dir='vertical',vmin=vmin,vmax=vmax)
    [a.set_xlabel('Position along horizontal span [m]') for a in [ax1,ax2]]
    ax1.set_ylabel('Time [s]')
    ax1.set_title('Horizontal velocity')
    ax2.set_title('Vertical velocity')
    
    return fig_column,fig_row

def vertical_percentile_distributions(ff,g,time,row_lims,col_lims,other_percentiles=[10,25,75,90]):
    
    fig=plt.figure(figsize=(9,4))
    ax1=fig.add_subplot(131)
    ax2=fig.add_subplot(132,sharey=ax1,sharex=ax1)
    ax3=fig.add_subplot(133,sharey=ax1)
   
    [a.axvline(0,color='gray',alpha=0.3) for a in [ax1,ax2,ax3]]
    
    
    ax1.plot(np.nanmean(np.nanpercentile(ff[:,:,col_lims[0]:col_lims[1],0],50,axis=0),axis=1),g.y,color='k',alpha=1)
    for other_percentile in other_percentiles:
        ax1.plot(np.nanmean(np.nanpercentile(ff[:,:,col_lims[0]:col_lims[1],0],other_percentile,axis=0),axis=1),g.y,color='k',alpha=0.5)
    
    ax2.plot(np.nanmean(np.nanpercentile(ff[:,:,col_lims[0]:col_lims[1],1],50,axis=0),axis=1),g.y,color='k',alpha=1)
    for other_percentile in other_percentiles:
        ax2.plot(np.nanmean(np.nanpercentile(ff[:,:,col_lims[0]:col_lims[1],1],other_percentile,axis=0),axis=1),g.y,color='k',alpha=0.5)
        
    mean_flow=np.nanmean(ff,axis=0)
    fluc = ff-mean_flow
    u_rms = np.sqrt( np.nanmean( (fluc[:,:,:,0])**2,axis=0) + np.nanmean( (fluc[:,:,:,1])**2,axis=0) )    
    
    ax3.plot(np.nanmean(np.linalg.norm(mean_flow[:,col_lims[0]:col_lims[1],:],ord=2,axis=2),axis=1),g.y,color='k',alpha=0.5,label='''$\sqrt{\overline{U}^2 + \overline{V}^2}$''')
    ax3.plot(np.nanmean(u_rms[:,col_lims[0]:col_lims[1]],axis=1),g.y,color='k',alpha=1,label='''$\sqrt{\overline{(u-\overline{U})^2}+\overline{(v-\overline{V})^2}}$''')
    
    ax1.set_title('u distribution')
    ax2.set_title('v distribution, mean is'+str(np.nanmean(np.nanmean(np.nanpercentile(ff[:,:,col_lims[0]:col_lims[1],1],50,axis=0),axis=1))))
    ax3.set_title('fluctuations')
    ax3.legend()
    ax1.set_ylabel('vertical position [m]')
    [a.set_xlabel('velocity [m/s]') for a in [ax1,ax2,ax3]]
        
    return fig

    

def plot_both_components(vel,time=None):
    fig=plt.figure(figsize=(9,5))
    ax1=fig.add_subplot(121)
    ax2=fig.add_subplot(122)
    
    mean_flow = np.nanmean(vel,axis=0)
    print(np.shape(mean_flow))
    
    if time is None:
        time = np.arange(0,np.shape(vel)[0])
    
    ax1.axhline(0,color='k',alpha=0.2)
    ax1.plot(time,vel[:,0],'--',label='u',color='gray',lw=0.8)
    ax1.plot(time,vel[:,1],'-.',label='v',color='gray',lw=0.8)
    ax1.plot(time,np.sqrt((vel[:,0]-mean_flow[0])**2+(vel[:,1]-mean_flow[1])**2),color='r',lw=1,label='''$\sqrt{ u'^2 + v'^2}$''')
    ax1.plot(time,np.sqrt((vel[:,0])**2+(vel[:,1])**2),color='k',lw=2,label='''$\sqrt{ u^2 + v^2}$''')
    ax1.legend()
    
    ax2.axvline(0,color='k',alpha=0.2)
    ax2.axhline(0,color='k',alpha=0.2)
    ax2.scatter(vel[:,0],vel[:,1],c=time,alpha=0.3)
    ax2.set_ylabel('v')
    ax2.set_ylabel('u')

def overlay_vectors_frame(p,frame):
    
    fig = plt.figure()
    ax=fig.add_subplot(111)
    
    indx_in_cine = p.a_frames[frame]
    
    cine=pims.open(p.cine_filepath)
    ax.imshow(cine[indx_in_cine],cmap='gray')
    
    '''
    Show the mask
    '''
    mask = p.maskers[0].create_mask(indx_in_cine)    
    ax.imshow(mask,alpha=0.5)

    
def show_mean_flows(ff):
    fig=plt.figure(figsize=(8,8))
    ax1=fig.add_subplot(2,2,1)
    ax2=fig.add_subplot(2,2,2)
    ax3=fig.add_subplot(2,2,3)
    ax4=fig.add_subplot(2,2,4)
    
    ax1.imshow(np.nanmean(ff[:,:,:,0],axis=0))
    ax2.imshow(np.nanmean(ff[:,:,:,1],axis=0))
    ax3.imshow(np.nanstd(ff[:,:,:,0],axis=0))
    ax4.imshow(np.nanstd(ff[:,:,:,1],axis=0))
    
def show_frame(u,v,bg='speed',ax=None):
    if ax is None:
        fig=plt.figure(); ax=fig.add_subplot(111)
        
    if bg=='speed':
        bg = np.sqrt(u**2+v**2)
    elif bg=='shear':
        bg = compute_frame_shear(u,v)
                                    
    ax.matshow(bg,vmin=0)
    ax.quiver(u,v,color='white',alpha=0.5)
    plt.show()
    plt.pause(0.1)
    return ax

def convert_ax_to_cbar(ax,cmap,vmin,vmax):
        
    import matplotlib as mpl
    pos = ax.get_position().bounds
    ax.set_position([pos[0],pos[1]+pos[3]/2,pos[2],(pos[3])/5])
    
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    return cb1,ax



class PIVComparisonsFigures:
    '''
    Class to create the figures needed for each comparison of PIV cases
    '''
    
    def __init__(self,nrows,ncols,figsize=(11,8),max_speed=1.0,vmin=-.5,vmax=0.5,legend_axi=None):
        
        self.nrows = nrows
        self.ncols = ncols
        self.figsize = figsize
        self.legend_axi = legend_axi
        self.left_axs = np.arange(0,nrows*ncols,ncols)
        self.bottom_axs = np.arange(ncols*(nrows-1),ncols*nrows,1)
        
        self.max_speed = max_speed
        self.vmin = vmin
        self.vmax = vmax
        
        self.fig_names = ['mean','u_rms','intns','U','V','isotropy']
        
        self.figs = {}
        self.axs = {}
        for name in self.fig_names:
            self.figs[name],self.axs[name] = plt.subplots(self.nrows,self.ncols,figsize=self.figsize)
            self.axs[name] = self.axs[name].flatten()
            
    def add_case(self,ai,ff,g,time):
        
        mean_flow=np.nanmean(ff,axis=0)
        fluc = ff-mean_flow
        u_rms = np.sqrt( np.nanmean( (fluc[:,:,:,0])**2,axis=0) + np.nanmean( (fluc[:,:,:,1])**2,axis=0) )
        meanflow_speed = np.sqrt((mean_flow[:,:,0])**2 + (mean_flow[:,:,1])**2)
        isotropy_rms = np.sqrt( np.nanmean( (fluc[:,:,:,0])**2,axis=0)) / np.sqrt( np.nanmean( (fluc[:,:,:,1])**2,axis=0))
        
        # Add the images to the figures
        add_fieldimg_to_ax(np.log10(u_rms / meanflow_speed),g,self.axs['intns'][ai],time=time,slice_dir=None,vel_dir=None,vmin=-1,vmax=1,cmap_other='coolwarm')
        add_fieldimg_to_ax(meanflow_speed,g,self.axs['mean'][ai],time=time,slice_dir=None,vel_dir=None,vmin=0,vmax=self.max_speed)
        add_fieldimg_to_ax(u_rms,g,self.axs['u_rms'][ai],time=time,slice_dir=None,vel_dir=None,vmin=0,vmax=self.max_speed)    
        add_fieldimg_to_ax(mean_flow[:,:,0],g,self.axs['U'][ai],vel_dir='horizontal',vmin=self.vmin,vmax=self.vmax)
        add_fieldimg_to_ax(mean_flow[:,:,1],g,self.axs['V'][ai],vel_dir='vertical',vmin=self.vmin,vmax=self.vmax)
        add_fieldimg_to_ax(np.log2(isotropy_rms),g,self.axs['isotropy'][ai],slice_dir=None,vel_dir=None,vmin=-2,vmax=2,cmap_other='PRGn')
        
        # update the limits
        [g.set_axes_limits(a[ai]) for a in list(self.axs.values())]
        
        # clean up the axis labels etc
        if ai not in self.left_axs:
            [a[ai].yaxis.set_ticklabels([]) for a in list(self.axs.values())]
        else:
            [a[ai].set_ylabel('y [m]') for a in list(self.axs.values())]
            
        if ai not in self.bottom_axs:
            [a[ai].xaxis.set_ticklabels([]) for a in list(self.axs.values())]
        else:
            [a[ai].set_xlabel('x [m]') for a in list(self.axs.values())]
            
    def update_limits(self,ai,xlims,ylims):
        [a[ai].set_xlim(xlims) for a in list(self.axs.values())]
        [a[ai].set_ylim(ylims) for a in list(self.axs.values())]
            
    def add_rect(self,ai,x,y,width,height,color,ls,lw=2):
        for a in list(self.axs.values()):
            r = matplotlib.patches.Rectangle([x,y],width,height,edgecolor=color,ls=ls,lw=lw,fill=False)
            a[ai].add_patch(r)
            
    def add_text(self,ai,x,y,text):
        [a[ai].text(x,y,text,ha='right',va='top',color='k',fontsize=10) for a in list(self.axs.values())]
        
    def tight_layout(self):
        [f.tight_layout() for f in list(self.figs.values())]
        
    def add_legends(self):
        
        # intensity
        cbar,ax_cbar = convert_ax_to_cbar(self.axs['intns'][self.legend_axi],'coolwarm',-1,1)
        cbar.set_ticks([-1,0,1])
        cbar.set_ticklabels([0.1,1,10])
        ax_cbar.set_title('turbulence intensity')
        
        # mean flow speed
        cbar,ax_cbar = convert_ax_to_cbar(self.axs['mean'][self.legend_axi],'viridis',0,self.max_speed)
        ax_cbar.set_title('mean flow speed [m/s]')
        
        # u_rms
        cbar,ax_cbar = convert_ax_to_cbar(self.axs['u_rms'][self.legend_axi],'viridis',0,self.max_speed)
        ax_cbar.set_title('$u_\mathrm{rms}$ [m/s]')
        
        # mean U
        cbar,ax_cbar = convert_ax_to_cbar(self.axs['U'][self.legend_axi],'PuOr',self.vmin,self.vmax)
        ax_cbar.set_title('$\overline{U}$ [m/s]')
        
        # mean V
        cbar,ax_cbar = convert_ax_to_cbar(self.axs['V'][self.legend_axi],'seismic',self.vmin,self.vmax)
        ax_cbar.set_title('$\overline{V}$ [m/s]')
        
        # isotropy
        cbar,ax_cbar = convert_ax_to_cbar(self.axs['isotropy'][self.legend_axi],'PRGn',-2,2)
        cbar.set_ticks([-2,0,2])
        cbar.set_ticklabels([.25,1,4])
        ax_cbar.set_title('''$\sqrt{ \overline{u'^2} } / \sqrt{ \overline{w'^2} }$''')
        
    def remove_axes(self,ai):
        [a[ai].set_axis_off() for a in list(self.axs.values())]
        
    def save_figs(self,figfolder,prefix):
        for key in list(self.figs.keys()):
            fig = self.figs[key]
            fig.savefig(figfolder+prefix+'_'+key+'.pdf')
        

if __name__ == '__main__':
    
    '''
    Run the module directly to do a single PIV case.
    '''
    
    #parent_folder = r'C:\Users\Luc Deike\highspeed_data\171105\PIV_sv_vp_fps8000_makrozeiss100mm_without_balloon\\'
    #parent_folder = r'D:\high_speed_data\171222\\'
    #cine_name = 'piv_4pumpsDiffusers_topDown_sched-alwaysOn_fullView_fps4000_dur4s'
    
    parent_folder = r'C:\Users\Luc Deike\data_comp3_C\180323\\'
    cine_name = r'piv_scanning_bubblesRising_angledUp_galvo100Hz_galvoAmp500mV_fps1000'

    
#    parent_folder = r'C:\Users\Luc Deike\highspeed_data\171107\PIV_tv_hp_fps10000_makrozeiss100mm_with_balloon_1\\'
#    parent_folder = r'C:\Users\Luc Deike\highspeed_data\171105\PIV_sv_vp_fps8000_makrozeiss100mm_without_balloon\\'

     
    #cine_name = r'PIV_sv_vp_fps8000_makrozeiss100mm_with_balloon__Cam_20861_Cine1'

    
    #cine_nums = np.arange(1,11)
    
    #cine_names = [r'PIV_sv_vp_fps8000_makrozeiss100mm_without_balloon__Cam_20861_Cine'+str(n) for n in cine_nums]   
    #cine_names = [r'PIV_tv_hp_fps10000_makrozeiss100mm_with_balloon_4_Cam_20861_Cine'+str(n) for n in cine_nums]      
    #cine_names = [r'PIV_tv_hp_MakroZeiss100mm_fps10000_4pumps__Cam_20861_Cine'+str(n) for n in cine_nums]
    #cine_names = [r'PIV_tv_hp_fps10000_makrozeiss100mm_with_balloon_1_Cam_20861_Cine'+str(n) for n in cine_nums]


    #dx=0.000225257289297 # 171221
    #dx = 0.00020305876635 # 171222
    #dx = 0.000245911883132 #180107
    #dx = 0.000140480421644 # 180108, 4 pumps
    #dx = 0.000168066277433 # 180108, diffuser
    #dx = 0.000140480421644 # 180108 4pumps side view
    #dx_sideView = 0.00014630278711
    #dx = dx_sideView
    #dx = 0.000221838581492 # 180117
    #dx = 0.000222575516693 # 181019
    #dx = 0.000203704906811 # 180121
    #dx = 0.000204237510653
    #dx = 0.000185198809242 # 180202
    #dx = 0.00027228836458 # 180205
    #dx = 3.5005834306e-05*2.
    #dx = 8.5353829427E-05
    #dx = 9.8467328387E-05 # 180228
    dx = .00015
    
    name_for_save = cine_name+'_start_frame_3'
    
     
    #crop_lims = [0,-1,395,860]
    #crop_lims = [0,800,450,900]
    #crop_lims = [300,365,1000,1065]
    #crop_lims = [0,2160,750,3800,]
    crop_lims=None
    pre_constructed_masker = None
    
    #crop_lims = [200,610,0,-1]
    
    #a_frames=np.arange(0,8000,500)
    #a_frames = np.arange(0,86354,2)
    a_frames = np.arange(3833,8833,20)
    processing = PIVDataProcessing(parent_folder,cine_name,dx=dx,dt_orig=1./1000,frame_diff=10,crop_lims=crop_lims,maskers=None,window_size=32,overlap=16,search_area_size=32,name_for_save=name_for_save)
    #processing.cine_filepath = r'C:\Users\Luc Deike\highspeed_data\171221\piv_4pumps_topDown_sched-tenthOn_T0250ms_centerView_fps4000_tiffs\*.tif'W
    processing.run_analysis(a_frames=a_frames,save=True,s2n_thresh=1.2)
    processing.associate_flowfield()
    
#    stophere
#    ff_3 = processing.data.ff.copy()
#
#    i = 0
#    fig=plt.figure(figsize=(16,8))
#    ax=fig.add_subplot(111)
#    for i in np.arange(len(a_frames)):
#        ax.clear()
#        ax.imshow(np.sqrt(processing.data.ff[i,:,:,0]**2+processing.data.ff[i,:,:,1]**2),vmin=0,vmax=.5)
#        #ax.quiver(processing.data.ff[i,:,:,0],processing.data.ff[i,:,:,1])
#        ax.set_title('t = '+str(processing.dt_orig*processing.a_frames[i]))
#        plt.show()
#        plt.pause(1)
##    
#
#    mean_flow = np.nanmean(processing.data.ff,axis=0)
#    fig=plt.figure(figsize=(12,8))
#    ax=fig.add_subplot(111)
#    #ax.imshow(np.sqrt(mean_flow[:,:,0]**2+mean_flow[:,:,1]**2),vmin=0,vmax=0.3)
#    ax.imshow(mean_flow[:,:,0],vmin=-.1,vmax=0.1,cmap='seismic')
#    #ax.quiver(mean_flow[:,:,0],mean_flow[:,:,1])
#    
#    plt.show()
#    plt.pause(1)
#        
#    
#    fig,axs = plt.subplots(3,3,figsize=(12,8))
#    axs = axs.flatten()
#    
#    for i in np.arange(len(a_frames)):
#        axs[i].imshow(np.sqrt(processing.data.ff[i,:,:,0]**2+processing.data.ff[i,:,:,1]**2),vmin=0,vmax=1)
#        axs[i].quiver(processing.data.ff[i,:,:,0],processing.data.ff[i,:,:,1])
#        #axs[i].set_title('t = '+str(processing.dt_orig*processing.a_frames[i]))
#        #plt.show()
#        #plt.pause(1)
#        
#    plt.tight_layout()
#    