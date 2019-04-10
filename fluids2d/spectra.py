# -*- coding: utf-8 -*-
"""
Created on Thu Jan 04 19:55:56 2018

@author: Dan Ruth
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fluids2d.piv as piv
import pickle
import os.path
from fluids2d.geometry import GeometryScaler

###############################################################################

'''
FUNCTIONS FOR TIME SPECTRA
'''

class AutocorrResults:
    
    def __init__(self,g,parent_folder=None,case_name=None):
        self.g = g
        self.coords_used = None # [top,bottom,left,right]
        self.parent_folder = parent_folder
        self.case_name = case_name
        if self.case_name is not None:
            self.filepath = self.parent_folder+self.case_name+r'_autocorr.pkl'
        else:
            self.filepath = None
        
        self.C = None
        self.lags = None
        self.C_avg = None
        
        self.B = None
        self.freq = None
        
        return
    
    def save(self):
        pickle.dump(self,open(self.filepath,'wb'))
    
    def run_autocorr(self,ff,time,num_lags,coords_used):
        
        '''
        First see if the file exists already
        '''
        if (self.filepath is not None) and os.path.isfile(self.filepath):
            pass
        
        '''
        Run and save the autocorrelation
        '''
    
        self.coords_used = coords_used
        self.dt = time[1]-time[0]
        self.num_lags = num_lags
        
        fluc = ff - np.nanmean(ff,axis=0)
        
        fluc_to_use = fluc.copy()[:,self.coords_used[0]:self.coords_used[1]+1,self.coords_used[2]:self.coords_used[3]+1,:]
        
        '''
        Fill in the NaNs
        '''
        for d in [0,1]:
            fluc_to_use[:,:,:,d] = piv.fill_nans_3d(fluc_to_use[:,:,:,d])
            
        '''
        Do the autocorrelation
        '''
        #self.i_lags = range(0,2000*8)
        self.C,self.lags = autocorr(fluc_to_use,i_lags=np.arange(int(self.num_lags)),dt=self.dt)
        self.C_avg = np.nanmean(np.nanmean(self.C[:,:,:,:],axis=1),axis=1) # spatial average
        
        '''
        Save it
        '''
        if self.parent_folder is not None:
            self.save()
        else:
            print('Not saving the autocorr, since the parent folder has not been specified.')
            
    def eulerian_spectrum(self):
        self.B, self.freq = temporal_spectrum_from_autocorr(self.C_avg,self.lags)
        return self.B, self.freq
    
def autocorr_if_necessary(g,parent_folder,case_name,ff,time,num_lags,coords_used):
    return
    

def autocorr(x,i_lags=np.arange(0,20000), lag_spacing=1, dt=1):    
    '''
    Autocorrelation function
    '''
    
    x = x - np.nanmean(x,axis=0)    
    mean_x_sq = np.nanmean(x**2,axis=0)
    
    #lags = np.arange(0,num_lags*lag_spacing,lag_spacing).astype(int)
    lags = i_lags.astype(int)
    
    shape = list(np.shape(x))
    shape[0] = len(i_lags)
    C = np.zeros(shape)
    for li,lag in enumerate(lags):
        '''
        For each lag to be tested, find the average product of measurements
        separated by that time.
        '''
        #print(float(lag)/(num_lags*lag_spacing))
        print(li)
        
        C[li,...] = np.nanmean(x[0:len(x)-1*lag,...]*x[lag:len(x),...],axis=0)/mean_x_sq
#        
#    plt.figure()
#    plt.plot(lags*dt,np.nanmean(np.nanmean(C,axis=1),axis=1))
#    plt.show()
#    plt.pause(0.1)
        
    print(C)
    print(lags)
    print(dt)
    print(lags*dt)

    return C,lags.astype(float)*dt

def stitch_autocorrs(C_list,lags_list):
    C = []
    lags = []
    for this_C,this_lag in zip(C_list,lags_list):
        C.append(this_C)
        lags.append(this_lag)
        
    return C, lags

def autocorr_multiprocessing(x,i_lags=range(0,20000),dt=1,num_jobs=4):
    
    import multiprocessing as mp
    
    jobs = []
    for _ in range(num_jobs):
        p = mp.Process(target=autocorr,args=(x))
        jobs.append(p)
        p.start()
    
    for job in jobs:
        job.join()

def region_autocorrs(ff,num_lags,lag_spacing=1,dt=1):
    s = np.shape(ff)
    
    num_timepoints = s[0]
    num_rows = s[1]
    num_cols = s[2]
    num_dirs = s[3]
    
def abs_fft(y,x):
    keep_first = len(x)/2
    
    #s = pd.Series(index=x,data=y)
    #x = piv.fill_nans(x)
    
    f = np.absolute(np.fft.fft(y)[0:keep_first])
    freq = np.fft.fftfreq(len(x),d=x[1]-x[0])[0:keep_first]
    
    return f,freq

def temporal_spectrum_from_autocorr(C,lags):
    '''
    Energy spectrum from the autocorrelation
    '''
    
    keep_first = len(lags)/2
    
    B = np.real(np.fft.fft(C)[0:keep_first])
    freq = np.fft.fftfreq(len(lags),d=lags[1]-lags[0])[0:keep_first]
    
    return B, freq

###############################################################################

'''
FUNCTIONS FOR SPATIAL SPECTRA
'''

def calculate_spatial_correlations(g,im,center_row,center_col,center_size,search_x,search_y):
        
    im = im-np.nanmean(im,axis=0)
    
    im_shape = np.shape(im)
    num_t = im_shape[0]
    
    
    
    x_vec = range(center_col-center_size/2,center_col+center_size/2+1)
    print(x_vec)
    y_vec = range(center_row-center_size/2,center_row+center_size/2+1)

    xsearch_vec = range(-1*search_x/2,search_x/2+1)
    ysearch_vec = range(-1*search_y/2,search_y/2+1)
            
    dir_vec= [0,1]

    res = np.zeros((num_t,center_size,center_size,search_y+1,search_x+1,2,2))
    #res_avg = np.zeros((center_size,center_size,search_y,search_x,2,2))
    
    g_r = GeometryScaler(dx=g.dx,im_shape=(len(ysearch_vec),len(xsearch_vec)),origin_pos=(-search_y/2,-search_x/2),origin_units='pix')
    
    for xi,x in enumerate(x_vec):
        # for each originating x point
        
        print(xi)
        for yi,y in enumerate(y_vec):
            
            im_originating_point = im[:,y,x,:]
            # for each originating y point
                        
            for xsearchi,xsearch in enumerate(xsearch_vec):
                # for each step to test in thee x direction
                
                for ysearchi,ysearch in enumerate(ysearch_vec):
                    
                    for dir_1 in dir_vec:
                        for dir_2 in dir_vec:
                            # for each step to test in the y direction
                    
                            res[:,yi,xi,ysearchi,xsearchi,dir_1,dir_2] = im_originating_point[...,dir_1] * im[:,y+ysearch,x+xsearch,dir_2]
                            #res_avg[yi,xi,ysearchi,xsearchi,dir_1,dir_2] = np.nanmean(im[:,y+ysearch,x+xsearch,dir_1] * im_originating_point[...,dir_2],axis=0)
                            
    return res,g_r

#def get_axial_and_transverse_components(corr):
#    
#    search_y = np.shape(corr)[-4]
#    search_x = np.shape(corr)[-3]
#    
#    X,Y = np.meshgrid(np.arange(np.shape(corr)[1])-np.shape(corr)[1]/2,np.arange(np.shape(corr)[0])-np.shape(corr)[0]/2)
#    theta = np.arctan2(Y,X)
#    
#    axial = corr[:,:,0,0] * np.sin(theta) + corr[:,:,1,1] * np.cos(theta)
#    
#    return axial

def make_radial_correlations(corr,g_r,dr=0.5,n_r=200,n_theta=50):
    
    search_y = np.shape(corr)[-4]
    search_x = np.shape(corr)[-3]
    
    def polar2cart(rtheta):
        r = float(rtheta[0])*dr
        theta = float(rtheta[1])/n_theta*(2*np.pi)
        
        y = search_y/2 + r*np.sin(theta)
        x = search_x/2 + r*np.cos(theta)
        
        y_i = int(y)
        x_i = int(x)
        
        #print((y_i,x_i))
        
        return (y_i,x_i)
    
    from scipy.ndimage import geometric_transform
    
    dr_actual = dr*g_r.dx
    r = dr_actual*np.arange(n_r)
    
    polar = []
    line_avg = []
    integral = []
    
    for d in [0,1]:
    
        polar.append(geometric_transform(corr[...,d,d],polar2cart,order=1,output_shape=(n_r,n_theta),cval=np.nan)) # ,
        
#        plt.figure()
#        plt.imshow(polar[d],origin='lower',aspect='auto')    
        
        line_avg.append(np.nanmean(polar[d],axis=1))    
        integral.append(np.cumsum(line_avg[d])*dr_actual/line_avg[d][0])
#        
#        plt.figure()
#        plt.plot(r,line_avg[d])
#        plt.title('B'+str(d)+str(d)+'(r)')
#        
#        plt.figure()
#        plt.plot(r,integral[d])
#        plt.title('L'+str(d)+str(d)+' integral')
    
    polar = np.moveaxis(np.array(polar),0,-1)
    line_avg = np.moveaxis(np.array(line_avg),0,-1)
    integral = np.moveaxis(np.array(integral),0,-1)
    
    return polar,r,line_avg,integral
    

###############################################################################

if __name__ == '__main__':
    
    import fluids2d.piv as piv
    import fluids2d.geometry
    import pickle
    from fluids2d.piv import PIVDataProcessing
    
    parent_folder = r'\\DESKTOP-TDOAU0M\PIV_Data\Stephane\180107\\'
    #parent_folder = r'D:\high_speed_data\171222\\'
    #case_name = r'piv_4pumpsDiffusers_topDown_sched-thirdOn_T0500ms_fullView_fps4000_dur4s'
    case_name = r'piv_4pumps_back_topDown_sched-tenthOn_T0500ms_fps4000_dur4s'
    
    p = pickle.load(open(parent_folder+case_name+'.pkl'))
    p.parent_folder = parent_folder
    p.associate_flowfield()   
    
    if False:
        p.data.ff=piv.rotate_data_90(p.data.ff)
    ff = p.data.ff
    
    g_orig = fluids2d.geometry.GeometryScaler(dx=p.dx,im_shape=(1,1),origin_pos=(0,0),origin_units='pix')
    g = fluids2d.geometry.create_piv_scaler(p,g_orig)

    #im = ff[0:2000,...]
    
    center_size = 3 # must be odd
    
    center_row = (np.shape(ff)[1]-1)/2
    center_col = (np.shape(ff)[2]-1)/2
    
    search_x = np.shape(ff)[2] - center_size
    search_y = np.shape(ff)[1] - center_size
    
    
    res,g_r = calculate_spatial_correlations(g,ff,center_row,center_col,center_size,search_x,search_y)
    print('done the correlations')
                    
    spatial_average = np.nanmean(res,axis=(1,2))
    temporal_average = np.nanmean(res,axis=0)
    temporal_and_spatial_average = np.nanmean(spatial_average,axis=0)
    #temporal_and_spatial_average = np.nanmean(temporal_average,axis=(0,1))
    
#    line_avgs = []
#    for nx in np.arange(search_x):
#        for ny in np.arange(search_y):
#            corr_field = temporal_average[:,:,[ny],[nx],:,:]
#            polar,r,line_avg,integral = make_radial_correlations(corr_field,g_r,dr=0.5) 
#            line_avgs.append(line_avg)
#            
#    line_avgs = np.array(line_avgs)

    '''
    Normalize by starting position
    '''
    fig,axs = plt.subplots(2,2,figsize=(18,9))
    max_val = np.nanmax(temporal_and_spatial_average)
    min_val = np.nanmin(temporal_and_spatial_average)
    for i in [0,1]:
        for j in [0,1]:
            axs[i,j].imshow(temporal_and_spatial_average[...,i,j],extent=g_r.im_extent,vmin=min_val,vmax=max_val)
            axs[i,j].set_title('B'+str(i)+str(j))
            axs[i,j].set_xlabel('''$\Delta x$ [m]''')
            axs[i,j].set_ylabel('''$\Delta y$ [m]''')
            #g_r.set_axes_limits(axs[i,j])
    fig.tight_layout()
    
    '''
    mean flow fields
    '''
    
    plt.figure()
    plt.imshow(np.nanmean(ff[...,0],axis=0),extent=g.im_extent)
    plt.figure()
    plt.imshow(np.nanmean(ff[...,1],axis=0),extent=g.im_extent)
            
    '''
    hopefully u_rms
    '''
    fig,axs = plt.subplots(2,2,figsize=(15,9))
    
    for i in [0,1]:
        for j in [0,1]:
            axs[i,j].imshow(temporal_average[:,:,center_row-center_size/2,center_col-center_size/2,i,j])
            axs[i,j].set_title('B'+str(i)+str(j))
            
    polar,r,line_avg,integral = make_radial_correlations(temporal_and_spatial_average,g_r,dr=0.5)