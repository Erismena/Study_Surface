# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 16:22:36 2017

@author: danjr
"""

import matplotlib.pyplot as plt
import trackpy as tp
import numpy as np
import pandas as pd
from scipy.ndimage.morphology import binary_fill_holes
import skimage.measure
import skimage.filters
import skimage.morphology
import skimage.segmentation
import scipy.ndimage
from matplotlib.patches import Ellipse
import scipy
import cv2
from skimage import measure
import cv2
import os
'''
Some basic image processing functions
'''

def video(c):
    image_folder = 'images'
    video_name = 'video.avi'

    images = [c[f] for f in frames]
    height, width= c[0].shape

    video = cv2.VideoWriter(video_name, -1, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


def binarize(im,thresh,large_true=True):
    '''
    Return a boolean image based on some threshold, which can be either a 
    single value or a "threshold image"
    '''
    if large_true:
        return im>thresh
    elif large_true==False:
        return im<thresh

def apply_mask(im,mask):
    return np.array(im,dtype=int)*mask

# get_filled SHOULD BE USED INSTEAD
#def raw2binaryfilled(im,mask,thresh,large_true=True):
#    '''
#    binarize, mask, and fill holes
#    '''
#    binarized=binarize(im,thresh,large_true=large_true)
#    masked=apply_mask(binarized,mask)
#    filled = binary_fill_holes(masked)
#    return filled
def Is_inside_cercle(x=0,y=0,centre_x=0,centre_y=0,radius=0):
    A=np.sqrt((x-centre_x)**2+(y-centre_y)**2)
    if A>radius:
        return False
    else:
        return True

def combinations_given_size(iterable, r):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = range(r)
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)
        
def combinations(iterable,r_max):
    A=[]
    for r in np.arange(0,r_max+1,1):
        r=int(r)
        A=A+list(combinations_given_size(iterable, r))
    return A

def construct_bg_image(c,frames=None,usemax=True):
    '''
    Use the 90th percentile of each frame's intensity over some frames as the 
    background 
    '''
    if frames is None:
        frames = np.linspace(0,len(c)-1,100).astype(int)
        
    '''
    Extract just the frames we are considering 
    '''
    imshape = np.shape(c[0])
    c_just_some_frames = np.zeros([len(frames),imshape[0],imshape[1]])
    for fi,f in enumerate(frames):
        c_just_some_frames[fi,:,:] = c[f]
        
    '''
    Make the bg image as the 90th (or 10th if the background is lower
    intensity) and then median filter it.
    '''
    if usemax:
        p = 90
    else:
        p=10        
    bg_image = np.percentile(c_just_some_frames,p,axis=0)
    bg_image = scipy.ndimage.filters.median_filter(bg_image,size=3)
    
    return bg_image

def get_filled(im,thresh):
    im_filt = scipy.ndimage.filters.median_filter(im,size=3)
    im_filt = binarize(im_filt,thresh,large_true=False)
    filled = binary_fill_holes(im_filt)
    return filled

def labeled_props(labeled,g,frame=None,min_area=0):
    '''
    Return a dataframe where each row corresponds to a labeled object in the 
    image "labeled" and each column is an output from regionprops.
    '''
    
    props = skimage.measure.regionprops(labeled)
    
    dx = g.dx
    
    df = pd.DataFrame(columns=['radius']) # some columns are necessary even if there will be no rows in the df
    if (len(props)==0)==False:
        for ri,region in enumerate(props):
            
            y,x = g.get_loc([region.centroid[0],region.centroid[1]])
            df.loc[ri,'y']=y[0]
            df.loc[ri,'x']=x[0]
            df.loc[ri,'y_pix']=region.centroid[0]
            df.loc[ri,'x_pix']=region.centroid[1]
            df.loc[ri,'radius'] = region.equivalent_diameter / 2. * dx
            df.loc[ri,'orientation'] = region.orientation / (2*np.pi) * 360
            df.loc[ri,'major_axis_length'] = region.major_axis_length* dx
            df.loc[ri,'minor_axis_length'] = region.minor_axis_length* dx
#            df.loc[ri,'perimeter'] = region.perimeter*dx
            df.loc[ri,'eccentricity'] = region.eccentricity
            df.loc[ri,'convex_area'] = region.convex_area * dx**2
            df.loc[ri,'frame'] = frame
            df.loc[ri,'filled_area'] = region.filled_area * dx**2
            df.loc[ri,'area/c_area']=df.loc[ri,'filled_area']/df.loc[ri,'convex_area']
        
        df = df[df['filled_area']>min_area]
        
    return df

def Velocity(DF,Dt=1):
    DF['Uy']=-DF['y'].diff()/Dt
    DF['Uy']=DF['Uy'].apply(triVelocity)
    DF['Ux']=DF['x'].diff()/Dt
    DF['U']=DF['Uy']**2+DF['Ux']**2
    DF['U']=DF['U'].apply(np.sqrt)
    return DF

def triVelocity(x):
    if x<0:
        x=np.nan
    return x
#A=[U[1][U[1]['particle']==p].index[0] for p in U[1]['particle'].unique()]
#for i in A:
#    U[1].loc[i,'U']=np.nan



def filled2regionpropsdf(filled,g=None,min_area=0,frame=None):
    objects, num_objects = scipy.ndimage.label(filled)
    df = labeled_props(objects,g,frame=frame,min_area=min_area)
    return df

def watershed_detection(im,thresh,g,viz=True,RadiusDiskMean=2):
    '''
    Entire processing pipeline using Nicolas's watershed method.
    '''
    
    # fill in the image
    filled_im = get_filled(im,thresh)
    
    # distance transform on the filled image
    distance_transform=scipy.ndimage.morphology.distance_transform_bf(filled_im, return_distances=True, return_indices=False)
    
    # median filter this image
    distancetransform_medfilt=skimage.filters.median(distance_transform.astype(int), skimage.morphology.disk(RadiusDiskMean))
    
    # local maxima in this image
    local_max = skimage.morphology.local_maxima(distancetransform_medfilt)
    
    # connected components
    ret,markers = cv2.connectedComponents(local_max)
    
    # watershed
    watershed_im=skimage.morphology.watershed(filled_im,markers,mask=filled_im)
    
    if viz:
        '''
        Make a figure showing each step of the process.
        '''
        
        print('starting the visualization')
        
        fig,axs = plt.subplots(2,4,figsize=(17,9),sharex=True,sharey=True)
        axs = axs.flatten()
        
        ims_to_show = [im,
                       filled_im,
                       distance_transform,
                       distancetransform_medfilt,
                       local_max,
                       markers,
                       watershed_im]
        
        names = ['original',
                 'filled',
                 'distance transform',
                 'medfilt of distance transform',
                 'local maxima',
                 'connected components',
                 'watershed']
        
        df = labeled_props(watershed_im,g)
        
        # go through each image to show and add to its own axes
        for i,(im,name) in enumerate(zip(ims_to_show,names)):
            axs[i].imshow(im,extent=g.im_extent)
            axs[i].set_title(name)
            
            # add all the ellipses to each image
            for ix in df.index:
                e = Ellipse([df.loc[ix,'x'],df.loc[ix,'y']],width=df.loc[ix,'major_axis_length'],height=df.loc[ix,'minor_axis_length'],angle=df.loc[ix,'orientation'])
                axs[i].add_artist(e)
                e.set_facecolor('None')
                e.set_edgecolor([1,0,0,0.5])            
                       
    return watershed_im

def random_walker_detection(im,thresh,g,mode='cg',RadiusDiskMean=2,tol=0.01,viz=True):
    '''
    Like watershed, but with random_walker in place of watershed
    '''
    
    # fill in the image
    filled_im = get_filled(im,thresh)
    
    # distance transform on the filled image
    distance_transform=scipy.ndimage.morphology.distance_transform_bf(filled_im, return_distances=True, return_indices=False)
    
    # median filter this image
    distancetransform_medfilt=skimage.filters.median(distance_transform.astype(int), skimage.morphology.disk(RadiusDiskMean))
    
    # local maxima in this image
    local_max = skimage.morphology.local_maxima(distancetransform_medfilt)
    
    # connected components
    ret,markers = cv2.connectedComponents(local_max)    
    markers[~filled_im] = -1
    
    # do the random walking segmentation
    rw_im = skimage.segmentation.random_walker(filled_im,markers,mode=mode,tol=tol)
    rw_im[rw_im==-1]=0
    
    if viz:
        '''
        Make a figure showing each step of the process.
        '''
        
        print('starting the visualization')
        
        fig,axs = plt.subplots(2,4,figsize=(17,9),sharex=True,sharey=True)
        axs = axs.flatten()
        
        ims_to_show = [im,
                       filled_im,
                       distance_transform,
                       distancetransform_medfilt,
                       local_max,
                       markers,
                       rw_im]
        
        names = ['original',
                 'filled',
                 'distance transform',
                 'medfilt of distance transform',
                 'local maxima',
                 'connected components',
                 'random walker']
        
        df = labeled_props(rw_im,g)
        
        # go through each image to show and add to its own axes
        for i,(im,name) in enumerate(zip(ims_to_show,names)):
            axs[i].imshow(im,extent=g.im_extent)
            axs[i].set_title(name)
            
            add_ellipses_to_ax(df,axs[i])
    
    return rw_im

def random_walker_detection_cluster_dist(im,thresh,g,frame=None,MinRadius=0,MaxRadius=25000,mode='cg',RadiusDiskMean=1,tol=0.01,viz_process=True,viz=True):
    '''
    Like watershed, but with random_walker in place of watershed
    '''
    # fill in the image
    filled_im = get_filled(im,thresh)
    
    # remove non wanted objects
    objects, num_objects = scipy.ndimage.label(filled_im)
    df_cluster = labeled_props(objects+1,g,frame=frame,min_area=0)
    df_cluster = df_cluster.drop(0,axis=0)
    
    "delete bad cluster"
    label_lower_cluster=df_cluster[df_cluster['radius']<=MinRadius].index.values
    label_higher_cluster=df_cluster[df_cluster['radius']>=MaxRadius].index.values
    label_non_wanted_cluster=np.concatenate((label_lower_cluster,label_higher_cluster),axis=0)
    
    df_cluster = df_cluster[df_cluster['radius']>MinRadius]
    df_cluster = df_cluster[df_cluster['radius']<MaxRadius]
    df_cluster['labelCluster']=df_cluster.index

    for i in label_non_wanted_cluster:
        objects[objects==i]=0
    
#    for i in np.arange(MaskTop,MaskBottom,1):
#        for j in np.arange(MaskLeft,MaskRight,1):
#            if objects[i][j] in label_non_wanted_cluster:
#                filled_im[i][j]=False
#                objects[i][j]=0

    '''CONTOURS'''
#    df_cluster_temp = pd.DataFrame(columns=['contours'], index=df_cluster.index,dtype=object)
#    C_clusters=contours(objects)
#    for i in df_cluster.index:
#        df_cluster_temp.loc[i,'contours']=C_clusters[i-1][0]
#    df_cluster['contours']=df_cluster_temp['contours']
    

#    df_cluster = df_cluster.reindex(df_cluster.index+1,method='ffill')
#    df_cluster = df_cluster.reset_index(drop=True)
    
    # distance transform on the filled image
    distance_transform=scipy.ndimage.morphology.distance_transform_bf(filled_im, return_distances=True, return_indices=False)
    
    # median filter this image
    distancetransform_medfilt=skimage.filters.median(distance_transform.astype(int), skimage.morphology.disk(RadiusDiskMean))
    
    # local maxima in this image
    local_max = skimage.morphology.local_maxima(distancetransform_medfilt)
    
    # connected components
    ret,markers = cv2.connectedComponents(local_max)    
    markers[~filled_im] = -1
    
    # do the random walking segmentation
    rw_im = skimage.segmentation.random_walker(filled_im,markers,mode=mode,tol=tol)
    rw_im[rw_im==-1]=0
    
    df_bubble = labeled_props(rw_im+1,g,frame=frame)
    df_bubble = df_bubble.drop(0,axis=0)

    "delete bad bubbles"
    label_lower_bubble=df_bubble[df_bubble['radius']<=MinRadius].index.values
    label_higher_bubble=df_bubble[df_bubble['radius']>=MaxRadius].index.values
    label_non_wanted_bubble=np.concatenate((label_lower_bubble,label_higher_bubble),axis=0)
    
    df_bubble = df_bubble[df_bubble['radius']>MinRadius]
    df_bubble = df_bubble[df_bubble['radius']<MaxRadius]
    df_bubble['labelBubble']=df_bubble.index

    for i in np.arange(MaskTop,MaskBottom,1):
        for j in np.arange(MaskLeft,MaskRight,1):
            if rw_im[i][j] in label_non_wanted_bubble:
                filled_im[i][j]=False
                rw_im[i][j]=0




    df_bubble_temp = pd.DataFrame(columns=['contours'], index=df_bubble.index,dtype=object)
    C_bubbles=contours(rw_im)
    for i in df_bubble.index:
        df_bubble_temp.loc[i,'contours']=C_bubbles[i-1][0]
    df_bubble['contours']=df_bubble_temp['contours']
    
    df_bubble['cluster']=objects[df_bubble['y_pix'].astype(int),df_bubble['x_pix'].astype(int)]
    df_bubble['labelBubble']=df_bubble.index
    
    df_cluster0 = pd.DataFrame(columns=['bubble'], index=df_cluster.index,dtype=object)
    for p in df_bubble['cluster'].unique():
        df_cluster0.loc[p,'bubble']=df_bubble[df_bubble['cluster']==p].index
    df_cluster['bubble'] = df_cluster0['bubble']

    df_cluster=df_cluster.dropna(axis=0)

    df_cluster['num_of_bubble'] = df_cluster['bubble']
    df_cluster['num_of_bubble'] = df_cluster['num_of_bubble'].apply(np.asarray)
    df_cluster['num_of_bubble'] = df_cluster['num_of_bubble'].apply(len)
    
    df_cluster['real_cluster'] = df_cluster['num_of_bubble']
    df_cluster['real_cluster'] = df_cluster['real_cluster'].apply(lambda x: False if x==1 else True)
    
    df_bubble['belong_real_cluster']=0
    for p in df_bubble.index:
        df_bubble.loc[p,'belong_real_cluster']=df_bubble['cluster'].value_counts()[df_bubble.loc[p,'cluster']]
    df_bubble['belong_real_cluster'] = df_bubble['belong_real_cluster'].apply(lambda x: False if x==1 else True)
    
#    df_cluster=Velocity(df_cluster,Dt=dt)
#    df_bubble=Velocity(df_bubble,Dt=dt)

    View=filled_im*im
    

    if viz:
        '''
        Make a figure showing each frames.
        '''
        print('starting the visualization')
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(objects,extent=g.im_extent)
        add_ellipses_to_ax(df_bubble,ax)
        add_ellipses_to_ax(df_cluster,ax)
        add_text_to_ax(df_bubble,ax)
        add_text_to_ax(df_cluster,ax)
    
    if viz_process:
        '''
        Make a figure showing each step of the process.
        '''
        
        print('starting the visualization')
        
        fig,axs = plt.subplots(2,4,figsize=(17,9),sharex=True,sharey=True)
        axs = axs.flatten()
        
        ims_to_show = [im,
                       filled_im,
                       distance_transform,
                       distancetransform_medfilt,
                       local_max,
                       markers,
                       rw_im]
        names = ['original',
                 'filled',
                 'distance transform',
                 'medfilt of distance transform',
                 'local maxima',
                 'connected components',
                 'random walker']
#        df = labeled_props(rw_im,g)
        # go through each image to show and add to its own axes
        for i,(im,name) in enumerate(zip(ims_to_show,names)):
            axs[i].imshow(im,extent=g.im_extent)
            axs[i].set_title(name)
            add_ellipses_to_ax(df_bubble,axs[i])
            add_ellipses_to_ax(df_cluster,axs[i])
            add_text_to_ax(df_bubble,axs[i])
            add_text_to_ax(df_cluster,axs[i])
            
#T[1].to_excel(r'C:\Users\Luc Deike\Nicolas\StudySurface\Trials\3\Clusters_6\Cluster_6_c.xls')
    return rw_im, df_bubble, df_cluster
   
    

def random_walker_detection_cluster_holes(im,thresh,g,frame=None,MinRadius=0,MaxRadius=25000,MinRadius_holes=1,mode='cg_mg',RadiusDiskMean=2,tol=0.01,viz_process=False,viz=False,path_folder='',name=''):
    '''
    Like watershed, but with random_walker in place of watershed, and makers with holes
    '''
    # fill in the image
#    filled_im = get_binarize(im,thresh)
    filled_im_with_holes=get_binarize(im,thresh)
    filled_im_without_holes=binary_fill_holes(filled_im_with_holes)
    # remove non wanted objects
    objects, num_objects = scipy.ndimage.label(filled_im_without_holes)
    df_cluster = labeled_props(objects+1,g,frame=frame,min_area=0)
    df_cluster = df_cluster.drop(0,axis=0)
    
    "delete bad cluster"
    label_lower_cluster=df_cluster[df_cluster['radius']<=MinRadius].index.values
    label_higher_cluster=df_cluster[df_cluster['radius']>=MaxRadius].index.values
    label_non_wanted_cluster=np.concatenate((label_lower_cluster,label_higher_cluster),axis=0)
    
    df_cluster = df_cluster[df_cluster['radius']>MinRadius]
    df_cluster = df_cluster[df_cluster['radius']<MaxRadius]
    df_cluster['labelCluster']=df_cluster.index

    for i in label_non_wanted_cluster:
        filled_im_with_holes[objects==i]=False
        filled_im_without_holes[objects==i]=False
        objects[objects==i]=0

    "delete what is not a hole of a bubble"
    holes=~filled_im_with_holes*filled_im_without_holes
    objects_holes, num_objects_holes = scipy.ndimage.label(holes)
    df_holes = labeled_props(objects_holes+1,g,frame=frame,min_area=0)
    '....delete lowest holes'
    label_lower_holes=df_holes[df_holes['radius']<=MinRadius_holes].index.values
    df_holes = df_holes[df_holes['radius']>MinRadius_holes]
    df_holes['labelHole']=df_holes.index
    for i in label_lower_holes:
        filled_im_without_holes[objects_holes==i]=False
        objects_holes[objects_holes==i]=0
    '....delete convex holes'
    label_convex_holes=df_holes[df_holes['area/c_area']<=0.93].index.values
    df_holes = df_holes[df_holes['area/c_area']>0.93]
    df_holes['labelHole']=df_holes.index
    for i in label_convex_holes:
        filled_im_without_holes[objects_holes==i]=False
        objects_holes[objects_holes==i]=0
    holes=np.copy(objects_holes)
    holes[holes>1]=1
      
#    df_cluster_temp = pd.DataFrame(columns=['contours'], index=df_cluster.index,dtype=object)
#    C_clusters=contours(objects)
#    for i in df_cluster.index:
#        df_cluster_temp.loc[i,'contours']=C_clusters[i-1][0]
#    df_cluster['contours']=df_cluster_temp['contours']
          
    markers=np.copy(objects_holes)
    markers[~filled_im_without_holes] = -1
    
    # do the random walking segmentation
    try:
        rw_im = skimage.segmentation.random_walker(filled_im_without_holes,markers,mode=mode,tol=tol)
    except:
        pass
        rw_im=filled_im_without_holes
        rw_im[:]=0
        print('err')
    rw_im[rw_im==-1]=0
    
    df_bubble = labeled_props(rw_im+1,g,frame=frame)
    df_bubble = df_bubble.drop(0,axis=0)

    'pas besoin car on a deja jarte les mauvais trous'
#    "delete bad bubbles" 
#    label_lower_bubble=df_bubble[df_bubble['radius']<=MinRadius].index.values
#    label_higher_bubble=df_bubble[df_bubble['radius']>=MaxRadius].index.values
#    label_non_wanted_bubble=np.concatenate((label_lower_bubble,label_higher_bubble),axis=0)
#    df_bubble = df_bubble[df_bubble['radius']>MinRadius]
#    df_bubble = df_bubble[df_bubble['radius']<MaxRadius]
#    df_bubble['labelBubble']=df_bubble.index
#    for i in np.arange(MaskTop,MaskBottom,1):
#        for j in np.arange(MaskLeft,MaskRight,1):
#            if rw_im[i][j] in label_non_wanted_bubble:
#                filled_im[i][j]=False
#                rw_im[i][j]=0
#    df_bubble_temp = pd.DataFrame(columns=['contours'], index=df_bubble.index,dtype=object)
#    C_bubbles=contours(rw_im)
#    for i in df_bubble.index:
#        df_bubble_temp.loc[i,'contours']=C_bubbles[i-1][0]
#    df_bubble['contours']=df_bubble_temp['contours']
    
    df_bubble['cluster']=objects[df_bubble['y_pix'].astype(int),df_bubble['x_pix'].astype(int)]
    df_bubble['labelBubble']=df_bubble.index
    
    df_cluster0 = pd.DataFrame(columns=['bubble'], index=df_cluster.index,dtype=object)
    df_cluster01 = pd.DataFrame(columns=['rad_bubble'], index=df_cluster.index,dtype=object)
    for p in df_bubble['cluster'].unique():
        df_cluster0.loc[p,'bubble']=df_bubble[df_bubble['cluster']==p].index.tolist()
        df_cluster01.loc[p,'rad_bubble']=np.asarray(df_bubble[df_bubble['cluster']==p]['radius']).tolist()
    df_cluster['bubble'] = df_cluster0['bubble']
    df_cluster['rad_bubble'] = df_cluster01['rad_bubble']

    df_cluster=df_cluster.dropna(axis=0)

    df_cluster['num_of_bubble'] = df_cluster['bubble']
    df_cluster['num_of_bubble'] = df_cluster['num_of_bubble'].apply(np.asarray)
    df_cluster['num_of_bubble'] = df_cluster['num_of_bubble'].apply(len)
    
    df_cluster['real_cluster'] = df_cluster['num_of_bubble']
    df_cluster['real_cluster'] = df_cluster['real_cluster'].apply(lambda x: False if x==1 else True)
    
    df_bubble['belong_real_cluster']=0
    for p in df_bubble.index:
        df_bubble.loc[p,'belong_real_cluster']=df_bubble['cluster'].value_counts()[df_bubble.loc[p,'cluster']]
    df_bubble['belong_real_cluster'] = df_bubble['belong_real_cluster'].apply(lambda x: False if x==1 else True)
    
#    df_cluster=Velocity(df_cluster,Dt=dt)
#    df_bubble=Velocity(df_bubble,Dt=dt)

    if viz:
        '''
        Make a figure showing each frames.
        '''
        print('starting the visualization')
        
        fig0,axs0 = plt.subplots(1,2,sharex=True,sharey=True)
        axs0 = axs0.flatten()
        axs0[0].imshow(im,extent=g.im_extent)
        axs0[0].set_title('Original frame '+str(frame))
        
        axs0[1].imshow(rw_im,extent=g.im_extent)
        axs0[1].set_title('Proceded frame '+str(frame)+' nb_C='+str(len(df_cluster))+' nb_B'+str(len(df_bubble)))   
        
#        axs0[0].set_xlim([MaskLeft, MaskRight])
#        axs0[0].set_ylim([MaskTop, MaskBottom])
#        
#        axs0[1].set_xlim([MaskLeft, MaskRight])
#        axs0[1].set_ylim([MaskTop, MaskBottom])

        ff = plt.gcf()
        dpi = ff.get_dpi()
        h, w = ff.get_size_inches()
        ff.set_size_inches(h*5, w*5)
        plt.show()
        
        if frame == 0:
            if not os.path.exists(path_folder+'\\Trait_'+name):
                os.mkdir(path_folder+'\\Trait_'+name)
        plt.savefig(path_folder+'\\Trait_'+name+'\\'+str(frame)+'.png')
#        plt.savefig(path_folder+name+'__'+str(frame)+'.png')
        plt.close('all')
#        for i in [0,1]:
#            add_label_to_ax(df_bubble,axs0[i])
#            add_label_to_ax(df_cluster,axs0[i])
#            add_ellipses_to_ax(df_bubble,axs0[i])
#            add_ellipses_to_ax(df_cluster,axs0[i])            
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
#        ax.imshow(rw_im,extent=g.im_extent)
#        add_ellipses_to_ax(df_bubble,ax)
#        add_ellipses_to_ax(df_cluster,ax)
#        add_label_to_ax(df_bubble,ax)
#        add_label_to_ax(df_cluster,ax)
    
    if viz_process:
        '''
        Make a figure showing each step of the process.
        '''
        
        print('starting the visualization')
        
        fig,axs = plt.subplots(2,3,sharex=True,sharey=True)
        axs = axs.flatten()
        
        ims_to_show = [im,
                       filled_im_with_holes,
                       objects,
                       markers,
                       rw_im]
        names = ['Original',
                 'Filled with holes',
                 'Clusters',
                 'Markers',
                 'Bubbles',]
#        df = labeled_props(rw_im,g)
        # go through each image to show and add to its own axes
        for i,(im,name) in enumerate(zip(ims_to_show,names)):
            axs[i].imshow(im,extent=g.im_extent)
            axs[i].set_title(name)
#            add_ellipses_to_ax(df_bubble,axs[i])
#            add_ellipses_to_ax(df_cluster,axs[i])
#            add_label_to_ax(df_bubble,axs[i])
#            add_label_to_ax(df_cluster,axs[i])
            
#T[1].to_excel(r'C:\Users\Luc Deike\Nicolas\StudySurface\Trials\3\Clusters_6\Cluster_6_c.xls')
    return rw_im, df_bubble, df_cluster



def watershed_detection_cluster_holes(im,thresh,g,frame=None,MinRadius=0,MaxRadius=25000,MinRadius_holes=1,mode='cg_mg',RadiusDiskMean=2,tol=0.01,viz_process=False,viz=False,path_folder='',name=''):
    '''
    Like watershed, but with random_walker in place of watershed, and makers with holes
    '''
    # fill in the image
#    filled_im = get_binarize(im,thresh)
    filled_im_with_holes=get_binarize(im,thresh)
    filled_im_without_holes=binary_fill_holes(filled_im_with_holes)
    # remove non wanted objects
    objects, num_objects = scipy.ndimage.label(filled_im_without_holes)
    df_cluster = labeled_props(objects+1,g,frame=frame,min_area=0)
    df_cluster = df_cluster.drop(0,axis=0)
    
    "delete bad cluster"
    label_lower_cluster=df_cluster[df_cluster['radius']<=MinRadius].index.values
    label_higher_cluster=df_cluster[df_cluster['radius']>=MaxRadius].index.values
    label_non_wanted_cluster=np.concatenate((label_lower_cluster,label_higher_cluster),axis=0)
    
    df_cluster = df_cluster[df_cluster['radius']>MinRadius]
    df_cluster = df_cluster[df_cluster['radius']<MaxRadius]
    df_cluster['labelCluster']=df_cluster.index

    for i in label_non_wanted_cluster:
        filled_im_with_holes[objects==i]=False
        filled_im_without_holes[objects==i]=False
        objects[objects==i]=0

    "delete what is not a hole of a bubble"
    holes=~filled_im_with_holes*filled_im_without_holes
    objects_holes, num_objects_holes = scipy.ndimage.label(holes)
    df_holes = labeled_props(objects_holes+1,g,frame=frame,min_area=0)
    '....delete lowest holes'
    label_lower_holes=df_holes[df_holes['radius']<=MinRadius_holes].index.values
    df_holes = df_holes[df_holes['radius']>MinRadius_holes]
    df_holes['labelHole']=df_holes.index
    for i in label_lower_holes:
        filled_im_without_holes[objects_holes==i]=False
        objects_holes[objects_holes==i]=0
    '....delete convex holes'
    label_convex_holes=df_holes[df_holes['area/c_area']<=0.96].index.values
    df_holes = df_holes[df_holes['area/c_area']>0.96]
    df_holes['labelHole']=df_holes.index
    for i in label_convex_holes:
        filled_im_without_holes[objects_holes==i]=False
        objects_holes[objects_holes==i]=0
    holes=np.copy(objects_holes)
    holes[holes>1]=1
      
#    df_cluster_temp = pd.DataFrame(columns=['contours'], index=df_cluster.index,dtype=object)
#    C_clusters=contours(objects)
#    for i in df_cluster.index:
#        df_cluster_temp.loc[i,'contours']=C_clusters[i-1][0]
#    df_cluster['contours']=df_cluster_temp['contours']
          
    markers=np.copy(objects_holes)
    markers[~filled_im_without_holes] = -1
    
    # do the random walking segmentation
    rw_im=skimage.morphology.watershed(filled_im_without_holes,markers,mask=filled_im_without_holes)
#    rw_im = skimage.segmentation.random_walker(filled_im_without_holes,markers,mode=mode,tol=tol)
    rw_im[rw_im==-1]=0
    
    df_bubble = labeled_props(rw_im+1,g,frame=frame)
    df_bubble = df_bubble.drop(0,axis=0)

    'pas besoin car on a deja jarte les mauvais trous'
#    "delete bad bubbles" 
#    label_lower_bubble=df_bubble[df_bubble['radius']<=MinRadius].index.values
#    label_higher_bubble=df_bubble[df_bubble['radius']>=MaxRadius].index.values
#    label_non_wanted_bubble=np.concatenate((label_lower_bubble,label_higher_bubble),axis=0)
#    df_bubble = df_bubble[df_bubble['radius']>MinRadius]
#    df_bubble = df_bubble[df_bubble['radius']<MaxRadius]
#    df_bubble['labelBubble']=df_bubble.index
#    for i in np.arange(MaskTop,MaskBottom,1):
#        for j in np.arange(MaskLeft,MaskRight,1):
#            if rw_im[i][j] in label_non_wanted_bubble:
#                filled_im[i][j]=False
#                rw_im[i][j]=0
#    df_bubble_temp = pd.DataFrame(columns=['contours'], index=df_bubble.index,dtype=object)
#    C_bubbles=contours(rw_im)
#    for i in df_bubble.index:
#        df_bubble_temp.loc[i,'contours']=C_bubbles[i-1][0]
#    df_bubble['contours']=df_bubble_temp['contours']
    
    df_bubble['cluster']=objects[df_bubble['y_pix'].astype(int),df_bubble['x_pix'].astype(int)]
    df_bubble['labelBubble']=df_bubble.index
    
    df_cluster0 = pd.DataFrame(columns=['bubble'], index=df_cluster.index,dtype=object)
    df_cluster01 = pd.DataFrame(columns=['rad_bubble'], index=df_cluster.index,dtype=object)
    for p in df_bubble['cluster'].unique():
        df_cluster0.loc[p,'bubble']=df_bubble[df_bubble['cluster']==p].index.tolist()
        df_cluster01.loc[p,'rad_bubble']=np.asarray(df_bubble[df_bubble['cluster']==p]['radius']).tolist()
    df_cluster['bubble'] = df_cluster0['bubble']
    df_cluster['rad_bubble'] = df_cluster01['rad_bubble']

    df_cluster=df_cluster.dropna(axis=0)

    df_cluster['num_of_bubble'] = df_cluster['bubble']
    df_cluster['num_of_bubble'] = df_cluster['num_of_bubble'].apply(np.asarray)
    df_cluster['num_of_bubble'] = df_cluster['num_of_bubble'].apply(len)
    
    df_cluster['real_cluster'] = df_cluster['num_of_bubble']
    df_cluster['real_cluster'] = df_cluster['real_cluster'].apply(lambda x: False if x==1 else True)
    
    df_bubble['belong_real_cluster']=0
    for p in df_bubble.index:
        df_bubble.loc[p,'belong_real_cluster']=df_bubble['cluster'].value_counts()[df_bubble.loc[p,'cluster']]
    df_bubble['belong_real_cluster'] = df_bubble['belong_real_cluster'].apply(lambda x: False if x==1 else True)
    
#    df_cluster=Velocity(df_cluster,Dt=dt)
#    df_bubble=Velocity(df_bubble,Dt=dt)

    if viz:
        if f<280:
            '''
            Make a figure showing each frames.
            '''
            print('starting the visualization')
            
            fig0,axs0 = plt.subplots(1,2,sharex=True,sharey=True)
            axs0 = axs0.flatten()
            axs0[0].imshow(im,extent=g.im_extent)
            axs0[0].set_title('Original frame '+str(frame))
            
            axs0[1].imshow(rw_im,extent=g.im_extent)
            axs0[1].set_title('Proceded frame '+str(frame)+' nb_C='+str(len(df_cluster))+' nb_B'+str(len(df_bubble)))   
            
            axs0[0].set_xlim([MaskLeft, MaskRight])
            axs0[0].set_ylim([MaskTop, MaskBottom])
            
            axs0[1].set_xlim([MaskLeft, MaskRight])
            axs0[1].set_ylim([MaskTop, MaskBottom])
            
            ff = plt.gcf()
            dpi = ff.get_dpi()
            h, w = ff.get_size_inches()
            ff.set_size_inches(h*5, w*5)
            plt.show()
            os.mkdir(path_folder+'\\Trait_'+name)
            plt.savefig(path_folder+'\\Trait_'+name+'\\'+str(frame)+'.png')
            plt.close('all')
#        for i in [0,1]:
#            add_label_to_ax(df_bubble,axs0[i])
#            add_label_to_ax(df_cluster,axs0[i])
#            add_ellipses_to_ax(df_bubble,axs0[i])
#            add_ellipses_to_ax(df_cluster,axs0[i])            
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
#        ax.imshow(rw_im,extent=g.im_extent)
#        add_ellipses_to_ax(df_bubble,ax)
#        add_ellipses_to_ax(df_cluster,ax)
#        add_label_to_ax(df_bubble,ax)
#        add_label_to_ax(df_cluster,ax)
    
    if viz_process:
        '''
        Make a figure showing each step of the process.
        '''
        
        print('starting the visualization')
        
        fig,axs = plt.subplots(2,3,sharex=True,sharey=True)
        axs = axs.flatten()
        
        ims_to_show = [im,
                       filled_im_with_holes,
                       objects,
                       markers,
                       rw_im]
        names = ['Original',
                 'Filled with holes',
                 'Clusters',
                 'Markers',
                 'Bubbles',]
#        df = labeled_props(rw_im,g)
        # go through each image to show and add to its own axes
        for i,(im,name) in enumerate(zip(ims_to_show,names)):
            axs[i].imshow(im,extent=g.im_extent)
            axs[i].set_title(name)
#            add_ellipses_to_ax(df_bubble,axs[i])
#            add_ellipses_to_ax(df_cluster,axs[i])
#            add_label_to_ax(df_bubble,axs[i])
#            add_label_to_ax(df_cluster,axs[i])
            
#T[1].to_excel(r'C:\Users\Luc Deike\Nicolas\StudySurface\Trials\3\Clusters_6\Cluster_6_c.xls')
    return rw_im, df_bubble, df_cluster




def run_bubble_detection(c,thresh,g,frames=None,mask=None,viz=False,filepath=None,method='standard'):


    if viz:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
    if frames is None:
        frames = np.arange(len(c))
    
    if mask is None:
        def mask(im): return im
    
    df_all = pd.DataFrame()
    for i,f in enumerate(frames):
        
        print('frame '+str(f))
            
        im = mask(c[f]) #- bgs[ci]
        
        if method=='standard':
        
            filled = get_filled(im,thresh)
            df = filled2regionpropsdf(filled,g=g,frame=f)
            
        elif method=='watershed':
            
            ws = watershed_detection(im,thresh,g,RadiusDiskMean=2,viz=False)
            df = labeled_props(ws,g,frame=f)
            
        elif method=='random_walker':
            rw = random_walker_detection(im,thresh,g,viz=False)
            df = labeled_props(rw,g,frame=f)
            
        df = df[df['radius']>0.0001]
        df = df[df['radius']<0.05] 
        print('... found '+str(len(df))+' objects.')
        df_all = pd.concat([df_all,df])
        
        if viz:        
            ax.clear()
            show_and_annotate(im,g,df,ax=ax,vmin=0,vmax=1000)
            
        if (i%100==0) and (i>0) and (filepath is not None):
            print('...saving, on iteration '+str(i)+' and frame '+str(f)+'.')
            df_all.to_pickle(filepath)

    df_all['time'] = df_all['frame']*dt
    print('Bubble detection complete!')
    
    if filepath is not None:
        print('Saving df_all to '+filepath)
        df_all.to_pickle(filepath)
        
    return df_all


'''
Functions for analyzing the computed trajectories
'''

def make_list_of_bubbles(tracked,min_length=None):
    
    # separate the tracked df into many dfs, each corresponding to one bubble
    bubbles = [tracked[tracked['particle']==p].set_index('time') for p in tracked['particle'].unique()]
    
    # filter out the short ones if necessary
    if min_length is not None:
        bubbles = [b for b in bubbles if len(b)>min_length]
        
    # add to each df a column "rel_time" of time since the bubble enters
    for b in bubbles:
        b['rel_time'] = b.index - b.index[0]
    
    return bubbles

def interp_df_by_frame(df,dt,dt_frames=1):
    df_reindexed = df.copy().reindex(np.arange(min(df['frame']),max(df['frame'])+1,dt_frames)*dt).interpolate().ffill().bfill()
    return df_reindexed

def terminate_once_static(df,dur_frames,dist_thresh):
    '''
    Function to chop off the last part of a dataframe after the rolling max 
    distance between points is less than some threshold distance over some 
    number of frames
    '''
    
    len_orig = len(df)
    
    for i in df['frame']:
        '''
        At each frame, check the max distance moved over the past dur_frames 
        frames.
        '''      
        
        # construct a dataframe of just the recent points considered
        df_recent = df[(df['frame']<=i) & (df['frame']>i-dur_frames)]
        
        if ((i-df['frame'].min())>=dur_frames) and (max(scipy.spatial.distance.pdist(df_recent[['x','y']].values))<dist_thresh):
            # if this dataframe is long enough and the max distance is too low,
            # cutoff the dataframe after this frame and return it
            df = df[df['frame']<=i]
            print('... cutting off at frame '+str(i)+'. Reduced to '+str(len(df))+' from '+str(len_orig)+' frames.')
            return df
        
    # if this point is reached, no truncating was performed, so just return
    # the dataframe    
    print('... no truncating performed for this bubble, with length '+str(len(df)))
    return df
        
def compute_transient_quantities(b,roll_vel=3):
    '''
    Make new columns for a dataframe corresponding to one tracked bubble. 
    Includes velocity components, etc.
    '''
    
    b['dt'] = np.nan
    b['dt'].iloc[0:-1] = np.diff(b.index)
    
    b['u'] = b['x'].diff()/b['dt']
    b['v'] = b['y'].diff()/b['dt']
    
    b['u'] = b['u'].rolling(window=roll_vel,center=True,min_periods=0).mean()
    b['v'] = b['v'].rolling(window=roll_vel,center=True,min_periods=0).mean()
    
    b['direction'] = np.arctan2(b['v'],b['u'])/np.pi * 180. # not sure why -1 makes the direction match the ellipse orientation
    
    b['slip'] = b['orientation']-b['direction']
    
    return b

def estimate_void_fraction(df,V_total=None):
    if V_total is None:
        V_total = ((df['x'].max()-df['x'].min())*(df['y'].max()-df['y'].min()))**(3./2)
        
    V_bubbles = df['radius']**3 * 4./3. * np.pi
    
    V_bubbles_total = V_bubbles.sum()
    
    return V_bubbles_total / V_total

'''
Functions for visualization
'''

def animate_cine_and_tracked(c,g,bubble_list,figfolder,skip=10,vmin=0,vmax=1000):
    '''
    Save .pngs of cine frames with trajectories and bubbles overlaid.
    '''
    fig = plt.figure(figsize=(14,12))
    ax = fig.add_subplot(111)
    
    frames = range(0,len(c),skip)
    
    for fi,f in enumerate(frames):
        print('frame '+str(f)+'/'+str(len(c)))
        ax.clear()
        ax.imshow(c[f],extent=g.im_extent,cmap='gray',vmin=vmin,vmax=vmax)
        
        for b in [b for b in bubble_list if np.min(np.abs(b['frame']-f))<100]:
            '''
            Only consider bubbles which are in-frame around this frame
            '''
            
            # plot the bubble location just at this frame
            df_this_frame = b[b['frame']==f]
            for idx in df_this_frame.index:
                ax.plot(df_this_frame['x'],df_this_frame['y'],'o',color='r',alpha=0.5)
            
            # plot the trajectory of this bubble up until this frame
            df_up_to_frame = b[b['frame']<=f]
            ax.plot(df_up_to_frame['x'],df_up_to_frame['y'],color='b',alpha=0.5)
            
        # save the figure to the specified folder
        fig.savefig(figfolder+'frame_'+str(f)+'.png')
        
def add_ellipses_to_ax(df,ax,color=[1,0,0,0.5]):
    for ix in df.index:
        e = Ellipse([df.loc[ix,'x'],df.loc[ix,'y']],width=df.loc[ix,'major_axis_length'],height=df.loc[ix,'minor_axis_length'],angle=df.loc[ix,'orientation'])
        ax.add_artist(e)
        e.set_facecolor('None')
        e.set_edgecolor(color)
        
def add_text_to_ax(df,ax):
    for ix in df.index:
        if 'cluster' in df.columns :
#            ax.text(df.loc[ix,'x'],df.loc[ix,'y'],str(ix)+'.'+'r'+str(df.loc[ix,'radius'])+'f_a'+str(df.loc[ix,'filled_area'])+'c'+str(df.loc[ix,'cluster']))
            ax.text(df.loc[ix,'x'],df.loc[ix,'y'],'B'+str(ix)+'.C'+str(df.loc[ix,'cluster'])+'f_a'+str(int(df.loc[ix,'filled_area'])),fontsize=8)
        if 'bubble' in df.columns :
#            ax.text(df.loc[ix,'x'],df.loc[ix,'y'],str(ix)+'.'+'r'+str(df.loc[ix,'radius'])+'f_a'+str(df.loc[ix,'filled_area'])+'b'+str(df.loc[ix,'bubble']))
            ax.text(df.loc[ix,'x'],df.loc[ix,'y'],'C'+str(ix)+'.f_a'+str(int(df.loc[ix,'filled_area']))+'b'+str(df.loc[ix,'bubble'].tolist()),fontsize=13)

def add_label_to_ax(df,ax):
    for ix in df.index:
        if 'cluster' in df.columns :
#            ax.text(df.loc[ix,'x'],df.loc[ix,'y'],str(ix)+'.'+'r'+str(df.loc[ix,'radius'])+'f_a'+str(df.loc[ix,'filled_area'])+'c'+str(df.loc[ix,'cluster']))
            ax.text(df.loc[ix,'x'],df.loc[ix,'y'],'B'+str(ix)+'.C'+str(df.loc[ix,'cluster']),fontsize=8)
        if 'bubble' in df.columns :
#            ax.text(df.loc[ix,'x'],df.loc[ix,'y'],str(ix)+'.'+'r'+str(df.loc[ix,'radius'])+'f_a'+str(df.loc[ix,'filled_area'])+'b'+str(df.loc[ix,'bubble']))
            ax.text(df.loc[ix,'x'],df.loc[ix,'y'],'C'+str(ix),fontsize=13)




def show_and_annotate(im,g,df,ax=None,vmin=0,vmax=1000):
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
    ax.imshow(im,cmap='gray',vmin=0,vmax=1000,extent=g.im_extent)
    
    add_ellipses_to_ax(df,ax,color=[1,0,0,0.5])
    
    return ax
        
    #plt.show()
    #plt.pause(.1)

class FilteringCriteria:
    '''
    Criteria with which to get rid of bad bubbles
    '''
    
    def __init__(self):
        self.criteria = {'min_vert_span':0.05,
                        'min_time':0.1,
                        'min_med_radius':0.5}
        return
    
    def apply_criteria(self,bubble_metadata,tracking_data):
        filtered_bubbles = bubble_metadata.copy()
        
        '''
        Loop through each bubble and see if it meets all the criteria
        '''
        
        for bi in bubble_metadata.index():
            meets_criteria = True
            
            while meets_criteria==True:
                return
            
class VisualizationScheme:
    
    def __init__(self):
        self.mapping = {'color':None,'ls':None,'ms':None,'marker':None}
        return
    
def scatter_with_rolling(df,x,y,identifying_param='intns'):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        colors=['k','b','g','orange','r']
        cdict = {c:colors[ci] for ci,c in enumerate(np.sort(to_use[identifying_param].unique()))}
        
        for val in np.sort(df[identifying_param].unique()):
            to_use1 = df[df[identifying_param]==val]
            to_use1 = to_use1.sort_values(x).rolling(window=10,center=True).mean()
            ax.plot(to_use1[x],to_use1[y],color=cdict[val],label=str(val))
            
        ax.scatter(to_use[x],to_use[y],c=[cdict[i] for i in to_use[identifying_param]],alpha=0.2)
        
        ax.legend(title=identifying_param)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        
def alpha_binary_cmap(x,map_orig,alpha=0.2):
    # https://stackoverflow.com/questions/2495656/variable-alpha-blending-in-pylab
    m = map_orig(x)
    m[:,:,3] = x*alpha
    return m
    
def oscillation_characteristics(t):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    smoothed = t.rolling(window=5,center=True,min_periods=0).mean()
    
    ax.plot(t.x,t.y)
    
    ax.plot(smoothed['x'],smoothed['y'],'--')
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.plot(t['u'],t.index)
    
    sign_change = np.sign(t['u'].iloc[0:-1]) * np.sign(t['u'].iloc[1:])
    
    ax.plot(sign_change)

#class BubbleDataset:
#    
#    def __init__(self,dict_by_cine,metadata):
#        '''
#        dict_by_cine[arbitrary_int] = dict of bubbles tracked in this cine
#        
#        metadata.loc[arbitrary_int,:] = metadata for this cine
#        
#        Cines are now referred to by their arbitrary index, not the filename 
#        (which may not be unique).
#        '''
#        
#        self.dict_by_cine = dict_by_cine # dict_by_cine[cine][bubble]
#        #self.metadata = metadata
#        
#        self.metadata = metadata
#        
#        
#    def split_to_bubbles(self):
#        
#        '''
#        The dataset is initialized with a number of groups of bubble data,
#        corresponding to the movie in which the bubbles are filmed.
#        
#        Create one dict containing all the bubbles' trajectories, and one df
#        with all the bubbles' metadata.
#        '''
#        
#        all_bubbles = pd.DataFrame()
#        self.tracking_data = dict()
#        bi = -1 # will index all_bubbles
#        
#        for ix in self.metadata.index:
#            '''
#            Top level of the loop is by cine. Each cine can contain multiple
#            bubbles.
#            '''
#            
#            #cine_filename = self.metadata.loc[ix,'cine_filename']
#            
#            meta = self.metadata.loc[ix,:]
#            
#            cine_data = self.dict_by_cine[ix]
#            
#            for bubble_num in list(cine_data.keys()):
#                '''
#                Extract info on each bubble from this cine
#                '''
#                
#                bi = bi+1
#                
#                tracking_data = cine_data[bubble_num]
#                tracking_data = tracking_data[tracking_data['y']>0.13]
#                tracking_data = tracking_data[tracking_data['y']<=0.23]
#                index = pd.Series(data=tracking_data.index,index=tracking_data.index)
#                for col in [['x','u'],['y','v']]:
#                    tracking_data[col[1]] = tracking_data[col[0]].diff()/index.diff()
#                self.tracking_data[bi] = tracking_data
#                
#                all_bubbles.set_value(bi,'cine_indx',ix)
#                [all_bubbles.set_value(bi,col,meta[col]) for col in list(metadata.columns)]                
#                [all_bubbles.set_value(bi,'med_'+col,tracking_data[col].median())  for col in ['radius','v']]
#                all_bubbles.set_value(bi,'med_u',abs(tracking_data['u']).median())
#                [all_bubbles.set_value(bi,'std_'+col,tracking_data[col].std())  for col in ['radius','u','v']]
#                [all_bubbles.set_value(bi,'range_'+col,tracking_data[col].max()-tracking_data[col].min())  for col in ['radius','x','y']]
#                
#        self.all_bubbles = all_bubbles
#        
#if __name__=='__main__':
#    
#    if False:
#        
#        parent_folder = r'E:\Experiments_Stephane\Grid column\Backlight_bubbles\20171015\\'
#        filenames = [r'Backlight_bubble_fps500_sv_Needle_grid3tx2x20_noCap_Cam_20861_Cine'+str(i) for i in range(1,61)]
#        
#        '''
#        Construct the metadata
#        '''
#        metadata = pd.DataFrame(index=np.arange(0,60))
#        metadata['cine_filename'] = filenames
#        metadata['A'] = [0]*10+[2]*5+[4]*5+[6]*5+[8]*5+[10]*5+[1]*5+[2]*5+[4]*5+[6]*5+[8]*5
#        metadata['freq'] = [0]*10+[5]*25+[10]*25
#        
#        '''
#        Construct the dict of data from individual cines
#        '''    
#        dict_by_cine = {fi:pd.read_pickle(parent_folder+f+r'_trackedParticles.pkl') for fi,f in enumerate(filenames)}
#        
#    if True:
#        
#        parent_folder = r'E:\Experiments_Stephane\Grid column\Backlight_bubbles\20171020\\'
#        filepaths = [parent_folder+r'backlight_bubbles_sv_grid3tx2x20_Cam_20861_Cine'+str(i) for i in range(1,61)]
#        
#        parent_folder = r'E:\Experiments_Stephane\Grid column\Backlight_bubbles\20171025\\'
#        filepaths = filepaths+[parent_folder+r'backlight_bubbles_sv_grid3tx2x20_171025_Cam_20861_Cine'+str(i) for i in range(1,61)]
#        filepaths = filepaths+[parent_folder+r'backlight_bubbles_sv_grid3tx2x20_171025b_Cam_20861_Cine'+str(i) for i in range(1,61)]
#        
#        parent_folder=r'E:\Experiments_Stephane\Grid column\Backlight_bubbles\20171028\\'
#        filepaths = filepaths+[parent_folder+r'backlight_bubbles_sv_grid3tx2x20_171028_noTurbulence_Cam_20861_Cine'+str(i) for i in range(1,61)]
#        
#        metadata = pd.DataFrame(index=np.arange(0,240))
#        metadata['cine_filename'] = filepaths
#        metadata['A'] = [0]*20 + [2]*20 + [4]*20 + [6]*30 + [10]*30 + [2]*30 + [4]*30 + [0]*60
#        metadata['freq'] = [0]*20 + [10]*20 + [10]*20 + [5]*30 + [5]*30 + [10]*30 + [10]*30 + [0]*60
#        
#        dict_by_cine = {fi:pd.read_pickle(f+r'_trackedParticles.pkl') for fi,f in enumerate(filepaths)}
#        
#    if False:
#        
#        parent_folder = r'E:\Experiments_Stephane\Grid column\Backlight_bubbles\20171014\\'
#        filenames = [r'Backlight_bubble_fps2000_svCloseUp_Needle_grid2tx20_noCap_Cam_20861_Cine'+str(i) for i in range(1,31)]
#        
#        metadata = pd.DataFrame(index=np.arange(0,30))
#        metadata['cine_filename'] = filenames
#        metadata['A'] = [0]*10 + [2]*10 + [4]*10
#        metadata['freq'] = [10]*10 + [10]*10 + [10]*10
#        
#        dict_by_cine = {fi:pd.read_pickle(parent_folder+f+r'_trackedParticles.pkl') for fi,f in enumerate(filenames)}
#    
#    
#    '''
#    Compile the dataset
#    '''
#    d = BubbleDataset(dict_by_cine,metadata)
#    d.split_to_bubbles()
#    
#    '''
#    Filter out the bubbles into a new dataframe, and add a column for the turbulence
#    '''
#    to_use = d.all_bubbles.copy()
#    to_use = to_use[to_use['range_y']>0.05]        
#    to_use['intns'] = to_use['A']*to_use['freq']   
#    
#    '''
#    Make some plots
#    '''        
#    scatter_with_rolling(to_use,'med_radius','med_v')
#    scatter_with_rolling(to_use,'med_radius','med_u')
#    scatter_with_rolling(to_use,'med_radius','range_x')
#    scatter_with_rolling(to_use,'med_radius','range_radius')
#    
#    
#    #for fi,f in enumerate(to_use['freq'].unique()):
#    #    to_use_now = to_use[to_use['freq']==f]
#    #    plt.scatter(to_use_now['med_radius'],to_use_now['med_v'],c=to_use_now['A'],marker=style_list[fi])
#    
#    

    
    
    