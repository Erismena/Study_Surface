# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 15:29:16 2018
"""

#import sys
##print(sys.path)
#sys.path.append(r"C:\Users\Luc Deike\Documents\GitHub\2d-fluids-analysis")

#print(sys.path)

import matplotlib
from matplotlib.contour import ContourSet
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import fluids2d.geometry
import fluids2d.backlight as backlight
import pims
import scipy.ndimage
import pandas as pd
from scipy.ndimage.morphology import binary_fill_holes
import skimage.measure
import skimage.filters
from matplotlib import cm
import trackpy as tp
import skimage
from sklearn.cluster import spectral_clustering
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from fluids2d.backlight import labeled_props
from fluids2d.backlight import filled2regionpropsdf
from random import randint
import itertools
import random

#c = pims.open(r'C:\Users\Luc Deike\Nicolas\StudySurface\Trials\3\Clusters_6\*.tiff')
c = pims.open(r'C:\Users\Luc Deike\Nicolas\StudySurface\5_7_18\Needle0o_P1560_T2750_1360sccm_waterfewhours_50fps_3exposuretime_Tlab236_hum39.cine')
w = pims.open(r'C:\Users\Luc Deike\Nicolas\StudySurface\5_7_18\metricNeedleo.cine')

z = pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\5_11_18\Needle04_50fps_10000expt_1500ssm_P2000_T2327_Tlab227_hum36_water01day\*.tiff')
metric = pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\5_11_18\metric\Basler acA2040-90um (22332433)_20180511_142725956_0001.tiff')
cN0B = pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\5_11_18\Needle0B_50fps_10000expt_1500ssm_P1260_T2695_Tlab231_hum34_water01day\*.tiff')
cN03 = pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\5_11_18\Needle03_50fps_10000expt_1500ssm_P1263_T2736_Tlab238_hum32_water01day\*.tiff')
cN02 = pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\5_11_18\Needle02_50fps_1000expt_1500ssm_P1263_T2736_Tlab238_hum31_water01day\*.tiff')
cN0o = pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\5_11_18\Needle0o_50fps_1000expt_1371ssm_P1223_T2495_Tlab226_hum42_water04day\*.tiff')
cN01 = pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\5_11_18\Needle01_50fps_100expt_1516ssm_P1591_T2763_Tlab231_hum42_water04day\*.tiff')

thresh = 200
dt = 0.02
dx= 0.057990
im_shape=(2048,2048)
dxNeedle01et02= 0.0510
dxNeedle01et02= 0.0570

#thresh_cluster=150
RadiusDiskMean=1
pts=[[502,401],[1670,403],[1682,1374],[419,1367]]

MinRadius=0.0007
MaxRadius=1000
MinRadius_holes=0.0005

MaskBottom=1828
MaskTop=550
MaskRight=1700
MaskLeft=950

#cerlce des erreurs dues au recouvement des images
#Centre_erreur=[0,0]
search_range_trackage=0.005

g = fluids2d.geometry.GeometryScaler(dx=dx,im_shape=(2048,2048),origin_pos=(0,0),origin_units='pix')

frames = np.arange(0,3994,1)


def mask(im):
    im[:,:MaskLeft] = 255
    im[:,MaskRight:] = 255
    im[:MaskTop,:] = 255
    im[MaskBottom:,:] = 255
    im[300:410,1568:1750] = 255
    im[1840:2000,1450:1750] = 255
    return im

def get_filled(im,thresh):
    im_filt = scipy.ndimage.filters.median_filter(im,size=3)
    im_filt = backlight.binarize(im_filt,thresh,large_true=False)
    filled = binary_fill_holes(im_filt)
    return filled

def get_binarize(im,thresh,size1=2,size2=2):
    im_filt = scipy.ndimage.filters.median_filter(im,size=size1)
    im_filt = backlight.binarize(im_filt,thresh,large_true=False)
    im_filt = scipy.ndimage.filters.median_filter(im_filt,size=size2)
    return im_filt


def contours(im,viz=False):
    if viz:
        plt.imshow(im)
    N=np.arange(1,np.max(im)+1,1)
#    P=[[0]*im.shape[0]]*im.shape[0]
    CT=[np.nan]*np.max(im)
    for i,j in enumerate(N):
        solo=np.copy(im)
        solo[im!=j]=0
        if np.all(solo<0.5):
            CT[j-1]=np.nan
        else:
            CT[j-1] = measure.find_contours(solo,level=0,fully_connected='low')
            if viz:
                plt.plot(CT[j-1][0][:,1],CT[j-1][0][:,0])
    return CT
        
            

def filled_props(filled,g):
    objects=filled
    num_objects = np.max(filled)
    props = skimage.measure.regionprops(objects)    
    df = pd.DataFrame(columns=['frame','radius','x','y'])
    if (len(props)==0)==False:
        for ri,region in enumerate(props):
            y,x = g.get_loc([region.centroid[0],region.centroid[1]])
            df.loc[ri,'y']=y[0]
            df.loc[ri,'x']=x[0]
            df.loc[ri,'radius'] = np.sqrt(region.filled_area * dx**2 / np.pi)
            df.loc[ri,'orientation'] = region.orientation / (2*np.pi) * 360
            df.loc[ri,'major_axis_length'] = region.major_axis_length* dx
            df.loc[ri,'minor_axis_length'] = region.minor_axis_length* dx
            df.loc[ri,'eccentricity'] = region.eccentricity
    return df

def p(im):
    plt.figure()
    plt.imshow(im)

plt.figure()
plt.imshow()
plt.title()

len_cine = len(frames)

fig = plt.figure()
ax = fig.add_subplot(111)

df_all = pd.DataFrame()

def traitement(c,thresh,g,frames,method='standard',MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean):
    
    df_all = pd.DataFrame()
    df_all_bubble = pd.DataFrame()
    df_all_cluster = pd.DataFrame()
    
    for i,f in enumerate(frames):
        print('frame '+str(f))
        im=mask(c[f])
        print('. get_filled and labeled')
        
        if method=='standard':
            filled = get_filled(im,thresh)
            df = filled2regionpropsdf(filled,g=g,frame=f)
            df = df[df['radius']>MinRadius]
            df = df[df['radius']<MaxRadius]
            print('... found '+str(len(df))+' objects.')
            df_all = pd.concat([df_all,df])
            
        elif method=='watershed':
            ws = watershed_detection(im,thresh,g,RadiusDiskMean=2,viz=False)
            df = labeled_props(ws,g,frame=f)
            df = df[df['radius']>3]
            df = df[df['radius']<200]
            print('... found '+str(len(df))+' objects.')
            df_all = pd.concat([df_all,df])
            
        elif method=='random_walker':
            rw=random_walker_detection(im,thresh,g,mode='cg',RadiusDiskMean=2,tol=0.01,viz=True)
            df = labeled_props(rw,g,frame=f)
            df = df[df['radius']>3]
            df = df[df['radius']<200]
            print('... found '+str(len(df))+' objects.')
            df_all = pd.concat([df_all,df])
            
        elif method=='random_walker_detection_cluster_dist':
            rw_cluster=random_walker_detection_cluster_dist(im,thresh,g,frame=f,MinRadius=MinRadius,MaxRadius=MaxRadius,mode='cg',RadiusDiskMean=RadiusDiskMean,tol=0.01,viz=False)
            df_bubble=rw_cluster[1]
            print('... found '+str(len(df_bubble))+' bubbles.')
            df_cluster=rw_cluster[2]
            print('... found '+str(len(df_cluster))+' clusters.')
            df_all_bubble = pd.concat([df_all_bubble,df_bubble])
            df_all_cluster = pd.concat([df_all_cluster,df_cluster])
            
        elif method=='random_walker_detection_cluster_holes':
            rw_cluster=random_walker_detection_cluster_holes(im,thresh,g,frame=f,MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,mode='cg',RadiusDiskMean=RadiusDiskMean,tol=0.01,viz_process=False,viz=False)
            df_bubble=rw_cluster[1]
            print('... found '+str(len(df_bubble))+' bubbles.')
            df_cluster=rw_cluster[2]
            print('... found '+str(len(df_cluster))+' clusters.')
            df_all_bubble = pd.concat([df_all_bubble,df_bubble])
            df_all_cluster = pd.concat([df_all_cluster,df_cluster])

    if method=='random_walker_detection_cluster_dist':
        df_all_bubble['time'] = df_all_bubble['frame']*dt
        df_all_cluster['time'] = df_all_cluster['frame']*dt
        return df_all_bubble,df_all_cluster
    
    if method=='random_walker_detection_cluster_holes':
        df_all_bubble['time'] = df_all_bubble['frame']*dt
        df_all_cluster['time'] = df_all_cluster['frame']*dt
        return df_all_bubble,df_all_cluster
    
    else:
        df_all['time'] = df_all['frame']*dt
        return df_all



tracked=tp.link_df(df_all_cluster,search_range=80,memory=3,search_range_trackage)


df_all['time'] = df_all['frame']*dt        
df_filtered = df_all[df_all['radius']>1]
df_filtered = df_filtered[df_filtered['radius']<200]

def trackage(df_bubble,df_cluster,search_range=40,memory=5,search_range_trackage=search_range_trackage):

    #    tracked_bubble=tp.link_df(df_bubble,search_range=search_range,memory=memory)
    tracked_cluster=tp.link_df(df_cluster,search_range=search_range,memory=memory)
    
    tracked_cluster['num_bubble_diff']=tracked_cluster['num_of_bubble'].diff()
    for p in tracked_cluster['particle'].unique().astype(int):    
        tracked_cluster.loc[tracked_cluster[tracked_cluster['particle']==p].index[0],'num_bubble_diff']=np.nan
    
    tracked_cluster['birth_frame']=np.nan
    tracked_cluster['death_frame']=np.nan
    for p in tracked_cluster['particle'].unique().astype(int):    
        tracked_cluster.loc[tracked_cluster[tracked_cluster['particle']==p].index[0],'birth_frame']=tracked_cluster.loc[tracked_cluster[tracked_cluster['particle']==p].index[0],'frame']
        tracked_cluster.loc[tracked_cluster[tracked_cluster['particle']==p].index[-1],'death_frame']=tracked_cluster.loc[tracked_cluster[tracked_cluster['particle']==p].index[-1],'frame']

    tracked_cluster_0 = pd.DataFrame(columns=['event'], index=df_cluster.index,dtype=object)
    tracked_cluster['event']=tracked_cluster_0
    tracked_cluster_1 = pd.DataFrame(columns=['integrate'], index=df_cluster.index,dtype=object)
    tracked_cluster['integrate']=tracked_cluster_1
    tracked_cluster_2 = pd.DataFrame(columns=['get_into_particle'], index=df_cluster.index,dtype=object)
    tracked_cluster['get_into_particle']=tracked_cluster_2

    for p in tracked_cluster['particle'].unique().astype(int):
        print('particle'+str(p))
        for k in tracked_cluster[tracked_cluster['particle']==p].index:
            print('index'+str(k))
            
            if tracked_cluster[tracked_cluster['particle']==p].loc[k,'num_bubble_diff']==0:
                tracked_cluster[tracked_cluster['particle']==p].loc[k,'event']=np.nan
            
            elif tracked_cluster[tracked_cluster['particle']==p].loc[k,'num_bubble_diff']>0:
                nb=tracked_cluster[tracked_cluster['particle']==p].loc[k,'num_bubble_diff']
                frame=tracked_cluster[tracked_cluster['particle']==p].loc[k,'frame']
                if frame!=0:
                    num_cluster_dead_prior_frame=len(tracked_cluster[tracked_cluster['death_frame']==frame-1])
                    index_cluster_dead_prior_frame=list(tracked_cluster[tracked_cluster['death_frame']==frame-1].index)
                    for j in index_cluster_dead_prior_frame:
                        if Is_inside_cercle(x=tracked_cluster.loc[j,'x'],y=tracked_cluster.loc[j,'y'],centre_x=tracked_cluster.loc[k-1,'x'],centre_y=tracked_cluster.loc[k-1,'y'],radius=tracked_cluster.loc[k-1,'major_axis_length']/2+search_range_trackage)==False:
                            index_cluster_dead_prior_frame.remove(j)
                    growth_of_cluster=tracked_cluster[tracked_cluster['particle']==p].loc[k,'filled_area']-tracked_cluster[tracked_cluster['particle']==p].loc[k-1,'filled_area']
#                    if len(index_cluster_dead_prior_frame)<=nb:
                    comb=combinations(list(index_cluster_dead_prior_frame),r_max=nb)
                    comb=[comb,[0]*len(comb)]
                    for i in np.arange(0,np.shape(comb)[1],1):
                        aire=0
                        for m in comb[0][i]:
                            aire=aire+tracked_cluster.loc[m,'filled_area']
                        comb[1][i]=abs(aire-growth_of_cluster)
                    indice_comb_cluster=np.argmin(comb[1])
                    cluster_get_in_Cluster=comb[0][indice_comb_cluster]
                    print('cluster_get_in_Cluster'+str(cluster_get_in_Cluster))
                    cluster_dead_prior_frame=list(set(list(index_cluster_dead_prior_frame))-set(list(cluster_get_in_Cluster)))                    
                    print('cluster_dead_prior_frame'+str(cluster_dead_prior_frame))
                    
                    if len(cluster_get_in_Cluster)==0:
                        tracked_cluster.loc[k,'event']='integrate_new_cluster'

                    if len(cluster_get_in_Cluster)!=0:
                        tracked_cluster.loc[k,'event']='integrate_new_cluster'
                        tracked_cluster.loc[k,'integrate']=cluster_get_in_Cluster                    
                    
                    for t in cluster_get_in_Cluster:
                        tracked_cluster.loc[t,'event']='get_into'
                        tracked_cluster.loc[t,'get_into_particle']=p
                    for y in cluster_dead_prior_frame:
                        tracked_cluster.loc[y,'event']='definitive_death'

    return tracked_cluster



            elif tracked_cluster[tracked_cluster['particle']==p].loc[k,'num_bubble_diff']<0:
                nb=tracked_cluster[tracked_cluster['particle']==p].loc[k,'num_bubble_diff']
                frame=tracked_cluster[tracked_cluster['particle']==p].loc[k,'frame']
                if frame!=0:
                    index_bubble_frame=tracked_cluster[tracked_cluster['particle']==p].loc[k,'bubble']
                    index_bubble_previous_frame=tracked_cluster[tracked_cluster['particle']==p].loc[k-1,'bubble']                   
                    
                    



                    #identification des bulles aui ne sont ni sorties ni coalesce
                    table_correspondance_area=[[np.nan]*len(list(index_bubble_previous_frame))]*len(list(index_bubble_frame))
                    table_correspondance_dist_relative=[[np.nan]*len(list(index_bubble_previous_frame))]*len(list(index_bubble_frame))
                    tran_btw_frames=[tracked_cluster[tracked_cluster['particle']==p].loc[k,'x']-tracked_cluster[tracked_cluster['particle']==p].loc[k-1,'x'],tracked_cluster[tracked_cluster['particle']==p].loc[k,'y']-tracked_cluster[tracked_cluster['particle']==p].loc[k-1,'y']]

                    for i in np.arange(0,len(list(index_bubble_frame)),1):
                        for j in np.arange(0,len(list(index_bubble_previous_frame)),1):
                            table_correspondance_area[i][j]=abs(df_bubble[df_bubble['frame']==frame].loc[i,'filled_area']-df_bubble[df_bubble['frame']==frame-1].loc[j,'filled_area'])
                            table_correspondance_dist_relative[i][j]=np.sqrt((df_bubble[df_bubble['frame']==frame].loc[i,'x']-tran_btw_frames[0]-df_bubble[df_bubble['frame']==frame-1].loc[j,'x'])**2+(df_bubble[df_bubble['frame']==frame].loc[i,'y']-tran_btw_frames[1]-df_bubble[df_bubble['frame']==frame-1].loc[j,'y'])**2)
                    proba_table_correspondance_area=table_correspondance_area
                    proba_table_correspondance_area[j]=





                        
                        
                        
                        
#            elif tracked_cluster[tracked_cluster['particle']==p].loc[k,'num_bubble_diff']<0:
            

#    if viz:
#        tracked_bubble['color']=''
#        tracked_cluster['color']=''
##        for p in tracked_bubble['particle'].unique():
##            co = '#{:06x}'.format(randint(0, 256**3))
##            for k in tracked_bubble[tracked_bubble['particle']==p].index:    
##                tracked_bubble[tracked_bubble['particle']==p].loc[k,'color']=co
#
#        "couleurs par bubble ou cluster"
#        
#        co_bubble=['']*len(tracked_bubble['particle'].unique())
#        for i in tracked_bubble['particle'].unique().astype(int):
#            co_bubble[i] = '#{:06x}'.format(randint(0, 256**3))
#        
#        co_cluster=['']*len(tracked_cluster['particle'].unique())
#        for i in tracked_cluster['particle'].unique().astype(int):
#            co_cluster[i] = '#{:06x}'.format(randint(0, 256**3))
#   
#        "plot contour"
#        
#        fig_bu = plt.figure()
#        plt.title('evolution des bubbles')
#        ax = fig_bu.add_subplot(111)
#        for p in tracked_bubble['particle'].unique().astype(int):
#            for l in tracked_bubble[tracked_bubble['particle']==p].index:
#                ax.plot(tracked_bubble[tracked_bubble['particle']==p].loc[l,'contours'][:,1],tracked_bubble[tracked_bubble['particle']==p].loc[l,'contours'][:,0], color=co_bubble[p])
#            ax.plot(tracked_bubble[tracked_bubble['particle']==p]['x'],tracked_bubble[tracked_bubble['particle']==p]['y'], color=co_bubble[p])
#        
#        fig_cl = plt.figure()
#        plt.title('evolution des clusters')
#        ax = fig_cl.add_subplot(111)
#        for p in tracked_cluster['particle'].unique().astype(int):
#            for l in tracked_cluster[tracked_cluster['particle']==p].index:
#                ax.plot(tracked_cluster[tracked_cluster['particle']==p].loc[l,'contours'][:,1],tracked_cluster[tracked_cluster['particle']==p].loc[l,'contours'][:,0], color=co_cluster[p], linestyle='solid')
#            ax.plot(tracked_cluster[tracked_cluster['particle']==p]['x'],tracked_cluster[tracked_cluster['particle']==p]['y'], color=co_cluster[p],linestyle='dashdot')
#
#        "plot trajectoire"
#        
#        fig_traj_cl = plt.figure()
#        plt.title('paths cluster')
#        ax = fig_traj_cl.add_subplot(111)
#        for p in tracked_cluster['particle'].unique().astype(int):
#            ax.plot(tracked_cluster[tracked_cluster['particle']==p]['x'],tracked_cluster[tracked_cluster['particle']==p]['y'], color=co_cluster[p], label=p)
#            ax.legend()
def LifeTime(Df00, RadiusMin=0,RadiusMax=10,LifetimeMin=0,LifetimeMax=10, style='.',viz=False):
#    tracked=tp.link_df(Df,search_range=80,memory=3)
#    Ra=plt.figure()
#########################this next ligne is used to determine Rmin and Rmax and the lifetime min
#    plotRfctTperP(Df00, style='+')
    Df00['lifetime']=Df00['particle']
    for p in Df00['particle'].unique():
        index=Df00[Df00['particle']==p].index
        lifetime=len(Df00[Df00['particle']==p])*dt
        for i in index:
            Df00.loc[i,'lifetime']=lifetime
    Df00.sort_values(by='particle')
    
    'tri'
    Df00=Df00[Df00['radius']>RadiusMin]
    Df00=Df00[Df00['radius']<RadiusMax]
    Df00=Df00[Df00['lifetime']>LifetimeMin]
    Df00=Df00[Df00['lifetime']<LifetimeMax]
    if viz:
        plotRfctTperP(Df00, style=style)
    return Df00

def Histogramme(DfA,DfB,DfC,DfD,DfE,DfF,DfG):
    H=plt.figure()
    A=[DfA[DfA['particle']==p].iloc[0]['lifetime'] for p in DfA['particle'].unique()]
    B=[DfB[DfB['particle']==p].iloc[0]['lifetime'] for p in DfB['particle'].unique()]
    C=[DfC[DfC['particle']==p].iloc[0]['lifetime'] for p in DfC['particle'].unique()]
    D=[DfD[DfD['particle']==p].iloc[0]['lifetime'] for p in DfD['particle'].unique()]
    E=[DfE[DfE['particle']==p].iloc[0]['lifetime'] for p in DfE['particle'].unique()]+[DfF[DfF['particle']==p].iloc[0]['lifetime'] for p in DfF['particle'].unique()]
    G=[DfG[DfG['particle']==p].iloc[0]['lifetime'] for p in DfG['particle'].unique()]
#    F=[DfF[DfF['particle']==p].iloc[0]['lifetime'] for p in DfF['particle'].unique()]
    
    plt.hist(A, len(list(set(A))),histtype='step',label='Needle01',normed = True)
    plt.hist(B, len(list(set(B))),histtype='step',label='Needle02',normed = True)
    plt.hist(C, len(list(set(C))),histtype='step',label='Needle03',normed = True)
    plt.hist(D, len(list(set(D))),histtype='step',label='Needle04',normed = True)
    plt.hist(E, len(list(set(E))),histtype='step',label='Needle0V',normed = True)
    plt.hist(G, len(list(set(G))),histtype='step',label='Needle0orange',normed = True)

#    plt.hist(F, len(list(set(F))),histtype='step',label='Needle0Vsample02')
    plt.title('Histogram of the life time of the bubble')
    plt.legend()
    return A

def Histogrammeradius(DfA,DfB,DfC,DfD,DfE,DfF,DfG):
    H=plt.figure()
    A=[DfA[DfA['particle']==p].iloc[0]['lifetime'] for p in DfA['particle'].unique()]
    B=[DfB[DfB['particle']==p].iloc[0]['lifetime'] for p in DfB['particle'].unique()]
    C=[DfC[DfC['particle']==p].iloc[0]['lifetime'] for p in DfC['particle'].unique()]
    D=[DfD[DfD['particle']==p].iloc[0]['lifetime'] for p in DfD['particle'].unique()]
    E=[DfE[DfE['particle']==p].iloc[0]['lifetime'] for p in DfE['particle'].unique()]+[DfF[DfF['particle']==p].iloc[0]['lifetime'] for p in DfF['particle'].unique()]
    G=[DfG[DfG['particle']==p].iloc[0]['lifetime'] for p in DfG['particle'].unique()]
#    F=[DfF[DfF['particle']==p].iloc[0]['lifetime'] for p in DfF['particle'].unique()]
    
    plt.hist(A, len(list(set(A))),histtype='step',label='Needle01')
    plt.hist(B, len(list(set(B))),histtype='step',label='Needle02')
    plt.hist(C, len(list(set(C))),histtype='step',label='Needle03')
    plt.hist(D, len(list(set(D))),histtype='step',label='Needle04')
    plt.hist(E, len(list(set(E))),histtype='step',label='Needle0V')
    plt.hist(G, len(list(set(G))),histtype='step',label='Needle0orange')

#    plt.hist(F, len(list(set(F))),histtype='step',label='Needle0Vsample02')
    plt.title('Histogram of the life time of the bubble')
    plt.legend()
    return A




def histogramme(df_bubble,df_cluster,frames):
    
    'Number of bubbles per cluster as a function of time'
    max_num_bubble_per_cluster=max(df_cluster['num_of_bubble'])
    data=np.zeros([max_num_bubble_per_cluster+1,len(frames)],dtype=long)
    data[0]=frames
    for f in np.arange(0,len(frames),1):
        for j in df_cluster[df_cluster['frame']==frames[f]]['num_of_bubble'].value_counts().index:
            data[j][f]=df_cluster[df_cluster['frame']==frames[f]]['num_of_bubble'].value_counts().loc[j]
    bottom=np.zeros_like(data[0])
    hist=plt.figure()
    for i in np.arange(1,max_num_bubble_per_cluster+1,1):
        P=plt.bar(frames,data[i],bottom=bottom,label="num_b_in_c=%d"%(i,),width=1)
        bottom=bottom+data[i]
    plt.legend()
    plt.title('Number of bubbles per cluster as a function of time')
    plt.xlabel('frame')
    plt.ylabel('Number of clusters')
    
#    'Histogramme of the radius of the cluster with a given size as a function of time A REVOOOIIIIIR LOL'
#    fig,axs = plt.subplots(2,2,figsize=(17,9),sharex=True,sharey=False)
#    axs = axs.flatten()
#    datas = [df_cluster[df_cluster['num_of_bubble']==i]['radius'] for i in sorted(df_cluster['num_of_bubble'].unique())]
#    names = ['Radius of bubbles in clusters of '+str(i)+' bubbles' for i in sorted(df_cluster['num_of_bubble'].unique())]
#    for i,(d,name) in enumerate(zip(datas,names)):
#        axs[i].hist(d,bins=500)
#        axs[i].set_title(name)
#        axs[i].set_xlabel('radius')
#        axs[i].set_ylabel('num of bubble')

    list_of_particle_clusters = [df_cluster[df_cluster['particle']==p] for p in df_cluster['particle'].unique()]
    
    'plot trajectories'
    traj = plt.figure()
    plt.title('trajectories of the particles y fct x')
    ax = traj.add_subplot(111)
    for bubble_df in list_of_particle_clusters:
        ax.plot(bubble_df['x'],bubble_df['y'])
        plt.legend()
    
    'plot trajectories'    
    traj = plt.figure()
    plt.title('trajectories of the particles y fct x')
    ax = traj.add_subplot(111)
    for p in Df['particle'].unique():
        ax.plot(Df[Df['particle']==p]['x'],Df[Df['particle']==p]['y'])
    
    'Evolution num bubbles as a function of time for each cluster'
    NBT = plt.figure()
    plt.title('Evolution num bubbles as a function of time for each cluster')
    axNBT = NBT.add_subplot(111)
    for particle_clusters in list_of_particle_clusters:
        axNBT.plot(particle_clusters['time'],particle_clusters['num_of_bubble'],label="particle=%d"%(np.mean(particle_clusters['particle'],)))
    plt.legend()

    'Evolution overall num bubbles AND overall num cluster as a function of time'
    NOBT = plt.figure()
    plt.title('Evolution overall num bubbles AND overall num cluster as a function of time')
    A=df_bubble['time'].unique()
    B=[len(df_bubble[df_bubble['time']==time]) for time in df_bubble['time'].unique()]
    C=[len(df_cluster[df_cluster['time']==time]) for time in df_cluster['time'].unique()]
    D=[float(len(df_bubble[df_bubble['time']==time]))/float(len(df_cluster[df_cluster['time']==time])) for time in df_cluster['time'].unique()]
    
    std_D = np.sqrt(abs(np.asarray(D) - np.mean(D))**2)
    
    axNOBT = NOBT.add_subplot(111)
    axNOBT.plot(A,B,label='bubbles')
    axNOBT.plot(A,C,label='clusters')
    axNOBT.errorbar(A, D, std_D, linestyle='None', marker='x',label='num bubble per cluster')
    plt.legend()

    'Evolution overall num bubbles as a function of time'
    NOBT = plt.figure()
    plt.title('Evolution overall num bubbles as a function of time')
    A=df_bubble['time'].unique()
    B=[len(df_bubble[df_bubble['time']==time]) for time in df_bubble['time'].unique()]
    plt.plot(A,B)  



    'plot radius as a fct of time for each '
    traj2 = plt.figure()
    plt.title('trajectories with various linewidth of the particles y fct x')
    ax2 = traj2.add_subplot(111)
    for bubble_df in list_of_particle_clusters:
        list_of_particle_cst_num_bubble = [bubble_df[bubble_df['num_of_bubble']==p] for p in bubble_df['num_of_bubble'].unique()]
#        c = "%06x" % random.randint(0, 0xFFFFFF)
        for cst_num in list_of_particle_cst_num_bubble:
            linewidth=int(np.mean(cst_num['num_of_bubble']))*1.5
            ax2.plot(cst_num['x'],cst_num['y'],linewidth=linewidth)
    
    'Histogramme of the radius of the cluster with a given size as a function of time A REVOOOIIIIIR LOL'
    fig = plt.plots()
    datas = [df_cluster[df_cluster['num_of_bubble']==i]['radius'] for i in sorted(df_cluster['num_of_bubble'].unique())]
    names = ['Radius of bubbles in clusters of '+str(i)+' bubbles' for i in sorted(df_cluster['num_of_bubble'].unique())]
    for i,(d,name) in enumerate(zip(datas,names)):
        axs[i].hist(d,bins=500)
        axs[i].set_title(name)
        axs[i].set_xlabel('radius')
        axs[i].set_ylabel('num of bubble')
    
    '................plot paper.......................'
    'Histogramme of the radius of the bubbles as a function of time'
    fig, ax = plt.subplots(tight_layout=True)
    hist = ax.hist2d(np.asarray(Z['time'],dtype='float64'),np.asarray(Z['radius'],dtype='float64'),bins=(2737, 60))
    ax.set_title('Histogramme of the radius of the bubbles as a function of time')
    ax.set_xlabel('time')
    ax.set_ylabel('radius')
    plt.colorbar(hist[3],ax=ax)

def plotLifeTimefctRmeanperPall(DfA,DfB,DfC,DfD,DfE,DfF,DfG):
    'Lifetime as a function of the radius mean of each particle for 6 needle'
    LT = plt.figure()
    plt.plot([np.mean(DfA[DfA['particle']==p]['radius']) for p in DfA['particle'].unique()],[len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()],'.',label='Needle01')
    plt.plot([np.mean(DfB[DfB['particle']==p]['radius']) for p in DfB['particle'].unique()],[len(DfB[DfB['particle']==p])*dt for p in DfB['particle'].unique()],'x',label='Needle02')
    plt.plot([np.mean(DfC[DfC['particle']==p]['radius']) for p in DfC['particle'].unique()],[len(DfC[DfC['particle']==p])*dt for p in DfC['particle'].unique()],'<',label='Needle03')
    plt.plot([np.mean(DfD[DfD['particle']==p]['radius']) for p in DfD['particle'].unique()],[len(DfD[DfD['particle']==p])*dt for p in DfD['particle'].unique()],'d',label='Needle04')
    plt.plot([np.mean(DfE[DfE['particle']==p]['radius']) for p in DfE['particle'].unique()],[len(DfE[DfE['particle']==p])*dt for p in DfE['particle'].unique()],'s',label='Needle0VSample01')
    plt.plot([np.mean(DfF[DfF['particle']==p]['radius']) for p in DfF['particle'].unique()],[len(DfF[DfF['particle']==p])*dt for p in DfF['particle'].unique()],'s',label='Needle0VSample02')
    plt.plot([np.mean(DfG[DfG['particle']==p]['radius']) for p in DfG['particle'].unique()],[len(DfG[DfG['particle']==p])*dt for p in DfG['particle'].unique()],'s',label='Needle0orange')

    plt.title('Lifetime as a function of the radius mean of each particle Needle ')
    plt.xlabel('Mean Radius of each particle in mm')
    plt.ylabel('Life time of each particle in s')
    plt.legend()

def plotLifeTimefctRmeanperPone(Df00):
    'Lifetime as a function of the radius mean of each particle'
    LT = plt.figure()
    plt.plot([np.mean(DfA[DfA['particle']==p]['radius']) for p in DfA['particle'].unique()],[len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()],'.',label='Needle01')
    plt.title('Lifetime as a function of the radius mean of each particle Needle ')
    plt.xlabel('Mean Radius of each particle in mm')
    plt.ylabel('Life time of each particle in s')
    plt.legend()
    
def plotRfctTperP(Df, numframe=50000, style='+'):
    'Plot radius fct of time per particle'
    plt.title('Plot radius fct of time per particle')
    plt.xlabel('frame')
    plt.ylabel('radius in mm')
    Df_all=Df[Df['frame']<numframe]
    for p in Df_all['particle'].unique():
        plt.plot(Df_all[Df_all['particle']==p]['frame'],Df_all[Df_all['particle']==p]['radius'],style,label=p)
    
    
    
#    'Velocity'
#    vel,axs = plt.subplots(2,2,figsize=(17,9),sharex=True,sharey=False)
#    axs = axs.flatten()
#    datas = [U[1][U[1]['num_of_bubble']==i][U[1]['U']<0.5] for i in sorted(U[1]['num_of_bubble'].unique())]
#    names = ['Mean velocity of cluster in clusters of '+str(i)+' bubbles as fct of Mean radius' for i in sorted(U[1]['num_of_bubble'].unique())]
#    for i,(d,name) in enumerate(zip(datas,names)):
#        trackedmean=pd.DataFrame(index=d['particle'].unique(),columns=d.columns)
#        for p in d['particle'].unique():
#            for col in ('radius','U'):
#                trackedmean[col][p]=d[d['particle']==p].mean(axis=0)[col]
#        axs[i].plot(d['radius'],d['U'],'bo')
#        axs[i].set_title(name)
#        axs[i].set_xlabel('radius')
#        axs[i].set_ylabel('U')
    
    return data



def radiusbubble(df_bubble,df_cluster):
    df_cluster['list_rad_bu']=df_cluster['bubble']
    for c in df_cluster['frame'].index:
        f=df_cluster.loc[c]['frame']
        
        
def VelocityRadius(DfA,DfB,DfC,DfD,DfE,DfF):
    'Velocity as a fct of radius'
    LT = plt.figure()
    plt.plot(DfA['radius'],DfA['U'],'*',label='NeedleOrange')
    plt.plot(DfB['radius'],DfB['U'],'.',label='Needle01')
    plt.plot(DfC['radius'],DfC['U'],'x',label='Needle02')
    plt.plot(DfD['radius'],DfD['U'],'<',label='Needle03')
    plt.plot(DfE['radius'],DfE['U'],'d',label='Needle04')
    plt.plot(DfF['radius'],DfF['U'],'s',label='Needle0Bigone')



    plt.title('Mean Velocity as a function of the mean radius')
    plt.xlabel('Radius')
    plt.ylabel('Velocity')
    plt.legend()

def verticalVelocityRadius(DfA,DfB,DfC,DfD,DfE,DfF):
    'vertical Velocity as a fct of radius'
    LT = plt.figure()
    plt.plot(DfA['radius'],DfA['Uy'],'*',label='NeedleOrange')
    plt.plot(DfB['radius'],DfB['Uy'],'.',label='Needle01')
    plt.plot(DfC['radius'],DfC['Uy'],'x',label='Needle02')
    plt.plot(DfD['radius'],DfD['Uy'],'<',label='Needle03')
    plt.plot(DfE['radius'],DfE['Uy'],'d',label='Needle04')
    plt.plot(DfF['radius'],DfF['Uy'],'s',label='Needle0Bigone')



    plt.title('Mean vertical Velocity as a function of the mean radius')
    plt.xlabel('Radius')
    plt.ylabel('Velocity')
    plt.legend()

def xVelocityRadius(DfA,DfB,DfC,DfD,DfE,DfF):
    'vertical Velocity as a fct of radius'
    LT = plt.figure()
    plt.plot(DfA['radius'],DfA['Ux'],'*',label='NeedleOrange')
    plt.plot(DfB['radius'],DfB['Ux'],'.',label='Needle01')
    plt.plot(DfC['radius'],DfC['Ux'],'x',label='Needle02')
    plt.plot(DfD['radius'],DfD['Ux'],'<',label='Needle03')
    plt.plot(DfE['radius'],DfE['Ux'],'d',label='Needle04')
    plt.plot(DfF['radius'],DfF['Ux'],'s',label='Needle0Bigone')
    plt.title('Mean vertical Velocity as a function of the mean radius')
    plt.xlabel('Radius')
    plt.ylabel('Velocity')
    plt.legend()

tracked=tp.link_df(df_filtered,search_range=80,memory=3)
tracked_filtred=tp.filter_stubs(df, threshold=15)

def YfctT(Df):
    J = plt.figure()
    Ax = J.add_subplot(111)
    for p in Df['particle'].unique():
        Ax.plot(Df[Df['particle']==p]['time'],Df[Df['particle']==p]['y'])
    
def YfctTNEW(Df):
    J = plt.figure()
    Ax = J.add_subplot(111)
    for p in Df['particle'].unique():
        Ax.plot(Df[Df['particle']==p]['radius'].mean(),((Df[Df['particle']==p]['y'].max()-Df[Df['particle']==p]['y'].min())/(Df[Df['particle']==p]['time'].max()-Df[Df['particle']==p]['time'].min())),'o')
    
def ArrayU(Df):
    A=[Df[Df['particle']==p]['radius'].mean() for p in Df['particle'].unique()]
    B=[((Df[Df['particle']==p]['y'].max()-Df[Df['particle']==p]['y'].min())/(Df[Df['particle']==p]['time'].max()-Df[Df['particle']==p]['time'].min())) for p in Df['particle'].unique()]
    return [A,B]


def ArrayU0o(Df):
    A=[]
    B=[]
    for p in Df['particle'].unique():
        if ((Df[Df['particle']==p]['y'].max()-Df[Df['particle']==p]['y'].min())/(Df[Df['particle']==p]['time'].max()-Df[Df['particle']==p]['time'].min()))<300:
            if ((Df[Df['particle']==p]['y'].max()-Df[Df['particle']==p]['y'].min())/(Df[Df['particle']==p]['time'].max()-Df[Df['particle']==p]['time'].min()))>200:
                A.append(Df[Df['particle']==p]['radius'].mean())
                B.append(((Df[Df['particle']==p]['y'].max()-Df[Df['particle']==p]['y'].min())/(Df[Df['particle']==p]['time'].max()-Df[Df['particle']==p]['time'].min())))
    return [A,B]


def REGVelocityRadius(DfA,DfB,DfC,DfD,DfE,DfF):
    'vertical Velocity as a fct of radius'
    LT = plt.figure()
    plt.plot(ArrayU(DfA)[0],ArrayU(DfA)[1],'*',label='NeedleOrange')
    plt.plot(ArrayU0o(DfB)[0],ArrayU0o(DfB)[1],'.',label='Needle01')
    plt.plot(ArrayU(DfC)[0],ArrayU(DfC)[1],'x',label='Needle02')
    plt.plot(ArrayU(DfD)[0],ArrayU(DfD)[1],'<',label='Needle03')
#    plt.plot(ArrayU(DfE)[0],ArrayU(DfE)[1],'d',label='Needle04')
    plt.plot(ArrayU(DfF)[0],ArrayU(DfF)[1],'s',label='Needle0Bigone')
    plt.title('REGRESSIOn Mean vertical Velocity as a function of the mean radius')
    plt.xlabel('Radius in mm')
    plt.ylabel('Velocity in mm per sec')
    plt.legend()

def histogrammeRadius(Df,ax,needle):
    A=[Df[Df['particle']==p]['radius'].mean() for p in Df['particle'].unique()]
    ax.hist(A,label=needle+'mean='+str(np.mean(A))+'stand.dev='+str(np.std(A)),normed=True)
    
Histo=plt.figure()
ax=Histo.add_subplot(111)




def negatif(x):
    if x<=0 :
        return 
    elif x>0.6:
        return
    else :
        return x

def traj(Df,needle='00'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    tp.plot_traj(Df,ax=ax,colorby='particle')
    plt.title('Trajectories of the '+str(len(Df['particle'].unique()))+' bubbles of needle '+str(needle))
    plt.xlabel('x en mm')
    plt.ylabel('y en mm')
    plt.figure()
    plt.hist([len(Df[Df['particle']==p]) for p in Df['particle'].unique()],histtype='step',label='Needle0'+str(needle),bins=36)
#ax.invert_yaxis()

def tri(Df,min,max):
    A=[]
    for p in Df['particle'].unique():
        if len(Df[Df['particle']==p])<min:
            Df=Df[Df['particle']!=p]
        if len(Df[Df['particle']==p])>max:
            Df=Df[Df['particle']!=p]

    return Df



def Dfmean(Df):
    trackedmean=pd.DataFrame(index=Df['particle'].unique(),columns=Df.columns)
    for p in Df['particle'].unique():
        for col in Df.columns:
            trackedmean[col][p]=Df[Df['particle']==p].mean(axis=0)[col]
    return trackedmean
    
    
list_of_bubbles = [tracked[tracked['particle']==p] for p in tracked['particle'].unique()]

#################PLOTS####################################

fig = plt.figure()
plt.title('y fct x')
ax = fig.add_subplot(111)
for bubble_df in list_of_bubbles:
    ax.plot(bubble_df['x'],bubble_df['y'])





"""
Df01=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudySurface\5_7_18\Neddle01.xlsx')
Df02=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudySurface\5_7_18\Neddle02.xlsx')
Df03=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudySurface\5_7_18\Neddle03.xlsx')
Df04=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudySurface\5_7_18\Neddle04.xlsx')
Df0V=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudySurface\5_7_18\Neddle0VSample01.xlsx')
Df0V02=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudySurface\5_7_18\Neddle0VSample02.xlsx')
Df0o=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudySurface\5_7_18\Neddle0orange.xlsx')

Df01.to_excel(r'C:\Users\Luc Deike\Nicolas\StudySurface\5_7_18\Neddle01.xlsx')
Df02.to_excel(r'C:\Users\Luc Deike\Nicolas\StudySurface\5_7_18\Neddle02.xlsx')
Df03.to_excel(r'C:\Users\Luc Deike\Nicolas\StudySurface\5_7_18\Neddle03.xlsx')
Df04.to_excel(r'C:\Users\Luc Deike\Nicolas\StudySurface\5_7_18\Neddle04.xlsx')
Df0V.to_excel(r'C:\Users\Luc Deike\Nicolas\StudySurface\5_7_18\Neddle0VSample01.xlsx')
Df0V02.to_excel(r'C:\Users\Luc Deike\Nicolas\StudySurface\5_7_18\Neddle0VSample02.xlsx')
Df0o.to_excel(r'C:\Users\Luc Deike\Nicolas\StudySurface\5_7_18\Neddle0orange.xlsx')


Df01filled=LifeTime(Df00, RadiusMin=0,RadiusMax=10,LifetimeMin=0,LifetimeMax=10, style='.')
Df02filled=LifeTime(Df00, RadiusMin=0,RadiusMax=10,LifetimeMin=0,LifetimeMax=10, style='.')
#Df03filled=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudySurface\5_7_18\Neddle03.xlsx')
Df04filled=LifeTime(Df00, RadiusMin=0,RadiusMax=10,LifetimeMin=0,LifetimeMax=10, style='.')

Df01=LifeTime(Df01, RadiusMin=0.5,RadiusMax=0.6,LifetimeMin=0.15,LifetimeMax=10)
Df02=LifeTime(Df02, RadiusMin=0.8,RadiusMax=0.9,LifetimeMin=0.15,LifetimeMax=10)
Df03=LifeTime(Df03, RadiusMin=1.8,RadiusMax=2.05,LifetimeMin=0.15,LifetimeMax=10)
Df04=LifeTime(Df04, RadiusMin=1.9,RadiusMax=2.5,LifetimeMin=0.15,LifetimeMax=10)
Df0V=LifeTime(Df0V, RadiusMin=1.850,RadiusMax=1.96,LifetimeMin=0.15,LifetimeMax=10)
Df0V02=LifeTime(Df0V02, RadiusMin=1.850,RadiusMax=1.96,LifetimeMin=0.15,LifetimeMax=10)
Df0o=LifeTime(Df0o, RadiusMin=0.86,RadiusMax=0.97,LifetimeMin=0.15,LifetimeMax=10)


Ne01.to_excel(r'C:\Users\Luc Deike\Nicolas\StudySurface\5_7_18\Neddle01.xlsx')
Ne0o.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\5_11_18\Needle0o.xlsx')
Ne02.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\5_11_18\Needle02.xlsx')
Ne03.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\5_11_18\Needle03.xlsx')
Ne0B.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\5_11_18\Needle0B.xlsx')

trackedNe0B.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\5_11_18\trackedNe0B.xlsx')
trackedNe03.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\5_11_18\trackedNe03.xlsx')
trackedNe02.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\5_11_18\trackedNe02.xlsx')
trackedNe0o.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\5_11_18\trackedNe0o.xlsx')
trackedNe01.to_excel(r'C:\Users\Luc Deike\Nicolas\StudySurface\5_7_18\trackedNe01.xlsx')
"""













