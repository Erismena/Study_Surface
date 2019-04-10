# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 15:29:16 2018
"""

#import sys
##print(sys.path)
#sys.path.append(r"C:\Users\Luc Deike\Documents\GitHub\2d-fluids-analysis")

#print(sys.path)
#N13sccm36s0=Record('N13sccm36s0',13,'\N13_sccm0036_sruf0_water4_P1580T2793_Tlab241Hum45Teau222_100fps10exptime.cine',r'D:\June_Studies\surf_0',100.0,10,r'C:\Users\Luc Deike\Nicolas\StudySurface\June_Studies\metric.bmp',0.0925,'6/10/2018','surface','PhantomV2012','nikon105mm','36mm',495,17,752,1242,415,0.61,0.568,2.82,0.13,'no_data','no_data','standard',0.36,1580,27.93,24.1,21.2,45,0,6,df_Brut='\N13sccm36s0_df_Brut_ALL.xlsx',df_tracked_filled='\N13sccm36s0_df_tracked_filled.xlsx')
#N12sccm40s0=Record('N12sccm40s0',12,'\N12_sccm0040_sruf0_water6_P1580T2793_Tlab241Hum43Teau212_100fps10exptime.cine',r'D:\June_Studies\surf_0',100.0,10,r'C:\Users\Luc Deike\Nicolas\StudySurface\June_Studies\metric.bmp',0.0925,'6/10/2018','surface','PhantomV2012','nikon105mm','36mm',495,17,752,1242,415,0.57,0.528,1.5,0.14,'no_data','no_data','standard',0.4,1580,27.93,24.1,21.2,43,0,6,df_Brut='\N12sccm40s0_df_Brut_ALL.xlsx',df_tracked_filled='\N12sccm40s0_df_tracked_filled.xlsx')
#N11sccm50s0=Record('N11sccm50s0',11,'\N11_sccm0050_sruf0_water6_P1580T2793_Tlab241Hum43Teau212_100fps10exptime.cine',r'D:\June_Studies\surf_0',100.0,10,r'C:\Users\Luc Deike\Nicolas\StudySurface\June_Studies\metric.bmp',0.0925,'6/10/2018','surface','PhantomV2012','nikon105mm','36mm',495,17,752,1242,415,0.64,0.58,1.82,0.15,'no_data','no_data','standard',0.5,1580,27.93,24.1,21.2,43,0,6,df_Brut='\N11sccm50s0_df_Brut_ALL.xlsx',df_tracked_filled='\N11sccm50s0_df_tracked_filled.xlsx')
#N10sccm70s0=Record('N10sccm70s0',10,'\N10_sccm0070_sruf0_water6_P1580T2793_Tlab241Hum43Teau212_100fps10exptime.cine',r'D:\June_Studies\surf_0',100.0,10,r'C:\Users\Luc Deike\Nicolas\StudySurface\June_Studies\metric.bmp',0.0925,'6/10/2018','surface','PhantomV2012','nikon105mm','36mm',495,17,752,1242,415,0.71,0.64,2.64,0.15,'no_data','no_data','standard',0.7,1580,27.93,24.1,21.2,43,0,6,df_Brut='\N10sccm70s0_df_Brut_ALL.xlsx',df_tracked_filled='\N10sccm70s0_df_tracked_filled.xlsx')
#N10sccm10s0=Record('N10sccm10s0',10,'\N10_sccm0010_sruf0_water6_P1580T2793_Tlab241Hum43Teau212_100fps10exptime.cine',r'D:\June_Studies\surf_0',100.0,10,r'C:\Users\Luc Deike\Nicolas\StudySurface\June_Studies\metric.bmp',0.0925,'6/10/2018','surface','PhantomV2012','nikon105mm','36mm',495,17,752,1242,415,0.71,0.623,1.44,0.15,'no_data','no_data','standard',0.1,1580,27.93,24.1,21.2,43,0,6,df_Brut='\N10sccm10s0_df_Brut_ALL.xlsx',df_tracked_filled='\N10sccm10s0_df_tracked_filled.xlsx')
#N9sccm200s0=Record('N9sccm200s0',9,'\N9_sccm0200_sruf0_water6_P1580T2793_Tlab241Hum43Teau212_100fps10exptime.cine',r'D:\June_Studies\surf_0',100.0,10,r'C:\Users\Luc Deike\Nicolas\StudySurface\June_Studies\metric.bmp',0.0925,'6/10/2018','surface','PhantomV2012','nikon105mm','36mm',495,17,752,1242,415,0.85,0.79,3.6,0.2,'no_data','no_data','standard',2,1580,27.93,24.1,21.2,43,0,6,df_Brut='\N9sccm200s0_df_Brut_ALL.xlsx',df_tracked_filled='\N9sccm200s0_df_tracked_filled.xlsx')
#N8sccm100s0=Record('N8sccm100s0',8,'\N8_sccm0100_sruf0_water6_P1580T2793_Tlab241Hum43Teau212_100fps10exptime.cine',r'D:\June_Studies\surf_0',100.0,10,r'C:\Users\Luc Deike\Nicolas\StudySurface\June_Studies\metric.bmp',0.0925,'6/10/2018','surface','PhantomV2012','nikon105mm','36mm',495,17,752,1242,415,0.93,0.87,3.16,'no_data','no_data','no_data','standard',1,1580,27.93,24.1,21.2,43,0,6,df_Brut='\N8sccm100s0_df_Brut_ALL.xlsx',df_tracked_filled='\N8sccm100s0_df_tracked_filled.xlsx')
#N7sccm300s0=Record('N7sccm300s0',7,'\N7_sccm0300_sruf0_water6_P1580T2793_Tlab241Hum43Teau212_100fps10exptime.cine',r'D:\June_Studies\surf_0',100.0,10,r'C:\Users\Luc Deike\Nicolas\StudySurface\June_Studies\metric.bmp',0.0925,'6/10/2018','surface','PhantomV2012','nikon105mm','36mm',495,17,752,1242,415,1.23,1.11,3.77,'no_data','no_data','no_data','standard',3,1580,27.93,24.1,21.2,43,0,6,df_Brut='\N7sccm300s0_df_Brut_ALL.xlsx',df_tracked_filled='\N7sccm300s0_df_tracked_filled.xlsx')
#N6sccm300s0=Record('N6sccm300s0',6,'\N6_sccm0300_sruf0_water6_P1580T2793_Tlab241Hum43Teau212_100fps10exptime.cine',r'D:\June_Studies\surf_0',100.0,10,r'C:\Users\Luc Deike\Nicolas\StudySurface\June_Studies\metric.bmp',0.0925,'6/10/2018','surface','PhantomV2012','nikon105mm','36mm',495,17,752,1242,415,1.48,1.38,4.34,'no_data','no_data','no_data','standard',3,1580,27.93,24.1,21.2,43,0,6,df_Brut='\N6sccm300s0_df_Brut_ALL.xlsx',df_tracked_filled='\N6sccm300s0_df_tracked_filled.xlsx')
#N5sccm300s0=Record('N5sccm300s0',5,'\N5_sccm0300_sruf0_water6_P1580T2793_Tlab241Hum43Teau212_100fps10exptime.cine',r'D:\June_Studies\surf_0',100.0,10,r'C:\Users\Luc Deike\Nicolas\StudySurface\June_Studies\metric.bmp',0.0925,'6/10/2018','surface','PhantomV2012','nikon105mm','36mm',495,17,752,1242,415,1.77,1.61,3.6,'no_data','no_data','no_data','standard',3,1580,27.93,24.1,21.2,43,0,6,df_Brut='\N5sccm300s0_df_Brut_ALL.xlsx',df_tracked_filled='\N5sccm300s0_df_tracked_filled.xlsx')
#N4sccm500s0=Record('N4sccm500s0',4,'\N4_sccm0500_sruf0_water6_P1580T2793_Tlab241Hum43Teau212_100fps10exptime.cine',r'D:\June_Studies\surf_0',100.0,10,r'C:\Users\Luc Deike\Nicolas\StudySurface\June_Studies\metric.bmp',0.0925,'6/10/2018','surface','PhantomV2012','nikon105mm','36mm',495,17,752,1242,415,2.06,1.9,4.21,'no_data','no_data','no_data','standard',5,1580,27.93,24.1,21.2,43,0,6,df_Brut='\N4sccm500s0_df_Brut_ALL.xlsx',df_tracked_filled='\N4sccm500s0_df_tracked_filled.xlsx')
#N3sccm500s0=Record('N3sccm500s0',3,'\N3_sccm0500_sruf0_water6_P1580T2793_Tlab241Hum43Teau212_100fps10exptime.cine',r'D:\June_Studies\surf_0',100.0,10,r'C:\Users\Luc Deike\Nicolas\StudySurface\June_Studies\metric.bmp',0.0925,'6/10/2018','surface','PhantomV2012','nikon105mm','36mm',495,17,752,1242,415,2.27,2,5.2,0.4,'no_data','no_data','standard',5,1580,27.93,24.1,21.2,43,0,6,df_Brut='\N3sccm500s0_df_Brut_ALL.xlsx',df_tracked_filled='\N3sccm500s0_df_tracked_filled.xlsx')
#N2sccm500s0=Record('N2sccm500s0',2,'\N2_sccm0500_sruf0_water6_P1580T2793_Tlab241Hum43Teau212_100fps10exptime.cine',r'D:\June_Studies\surf_0',100.0,10,r'C:\Users\Luc Deike\Nicolas\StudySurface\June_Studies\metric.bmp',0.0925,'6/10/2018','surface','PhantomV2012','nikon105mm','36mm',495,17,752,1242,415,3.12,2.4,5.4,'no_data','no_data','no_data','standard',5,1580,27.93,24.1,21.2,43,0,6,df_Brut='\N2sccm500s0_df_Brut_ALL.xlsx',df_tracked_filled='\N2sccm500s0_df_tracked_filled.xlsx')


import matplotlib.pyplot as plt
import numpy as np
import fluids2d.geometry
import fluids2d.backlight as backlight
import pims
import scipy.ndimage
import pandas as pd
from scipy.ndimage.morphology import binary_fill_holes
import skimage.measure
import skimage.filters
import trackpy as tp
import skimage
from skimage import data
from fluids2d.backlight import labeled_props
from fluids2d.backlight import filled2regionpropsdf
from scipy import stats 
import scipy.integrate as integrate
from skimage import data, img_as_float
from skimage import exposure
import random
import os
import matplotlib.gridspec as gridspec
#ct=pims.open(r'C:\Users\Luc Deike\Nicolas\StudySurface\June_Studies\surf_0\test.cine')
#
#c=pims.open(r'C:\Users\Luc Deike\Nicolas\StudySurface\June_Studies\surf_0\N4_sccm0090_sruf0_water6_P1580T2793_Tlab241Hum43Teau212_100fps10exptime.cine')
#c_G0170=pims.open(r'C:\Users\Luc Deike\Nicolas\StudySurface\6_13_2018\NeedleG0170sccm.cine')
#c_G0270=pims.open(r'C:\Users\Luc Deike\Nicolas\StudySurface\6_13_2018\NeedleG0270sccm.cine')
#c_G0970=pims.open(r'C:\Users\Luc Deike\Nicolas\StudySurface\6_13_2018\NeedleG0970sccm.cine')
#
#c_G0170=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudySurface\6_13_2018\NeedleG0170sccm')
#
#metric = pims.ImageSequence(r'C:\Users\Luc Deike\Nicolas\StudySurface\June_Studies\metric.bmp')
#metric2 = pims.ImageSequence(r'C:\Users\Luc Deike\Nicolas\StudySurface\June_Studies\metric-bis.bmp')
#metric3 = pims.ImageSequence(r'C:\Users\Luc Deike\Nicolas\StudySurface\June_Studies\metric3.bmp')
err=pims.ImageSequence(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Plot all\distance\final\Figure_243 - Copy (2).png')
Radius=[0.705,0.787,0.943,0.998,1.119,1.557,1.82,2.078]
thresh_bmp = 190
thresh = 610
dt = 0.01
dx= 0.1251042535446205170975813177648
im_shape=(800,1280)
#thresh_cluster=150
RadiusDiskMean=1
pts=[[502,401],[1670,403],[1682,1374],[419,1367]]
MinRadius=0.0000
MaxRadius=1000
MinRadius_holes=0.000
MaskBottom=752
MaskTop=17
MaskRight=1242
MaskLeft=415
#cerlce des erreurs dues au recouvement des images#Centre_erreur=[0,0]
search_range_trackage=0.005
g = fluids2d.geometry.GeometryScaler(dx=dx,im_shape=(800,1280),origin_pos=(0,0),origin_units='pix')
frames = np.arange(0,682,1)


class Record:
    #constructeur, appelee invariablement quand on souhaite cree un objet depuis la classe
    #tous les constructeurs s apellent __init__
    #le premier parametre doit etre self
    def __init__(self,
                 name,
                 needle,
                 name_record_file,
                 path_folder,
                 
                 fps,
                 Exp_time,
                 path_metric,
                 dx,
                 date,
                 study,
                 camera,
                 objectif,
                 obectif_extension,
                 thresh,
                 
                 MaskBottom,
                 MaskTop,
                 MaskRight,
                 MaskLeft,
                 Rmean,
                 MinRadius,
                 MaxRadius,
                 MinRadius_holes,
                 
                 search_range,
                 memory,
                 method,
                 Flow_rate_sccm,
                 Inlet_pressure_bar,
                 T_controler_oC,
                 T_lab_oC,
                 T_water_oC,
                 Humidity_pc,
                 Surfactant,
                 Age_of_the_water_day,
                 
                 color='no_data',
                 df_Brut='no_data',
                 df_filled='no_data',
                 df_tracked='no_data',
                 df_tracked_filled='no_data',
                 
                 df_Brut_BUBB='no_data',
                 df_filled_BUBBLE='no_data',
                 df_tracked_BUBBLE='no_data',
                 df_tracked_BUBBLE_filled='no_data',
                 
                 df_Brut_CLUSTER='no_data',
                 df_filled_CLUSTER='no_data',
                 df_tracked_CLUSTER='no_data',
                 df_tracked_CLUSTER_filled='no_data',
                 ):
        #RECORD
        self.name=name
        self.needle=needle
        self.name_record_file=name_record_file
        self.path_folder=path_folder
        self.fps=fps
        self.Exp_time=Exp_time
        self.path_metric=path_metric
        self.dx=dx
        self.date=date
        self.study=study
        self.camera=camera
        self.objectif=objectif
        self.obectif_extension=obectif_extension
        
        #IMG PROCESSING
        self.thresh=thresh
        self.MaskBottom=MaskBottom
        self.MaskTop=MaskTop
        self.MaskRight=MaskRight
        self.MaskLeft=MaskLeft
        self.Rmean=Rmean
        self.MinRadius=MinRadius
        self.MaxRadius=MaxRadius
        self.MinRadius_holes=MinRadius_holes
        self.search_range=search_range
        self.memory=memory
        self.method=method
        
        #EXP CONDITIONS
        self.Flow_rate_sccm=Flow_rate_sccm
        self.Inlet_pressure_bar=Inlet_pressure_bar
        self.T_controler_oC=T_controler_oC
        self.T_lab_oC=T_lab_oC
        self.T_water_oC=T_water_oC
        self.Humidity_pc=Humidity_pc
        self.Surfactant=Surfactant
        self.Age_of_the_water_day=Age_of_the_water_day

        if name_record_file=='no_data':
            self.c='no_data'
        else:
            self.c=pims.open(path_folder+name_record_file)
            self.nb_images=len(self.c)
            self.shape_im=(np.shape(self.c[0]))
            self.frames_all=np.arange(0,self.nb_images,1)
        self.dt=1/float(self.fps)
        
        #geometry+time+processing
        self.g=fluids2d.geometry.GeometryScaler(dx=self.dx,im_shape=[800,1280],origin_pos=(0,0),origin_units='pix')

#        for df in [df_Brut,
#                   df_filled,
#                   df_tracked,
#                   df_tracked_filled,
#                   df_Brut_BUBB,
#                   df_filled_BUBBLE,
#                   df_tracked_BUBBLE,
#                   df_tracked_BUBBLE_filled,
#                   df_Brut_CLUSTER,
#                   df_filled_CLUSTER,
#                   df_tracked_CLUSTER,
#                   df_tracked_CLUSTER_filled]:
#            print df
        if df_Brut=='no_data':
            self.df_Brut=pd.DataFrame()
        if df_Brut!='no_data':
            if type(df_Brut)==str:
                self.df_Brut=pd.read_excel(path_folder+df_Brut)

        if df_filled=='no_data':
            self.df_filled=pd.DataFrame()
        if df_filled!='no_data':
            if type(df_filled)==str:
                self.df_filled=pd.read_excel(path_folder+df_filled)

        if df_tracked=='no_data':
            self.df_tracked=pd.DataFrame()
        if df_tracked!='no_data':
            if type(df_tracked)==str:
                self.df_tracked=pd.read_excel(path_folder+df_tracked)

        if df_tracked_filled=='no_data':
            self.df_tracked_filled=pd.DataFrame()
        if df_tracked_filled!='no_data':
            if type(df_tracked_filled)==str:
                self.df_tracked_filled=pd.read_excel(path_folder+df_tracked_filled)

        if df_Brut_BUBB=='no_data':
            self.df_Brut_BUBB=pd.DataFrame()
        if df_Brut_BUBB!='no_data':
            if type(df_Brut_BUBB)==str:
                self.df_Brut_BUBB=pd.read_excel(path_folder+df_Brut_BUBB)

        if df_filled_BUBBLE=='no_data':
            self.df_filled_BUBBLE=pd.DataFrame()
        if df_filled_BUBBLE!='no_data':
            if type(df_filled_BUBBLE)==str:
                self.df_filled_BUBBLE=pd.read_excel(path_folder+df_filled_BUBBLE)

        if df_tracked_BUBBLE=='no_data':
            self.df_tracked_BUBBLE=pd.DataFrame()
        if df_tracked_BUBBLE!='no_data':
            if type(df_tracked_BUBBLE)==str:
                self.df_tracked_BUBBLE=pd.read_excel(path_folder+df_tracked_BUBBLE)

        if df_tracked_BUBBLE_filled=='no_data':
            self.df_tracked_BUBBLE_filled=pd.DataFrame()
        if df_tracked_BUBBLE_filled!='no_data':
            if type(df_tracked_BUBBLE_filled)==str:
                self.df_tracked_BUBBLE_filled=pd.read_excel(path_folder+df_tracked_BUBBLE_filled)

        if df_Brut_CLUSTER=='no_data':
            self.df_Brut_CLUSTER=pd.DataFrame()
        if df_Brut_CLUSTER!='no_data':
            if type(df_Brut_CLUSTER)==str:
                self.df_Brut_CLUSTER=pd.read_excel(path_folder+df_Brut_CLUSTER)

        if df_filled_CLUSTER=='no_data':
            self.df_filled_CLUSTER=pd.DataFrame()
        if df_filled_CLUSTER!='no_data':
            if type(df_filled_CLUSTER)==str:
                self.df_filled_CLUSTER=pd.read_excel(path_folder+df_filled_CLUSTER)

        if df_tracked_CLUSTER=='no_data':
            self.df_tracked_CLUSTER=pd.DataFrame()
        if df_tracked_CLUSTER!='no_data':
            if type(df_tracked_CLUSTER)==str:
                self.df_tracked_CLUSTER=pd.read_excel(path_folder+df_tracked_CLUSTER)

        if df_tracked_CLUSTER_filled=='no_data':
            self.df_tracked_CLUSTER_filled=pd.DataFrame()
        if df_tracked_CLUSTER_filled!='no_data':
            if type(df_tracked_CLUSTER_filled)==str:
                self.df_tracked_CLUSTER_filled=pd.read_excel(path_folder+df_tracked_CLUSTER_filled)

    def Process(self,frames,method,MinRadius=0,MaxRadius=10000,MinRadius_holes=0):
        if method=='random_walker_detection_cluster_holes' or method=='watershed_detection_cluster_holes':
            Trait_Temp=traitement(self.c,self.thresh,self.g,frames=frames,method=self.method,MinRadius=self.MinRadius,MaxRadius=self.MaxRadius,MinRadius_holes=self.MinRadius_holes,dt=self.dt,name=self.name,folder=self.path_folder)
            self.df_Brut_BUBB=Trait_Temp[0]
            self.df_Brut_CLUSTER=Trait_Temp[1]
        
            self.df_Brut_BUBB.to_excel(self.path_folder+'\\'+self.name+'_'+'df_Brut_BUBB_ALL'+'.xlsx')
            self.df_Brut_CLUSTER.to_excel(self.path_folder+'\\'+self.name+'_'+'df_Brut_CLUSTER_ALL'+'.xlsx')
        
        
        if method=='standard':
            self.df_Brut=traitement(self.c,self.thresh,self.g,frames=frames,method=self.method,MinRadius=self.MinRadius,MaxRadius=self.MaxRadius,MinRadius_holes=self.MinRadius_holes,dt=self.dt,name=self.name,folder=self.path_folder)
            self.df_Brut.to_excel(self.path_folder+'\\'+self.name+'_'+'df_Brut_ALL'+'.xlsx')
    
    
    def tri_radius_b(self,Rmin,Rmax,conc_min):
         self.df_filled_BUBBLE=self.df_Brut_BUBB[self.df_Brut_BUBB['radius']>Rmin]
         self.df_filled_BUBBLE=self.df_Brut_BUBB[self.df_Brut_BUBB['radius']<Rmax]
         self.df_filled_BUBBLE=self.df_Brut_BUBB[self.df_Brut_BUBB['area/c_area']>conc_min]
    def tri_radius_c(self,Rmin,Rmax):
         self.df_filled_CLUSTER=self.df_Brut_CLUSTER[self.df_Brut_CLUSTER['radius']>Rmin]
         self.df_filled_CLUSTER=self.df_Brut_CLUSTER[self.df_Brut_CLUSTER['radius']<Rmax]
         plt.plot(self.df_filled_CLUSTER['time'],self.df_filled_CLUSTER['radius'])


def error_bar(im):
    for j in np.arange(0,int(np.shape(im)[0])-4,1):
        for i in np.arange(1,int(np.shape(im)[1])-1,1):
            if np.array_equal(im[j][i-1],[255,255,255,255]) and np.array_equal(im[j][i+1],[255,255,255,255]):
                im[j][i]=im[0][0]
    return im
            
def record_toexcel(df,path_folder,name):
    df.to_excel(path_folder+'\\'+name+'.xlsx')
    
    
    
    
    
def mask(im):
    im[:,:MaskLeft] = 1012
    im[:,MaskRight:] = 1012
    im[:MaskTop,:] = 1012
    im[MaskBottom:,:] = 1012
    return im

    im[113:165,285:379] = 1012
def get_filled(im,thresh):
    im_filt = scipy.ndimage.filters.median_filter(im,size=3)
    im_filt = backlight.binarize(im_filt,thresh,large_true=False)
    filled = binary_fill_holes(im_filt)
    return filled

def get_binarize(im,thresh,size1=2,size2=2):
#    im_filt = scipy.ndimage.filters.median_filter(im,size=size1)
    im_filt = backlight.binarize(im,thresh,large_true=False)
#    im_filt = scipy.ndimage.filters.median_filter(im_filt,size=size2)
    return im_filt

def plot_tri(Record,Rmin=0,Rmax=10000000,xmin=0,xmax=100000,ymin=0,ymax=10000,cmin=0.9):
    fig,ax = plt.subplots(3,2,sharex=False,sharey=False,figsize=(16,11))
    df=Record.df_Brut
    plt.title(Record.name)
    
    ax[0,0].plot(df['frame'],df['radius'],'.',label='before')
    df=df[df['radius']>Record.MinRadius]
    df=df[df['radius']<Record.MaxRadius]
    ax[0,0].plot(df['frame'],df['radius'],'.',label='after')
    ax[0,0].set_title('radius Rmin='+str(Record.MinRadius)+' Rmax='+str(Record.MaxRadius))
    ax[0,0].legend()
    
    ax[1,0].plot(df['x_pix'],df['y_pix'],'.',label='before')
    df=df[df['x_pix']>Record.MaskLeft]
    df=df[df['x_pix']<Record.MaskRight]
    df=df[df['y_pix']>Record.MaskBottom]
    df=df[df['y_pix']<Record.MaskTop]
    ax[1,0].plot(df['x_pix'],df['y_pix'],'.',label='after')
    ax[1,0].set_title('pos Left '+str(Record.MaskLeft)+'R '+str(Record.MaskRight)+'T '+str(Record.MaskTop)+'B '+str(Record.MaskBottom))
    ax[1,0].legend()
    
    ax[2,0].plot(df['frame'],df['area/c_area'],'.',label='before')
    df=df[df['area/c_area']>cmin]
    ax[2,0].plot(df['frame'],df['area/c_area'],'.',label='after')
    ax[2,0].set_title(str(cmin))
    ax[2,0].legend()
    
    return df
    
def plot_tri2(Record,df,Rmin=0,Rmax=10000000,xmin=0,xmax=100000,ymin=0,ymax=10000,cmin=0.9):
    fig,ax = plt.subplots(3,2,sharex=False,sharey=False,figsize=(16,11))
    
    plt.title(Record.name)
    
    ax[0,0].plot(df['frame'],df['radius'],'.',label='before')
    df=df[df['radius']>Rmin]
    df=df[df['radius']<Rmax]
    ax[0,0].plot(df['frame'],df['radius'],'.',label='after')
    ax[0,0].set_title('radius Rmin='+str(Rmin)+' Rmax='+str(Rmax))
    ax[0,0].legend()
    
    ax[1,0].plot(df['x_pix'],df['y_pix'],'.',label='before')
    df=df[df['x_pix']>xmin]
    df=df[df['x_pix']<xmax]
    df=df[df['y_pix']>ymin]
    df=df[df['y_pix']<ymax]
    ax[1,0].plot(df['x_pix'],df['y_pix'],'.',label='after')
    ax[1,0].set_title('pos Left '+str(xmin)+'R '+str(xmax)+'T '+str(ymin)+'B '+str(ymax))
    ax[1,0].legend()
    
    ax[2,0].plot(df['frame'],df['area/c_area'],'.',label='before')
    df=df[df['area/c_area']>cmin]
    ax[2,0].plot(df['frame'],df['area/c_area'],'.',label='after')
    ax[2,0].set_title(str(cmin))
    ax[2,0].legend()
    return df
    
    
def tri_radius_c(df,Rmin,Rmax,name):
    df=df[df['radius']>Rmin]
    df=df[df['radius']<Rmax]
    plt.figure(figsize=(18,13))
    plt.plot(df['frame'],df['radius'],'.')
    plt.title(name)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    return df
    
def tri_po(df,xmin,xmax,ymin,ymax,name):
#    for df in df_t['variable'].tolist():
#       df.df_Mask=tri_pos_c(df.df_Brut,415,1242,17,725,df.name)
    plt.figure(figsize=(18,13))
    plt.plot(df['x_pix'],df['y_pix'],'.',label='before')
    df=df[df['x_pix']>xmin]
    df=df[df['x_pix']<xmax]
    df=df[df['y_pix']>ymin]
    df=df[df['y_pix']<ymax]
    plt.plot(df['x_pix'],df['y_pix'],'.',label='after')
    plt.title(name)
    plt.legend()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    return df
   
def tri_pos_convexite(df,cmin,name):
#    for df in df_t['variable'].tolist():
#        df.df_filled=tri_pos_convexite(df.df_filled,0.82,df.name)
    plt.figure(figsize=(18,13))
    plt.plot(df['frame'],df['area/c_area'],'.',label='before')
    df=df[df['area/c_area']>cmin]
    plt.plot(df['frame'],df['area/c_area'],'.',label='after')
    plt.title(name)
    plt.legend()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    return df

#def nb_neighboors(Record,Record_df):


    
    
    
    
def hist_radius_birth_death(Record,df):
    fig,ax = plt.subplots(2,sharex=True,sharey=True, figsize=(18,13))
    radius_birth=[]
    radius_death=[]
    for p in df['particle'].unique():
        radius_birth.append(df[df['particle']==p]['radius'].tolist()[0])
        radius_death.append(df[df['particle']==p]['radius'].tolist()[-1])
    ax[0].hist(radius_birth,histtype='stepfilled',bins=10)
    ax[1].hist(radius_death,histtype='stepfilled',bins=10)

#def plot_rad_time_hist_alone(R):
#    fig=plt.figure(figsize=(16,11))
#
#    b_per_s=np.round(bubble_per_second(R.df_tracked_filled),3)
#    fig.suptitle('Evolution of radius, Needle : '+str(R.needle)+'  FlowRate='+str(R.Flow_rate_sccm)+'sccm \n BubbleRate='+str(b_per_s),fontsize=15)
#    gs = gridspec.GridSpec(2, 2)
#    
#    ax = plt.subplot(gs[1, :])
#    for p in R.df_tracked_filled['particle'].unique():
#
#        ax.plot(R.df_tracked_filled[R.df_tracked_filled['particle']==p]['frame']*0.01,R.df_tracked_filled[R.df_tracked_filled['particle']==p]['radius'],'.')
#        ax.set_title('Radius(time) Flowrate = '+str(R.Flow_rate_sccm)+'sccm Bubb/sec='+str(b_per_s),fontsize=11)
#        ax.set_ylabel('Radius in mm',fontsize=11)
#        ax.set_xlabel('Time in s',fontsize=11)
#    
#    birth=[]
#    death=[]
#    for p in R.df_tracked_filled['particle'].unique():
#        birth.append(R.df_tracked_filled[R.df_tracked_filled['particle']==p]['radius'].tolist()[0])
#        death.append(R.df_tracked_filled[R.df_tracked_filled['particle']==p]['radius'].tolist()[-1])
#    
#    ax = plt.subplot(gs[0, 0])
#    ax.hist(birth,bins=int(len(birth)**(1.0/1.2))+5)
#    ax.set_title('Hist of the initial radius Flowrate = '+str(R.Flow_rate_sccm)+'sccm \n Bubb/sec='+str(b_per_s),fontsize=11)
#    ax.set_xlim(np.min(R.df_tracked_filled['radius'])*0.95,np.max(R.df_tracked_filled['radius'])*1.05
#
#    ax = plt.subplot(gs[0, 1])
#    ax.hist(death,bins=int(len(death)**(1.0/1.2))+5)
#    ax.set_title('Hist of the final radius Flowrate = '+str(R.Flow_rate_sccm)+'sccm \n Bubb/sec='+str(b_per_s),fontsize=11)
#    ax.set_xlim(np.min(R.df_tracked_filled['radius'])*0.95,np.max(R.df_tracked_filled['radius'])*1.05
#
#    if not os.path.exists(R.path_folder+'\\Evolution_of_radius_alone'):
#        os.mkdir(R.path_folder+'\\Evolution_of_radius_alone')
#    plt.savefig(R.path_folder+'\\Evolution_of_radius_alone'+'\\'+R.name+'_Needle'+str(R.needle)+'_BubbleRate'+str(b_per_s)+'Evolution_radius'+'.png',dpi=200)
#    

def plot_rad_time_hist_needle(liste,needle):
    liste_needle=[]    
    for R in liste:
        if R.needle==needle:
            liste_needle.append(R)
    
    fig,ax = plt.subplots(len(liste_needle),3,sharex=False,sharey=False, figsize=(16,11))

    i=0
    path_folder=liste[0].path_folder
    fontsize=10
    for R in liste_needle:
        b_per_s=np.round(bubble_per_second(R.df_tracked_filled),3)
        fig.suptitle('Evolution of radius for bubble rate ='+str(b_per_s)+', Needle : '+str(needle))
        for p in R.df_tracked_filled['particle'].unique():
            if len(liste_needle)==1:
                ax[0].plot(R.df_tracked_filled[R.df_tracked_filled['particle']==p]['frame']*0.01,R.df_tracked_filled[R.df_tracked_filled['particle']==p]['radius'],'.')
                ax[0].set_title('Radius(time) Flowrate = '+str(R.Flow_rate_sccm)+'sccm Bubb/sec='+str(b_per_s),fontsize=9)
                ax[0].set_ylabel('Radius in mm',fontsize=fontsize)
                ax[0].set_xlabel('Time in s',fontsize=fontsize)

            else:
                ax[i][0].plot(R.df_tracked_filled[R.df_tracked_filled['particle']==p]['frame']*0.01,R.df_tracked_filled[R.df_tracked_filled['particle']==p]['radius'],'.')
                ax[i][0].set_title('Radius(time) Flowrate = '+str(R.Flow_rate_sccm)+'sccm Bubb/sec='+str(b_per_s),fontsize=9)
                ax[i][0].set_ylabel('Radius in mm',fontsize=fontsize)
                ax[i][0].set_xlabel('Time in s',fontsize=fontsize)



        birth=[]
        death=[]
        df_rproduced=R.df_tracked_filled[R.df_tracked_filled['radius']<np.min(R.df_tracked_filled['radius'])*2.2]
        df_rcoalesced=R.df_tracked_filled
        for p in df_rproduced['particle'].unique():
            birth.append(df_rproduced[df_rproduced['particle']==p]['radius'].tolist()[0])
        for p in df_rcoalesced['particle'].unique():
            death.append(df_rcoalesced[df_rcoalesced['particle']==p]['radius'].tolist()[-1])
        
        if len(liste_needle)==1:
            ax[1].hist(birth,bins=int(len(birth)**(1.0/1.2))+5)
            ax[1].set_title('Hist of the initial radius Bubb/sec='+str(b_per_s),fontsize=fontsize)

            ax[1].set_xlim(np.min(df_rcoalesced['radius'])*0.95,np.max(df_rcoalesced['radius'])*1.05)
        else:
            ax[i][1].hist(birth,bins=int(len(birth)**(1.0/1.2))+5)
            ax[i][1].set_title('Hist of the initial radius Bubb/sec='+str(b_per_s),fontsize=fontsize)

            ax[i][1].set_xlim(np.min(df_rcoalesced['radius'])*0.95,np.max(df_rcoalesced['radius'])*1.05)
            
        if len(liste_needle)==1:
            ax[2].hist(death,bins=int(len(death)**(1.0/1.2))+5)
            ax[2].set_title('Hist of the final radius Bubb/sec='+str(b_per_s),fontsize=fontsize)

            ax[2].set_xlim(np.min(df_rcoalesced['radius'])*0.95,np.max(df_rcoalesced['radius'])*1.05)
        else:
            ax[i][2].hist(death,bins=int(len(death)**(1.0/1.2))+5)
            ax[i][2].set_title('Hist of the final radius Bubb/sec='+str(b_per_s),fontsize=fontsize)            

            ax[i][2].set_xlim(np.min(df_rcoalesced['radius'])*0.95,np.max(df_rcoalesced['radius'])*1.05)
        i=i+1
    if not os.path.exists(path_folder+'\\Evolution_of_radius_Needle_individual'):
        os.mkdir(path_folder+'\\Evolution_of_radius_Needle_individual')
    plt.savefig(path_folder+'\\Evolution_of_radius_Needle_individual'+'\\Needle'+str(needle)+'_buubSec='+str(b_per_s)+'Evolution_radius'+'.png',dpi=200)


def rad_rmergedmean_rmerged_max(liste):
    needles=[liste[i].needle for i in np.arange(0,len(liste),1)]
    needleset=list(set(needles))

    columns = int(np.sqrt(len(needleset)))+1
    rows = int(np.sqrt(len(needleset)))
    fig, ax = plt.subplots(rows, columns,squeeze=False,sharey=False,sharex=False,figsize=(13,13))
    folder=liste[0].path_folder
    fig.suptitle('Radius of Coalesced bubbles')
    fontsize=8
    for i in np.arange(0,len(needleset),1):
        print i
        flow_rates=[]
        nbbubbsec=[]
        Rproduced=[]
        RproducedStd=[]
        Rmeancoalesced=[]
        Rmediancoalesced=[]
        Rstdcoalesced=[]
        Rmaxcoalesced=[]
        for R in liste:
            if R.needle==needleset[i]:
                flow_rates.append(R.Flow_rate_sccm)
                nbbubbsec.append(bubble_per_second(R.df_tracked_filled))
                Rproduced.append(R.Rmean)
                RproducedStd.append(np.std(R.df_tracked_filled[R.df_tracked_filled['radius']<R.Rmean*1.16]['radius']))
                Rmeancoalesced.append(np.mean(R.df_tracked_filled[R.df_tracked_filled['radius']>R.Rmean*1.16]['radius']))
                Rmediancoalesced.append(np.median(R.df_tracked_filled[R.df_tracked_filled['radius']>R.Rmean*1.16]['radius']))
                Rstdcoalesced.append(np.std(R.df_tracked_filled[R.df_tracked_filled['radius']>R.Rmean*1.16]['radius']))
                Rmaxcoalesced.append(np.max(R.df_tracked_filled[R.df_tracked_filled['radius']>R.Rmean*1.16]['radius']))
                
        ax0=ax.flatten()[i]
        ax0.errorbar(nbbubbsec,Rproduced,yerr=RproducedStd,marker='d',label='Radius produced',color='blue')       
        ax0.errorbar(nbbubbsec,Rmeancoalesced,yerr=Rstdcoalesced,marker='.',fmt="none",color='orange')
        ax0.plot(nbbubbsec,Rmeancoalesced,'o',label='R Mean of bubble coalesced',color='orange')        
        ax0.plot(nbbubbsec,Rmediancoalesced,'s',label='R Median of bubble coalesced',color='green')
        ax0.plot(nbbubbsec,Rmaxcoalesced,'v',label='R Max of bubble coalesced',color='red')
    
        ax0.set_title('Needle='+str(needleset[i]),fontsize=fontsize)
        ax0.set_xlabel('Number of bubble per second',fontsize=fontsize)
        ax0.set_ylabel('Radius in mm',fontsize=fontsize)
        ax0.legend(fontsize=5)

def rad_rmergedmean_rmerged_maxindividual(liste):
    needles=[liste[i].needle for i in np.arange(0,len(liste),1)]
    needleset=list(set(needles))


    folder=liste[0].path_folder
    fontsize=19
    for i in np.arange(0,len(needleset),1):
        plt.figure(figsize=(18,14))
        print i
        flow_rates=[]
        nbbubbsec=[]
        Rproduced=[]
        RproducedStd=[]
        Rmeancoalesced=[]
        Rmediancoalesced=[]
        Rstdcoalesced=[]
        Rmaxcoalesced=[]
        for R in liste:
            if R.needle==needleset[i]:
                flow_rates.append(R.Flow_rate_sccm)
                nbbubbsec.append(bubble_per_second(R.df_tracked_filled))
                Rproduced.append(R.Rmean)
                RproducedStd.append(np.std(R.df_tracked_filled[R.df_tracked_filled['radius']<R.Rmean*1.16]['radius']))
                Rmeancoalesced.append(np.mean(R.df_tracked_filled[R.df_tracked_filled['radius']>R.Rmean*1.16]['radius']))
                Rmediancoalesced.append(np.median(R.df_tracked_filled[R.df_tracked_filled['radius']>R.Rmean*1.16]['radius']))
                Rstdcoalesced.append(np.std(R.df_tracked_filled[R.df_tracked_filled['radius']>R.Rmean*1.16]['radius']))
                Rmaxcoalesced.append(np.max(R.df_tracked_filled[R.df_tracked_filled['radius']>R.Rmean*1.16]['radius']))
                
        plt.errorbar(nbbubbsec,Rproduced,yerr=RproducedStd,marker='d',label='Radius produced',color='blue')       
        plt.errorbar(nbbubbsec,Rmeancoalesced,yerr=Rstdcoalesced,marker='.',fmt="none",color='orange')
        plt.plot(nbbubbsec,Rmeancoalesced,'o',label='R Mean of bubble coalesced',color='orange')        
        plt.plot(nbbubbsec,Rmediancoalesced,'s',label='R Median of bubble coalesced',color='green')
        plt.plot(nbbubbsec,Rmaxcoalesced,'v',label='R Max of bubble coalesced',color='red')
    
        plt.title('Radius of Coalesced bubbles'+'Needle='+str(needleset[i]),fontsize=fontsize)
        plt.xlabel('Number of bubble per second',fontsize=fontsize)
        plt.ylabel('Radius in mm',fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        
        if not os.path.exists(folder+'\\RadiusCoalescence'):
            os.mkdir(folder+'\\RadiusCoalescence')
        plt.savefig(folder+'\\RadiusCoalescence'+'\\'+'Needle'+str(i)+'__RadiusCoalescence'+'.png',dpi=200)

def rad_rmergedmean_rmerged_maxindividualnormalized(liste):
    needles=[liste[i].needle for i in np.arange(0,len(liste),1)]
    needleset=list(set(needles))
    folder=liste[0].path_folder
    fontsize=11
    flow_rates=[]
    nbbubbsec=[]
    Rproduced=[]
    RproducedStd=[]
    Rmeancoalesced=[]
    Rmediancoalesced=[]
    Rstdcoalesced=[]
    Rmaxcoalesced=[]
    for i in np.arange(0,len(needleset),1):
        plt.figure(figsize=(18,14))
        print i
        for R in liste:
            if R.needle==needleset[i]:
                flow_rates.append(R.Flow_rate_sccm)
                nbbubbsec.append(bubble_per_second(R.df_tracked_filled))
                Rproduced.append(R.Rmean/R.Rmean)
                RproducedStd.append(np.std(R.df_tracked_filled[R.df_tracked_filled['radius']<R.Rmean*1.16]['radius'])/R.Rmean)
                Rmeancoalesced.append(np.mean(R.df_tracked_filled[R.df_tracked_filled['radius']>R.Rmean*1.16]['radius'])/R.Rmean)
                Rmediancoalesced.append(np.median(R.df_tracked_filled[R.df_tracked_filled['radius']>R.Rmean*1.16]['radius'])/R.Rmean)
                Rstdcoalesced.append(np.std(R.df_tracked_filled[R.df_tracked_filled['radius']>R.Rmean*1.16]['radius'])/R.Rmean)
                Rmaxcoalesced.append(np.max(R.df_tracked_filled[R.df_tracked_filled['radius']>R.Rmean*1.16]['radius'])/R.Rmean)
                
    plt.errorbar(nbbubbsec,Rproduced,yerr=RproducedStd,marker='d',label='Normalized Radius produced',color='blue')       
    plt.errorbar(nbbubbsec,Rmeancoalesced,yerr=Rstdcoalesced,marker='.',fmt="none",color='orange')
    plt.plot(nbbubbsec,Rmeancoalesced,'o',label='Normalized R Mean of bubble coalesced',color='orange')        
    plt.plot(nbbubbsec,Rmediancoalesced,'s',label='Normalized R Median of bubble coalesced',color='green')
    plt.plot(nbbubbsec,Rmaxcoalesced,'v',label='Normalized R Max of bubble coalesced',color='red')
    
    plt.title('Normalized Radius of Coalesced bubbles'+'Needle='+str(needleset[i]),fontsize=fontsize)
    plt.xlabel('Number of bubble per second',fontsize=fontsize)
    plt.ylabel('Normalized Radius R/Rproduced',fontsize=fontsize)
    plt.legend(fontsize=fontsize)
        
    if not os.path.exists(folder+'\\RadiusCoalescence'):
        os.mkdir(folder+'\\RadiusCoalescence')
    plt.savefig(folder+'\\RadiusCoalescence'+'\\'+'normalizedALL_Needle'+'__RadiusCoalescence'+'.png',dpi=200)

def rad_rmergedmean_rmerged_maxNONnormalized(liste):
    needles=[liste[i].needle for i in np.arange(0,len(liste),1)]
    needleset=list(set(needles))
    folder=liste[0].path_folder
    fontsize=11
    flow_rates=[]
    nbbubbsec=[]
    Rproduced=[]
    RproducedStd=[]
    Rmeancoalesced=[]
    Rmediancoalesced=[]
    Rstdcoalesced=[]
    Rmaxcoalesced=[]
    for i in np.arange(0,len(needleset),1):
        plt.figure(figsize=(18,14))
        print i
        for R in liste:
            if R.needle==needleset[i]:
                flow_rates.append(R.Flow_rate_sccm)
                nbbubbsec.append(bubble_per_second(R.df_tracked_filled))
                Rproduced.append(R.Rmean)
                RproducedStd.append(np.std(R.df_tracked_filled[R.df_tracked_filled['radius']<R.Rmean*1.16]['radius']))
                Rmeancoalesced.append(np.mean(R.df_tracked_filled[R.df_tracked_filled['radius']>R.Rmean*1.16]['radius']))
                Rmediancoalesced.append(np.median(R.df_tracked_filled[R.df_tracked_filled['radius']>R.Rmean*1.16]['radius']))
                Rstdcoalesced.append(np.std(R.df_tracked_filled[R.df_tracked_filled['radius']>R.Rmean*1.16]['radius']))
                Rmaxcoalesced.append(np.max(R.df_tracked_filled[R.df_tracked_filled['radius']>R.Rmean*1.16]['radius']))
                
    plt.plot(nbbubbsec,Rproduced,'d',label='Radius produced',color='blue')       
    plt.errorbar(nbbubbsec,Rmeancoalesced,yerr=Rstdcoalesced,marker='.',fmt="none",color='orange')
    plt.plot(nbbubbsec,Rmeancoalesced,'o',label='R Mean of bubble coalesced',color='orange')        
    plt.plot(nbbubbsec,Rmediancoalesced,'s',label='R Median of bubble coalesced',color='green')
    plt.plot(nbbubbsec,Rmaxcoalesced,'v',label='R Max of bubble coalesced',color='red')
    
    plt.title('Radius of Coalesced bubbles'+'Needle='+str(needleset[i]),fontsize=fontsize)
    plt.xlabel('Number of bubble per second',fontsize=fontsize)
    plt.ylabel('Radius R/Rproduced',fontsize=fontsize)
    plt.legend(fontsize=fontsize)
        
    if not os.path.exists(folder+'\\RadiusCoalescence'):
        os.mkdir(folder+'\\RadiusCoalescence')
    plt.savefig(folder+'\\RadiusCoalescence'+'\\'+'ALL_Needle'+'__RadiusCoalescence'+'.png',dpi=200)


   
        
plt.figure(figsize=(18,13))    
#Lifetime_radius([listSURF[i] for i in np.arange(0,7,1)],'SURF 2 July : Tlab=24.1 C  Twater=21.2 C  Humidity=45%   Age of the water=08')
#Lifetime_radius([listSURF[i] for i in np.arange(7,13,1)],'SURF 3 July : Tlab=24.1 C  Twater=21.2 C  Humidity=45%   Age of the water=08')
#Lifetime_radius([listSURF[i] for i in np.arange(13,19,1)],'SURF 4 July : Tlab=24.1 C  Twater=21.2 C  Humidity=45%   Age of the water=08')
#
#
Lifetime_radius(July,'July : Tlab=24.1 C  Twater=21.2 C  Humidity=43%   Age of the water=06',dt=0.02)
Lifetime_radius(May7,'May 7th',dt=0.01)
Lifetime_radius(May24,'May 24th',dt=0.01)
plt.plot(np.arange(0,4,0.001),[lifetime_paper(R) for R in np.arange(0,4,0.001)],'-',label='Theorie')
plt.plot(radiusSTRUTHWOLF,lifetimeSTRUTHWOLF,'o',label='STRUTHWOLF1984 Tair=btw 18C and 20Cwater=distilled water(resistivity>18megohms ; max organic content=0.1p.p.m. ; no particle >0.22um')
plt.plot(radiusZHENG,lifetimeZHENG,'o',label='ZHENG1983 Tlab=btw 22.8C and 24.5C Twater=btw 20.9C and 21.6C  Humidity=43%   water=TapWater')
plt.title('Mean Lifetime as a function of the radius')
plt.legend()
plt.xlabel('Radius in mm')
plt.ylabel('Lifetime in s')
#
radiusSTRUTHWOLF=[0.034,0.036,0.04,0.045,0.05,0.055,0.06,0.07,0.08,0.1,0.12,0.15,0.19]
lifetimeSTRUTHWOLF=[0.5,0.4,0.28,0.22,0.17,0.14,0.12,0.1,0.07,0.06,0.048,0.04,0.02]

radiusZHENG=[0.7,0.75,0.95,1.15,3.35,3.7]
lifetimeZHENG=[1.26,1.43,1.67,1.87,3.87,3.36]

def Lifetime_radius(liste,name,dt=0):
    
    radius_all=[]
    radius_mean=[]
    
    lifetime_all=[]
    lifetime_mean=[]

    for R in liste:
        radius=[]
        lifetime=[]
        for p in R.df_tracked_filled['particle'].unique():
            radius.append(np.mean(R.df_tracked_filled[R.df_tracked_filled['particle']==p]['radius']))
            radius_all.append(np.mean(R.df_tracked_filled[R.df_tracked_filled['particle']==p]['radius']))
            
            lifetime.append(len(R.df_tracked_filled[R.df_tracked_filled['particle']==p])*dt)
            lifetime_all.append(len(R.df_tracked_filled[R.df_tracked_filled['particle']==p])*dt)
        
        radius_mean.append(np.mean(radius))
        lifetime_mean.append(np.mean(lifetime))
#    P=plt.semilogy(radius_all,lifetime_all,',')
#    color=P[0].get_color()
    plt.semilogy(radius_mean,lifetime_mean,'d',label=name)
#    plt.plot(radius_mean,lifetime_mean,'-',color=color)
            
def lifetime_paper(R):
    epsilon=0.00083
    eta=0.001002
    sigma=0.07197
    rho=997
    g=9.81
    lc=np.sqrt(sigma/(rho*g))
    Tb=(4.0/3.0)**(3.0/4.0)/(epsilon**(3.0/4.0))*eta*lc/sigma*(R/lc)**(1.0/2.0)
    return Tb
            
poi=0



def tracked(df,search_range,memory,dt,threshold=0):
    df=tp.link_df(df,search_range=search_range,memory=memory,adaptive_stop=0.001, adaptive_step=0.95)
    df=tp.filter_stubs(df, threshold=threshold)
    plt.figure()
    for p in df['particle'].unique():
        plt.plot(df[df['particle']==p]['frame'],df[df['particle']==p]['radius'],'.')
    plt.title('radius as a function of time for each particle ; num particle='+str(len(df['particle'].unique())))
    plt.xlabel('time in s')
    plt.ylabel('radius in mm')
    
    plt.figure()
    for p in df['particle'].unique():
        plt.plot(df[df['particle']==p]['x'],df[df['particle']==p]['y'])
    plt.title('trajectories (y fct x) of each particle ; num particle='+str(len(df['particle'].unique())))
    plt.xlabel('x in mm')
    plt.ylabel('y in mm')
    
    fig=plt.figure()
    ax0 = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)
    for p in df['particle'].unique():
        ax0.plot(p,len(df[df['particle']==p])*dt,'.')
    ax0.set_title('lifetime/particle')
    ax0.set_xlabel('particle')
    ax0.set_ylabel('lifetime of part in s')
    
    weights = np.ones_like([len(df[df['particle']==p])*dt for p in df['particle'].unique()])/float(len([len(df[df['particle']==p])*dt for p in df['particle'].unique()]))
    ax1.hist([len(df[df['particle']==p]) for p in df['particle'].unique()],histtype='stepfilled',bins=10)
    ax1.set_title('histogram of lifetime of paticle')
    return df

def viz_tracked_ax(Record,df,dt,savefig=True,viz=True):
    fig,ax = plt.subplots(2,2,sharex=False,sharey=False, figsize=(18,13))
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    LifeTime_All=[len(df[df['particle']==p])*dt for p in df['particle'].unique()]
    LifeTime_Mean=np.round(np.mean(LifeTime_All),3)
    LifeTime_Std=np.round(np.std(LifeTime_All),3)
    Rmean=np.round(np.mean(df['radius']),3)
    Q= df['particle'].unique().tolist()    

    bins=int(len(Q)**(1.0/3.0))+3
    LifeTime_Normalized=[x/LifeTime_Mean for x in LifeTime_All]
    inter=(np.max(LifeTime_Normalized)-np.min(LifeTime_Normalized))/bins
    weights = np.ones_like(LifeTime_Normalized)/(float(len(LifeTime_Normalized))*inter)
    AA=np.histogram(LifeTime_Normalized, weights=weights,bins=bins)
    fontsize=9
    step=int(len(Q)/30.0)

    
    for p in df['particle'].unique():
        ax[0,0].plot(df[df['particle']==p]['frame'],df[df['particle']==p]['radius'],'.')
    ax[0,0].set_title('Radius(frame) for each particle ; num part='+str(len(df['particle'].unique()))+' Rmean='+str(Rmean),fontsize=fontsize)
    ax[0,0].set_xlabel('frame',fontsize=fontsize)
    ax[0,0].set_ylabel('Radius in mm',fontsize=fontsize)
    fig00=plt.figure(figsize=(18,13))
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    for p in df['particle'].unique():
        plt.plot(df[df['particle']==p]['frame'],df[df['particle']==p]['radius'],'.')
    plt.title('Radius(frame) for each particle ; num part='+str(len(df['particle'].unique()))+' Rmean='+str(Rmean),fontsize=fontsize)
    plt.xlabel('frame',fontsize=fontsize)
    plt.ylabel('Radius in mm',fontsize=fontsize)
    if savefig:
        if not os.path.exists(Record.path_folder+'\\radius_filled2'):
            os.mkdir(Record.path_folder+'\\radius_filled2')
        plt.savefig(Record.path_folder+'\\radius_filled2'+'\\'+Record.name+'__radius'+'.png',dpi=200)
    if viz==False:
        plt.close(fig00)


    for p in df['particle'].unique():
        ax[1,0].plot(df[df['particle']==p]['x'],df[df['particle']==p]['y'])
    ax[1,0].set_title('Trajectories of each particle ; num particle='+str(len(df['particle'].unique())),fontsize=fontsize)
    ax[1,0].set_xlabel('x in mm',fontsize=fontsize)
    ax[1,0].set_ylabel('y in mm',fontsize=fontsize)
    fig10=plt.figure(figsize=(18,13))
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    for p in df['particle'].unique():
        plt.plot(df[df['particle']==p]['x'],df[df['particle']==p]['y'])
    plt.title('Trajectories of each particle ; num particle='+str(len(df['particle'].unique())),fontsize=fontsize)
    plt.xlabel('x in mm',fontsize=fontsize)
    plt.ylabel('y in mm',fontsize=fontsize)
    if savefig:
        if not os.path.exists(Record.path_folder+'\\traj_filled2'):
            os.mkdir(Record.path_folder+'\\traj_filled2')
        plt.savefig(Record.path_folder+'\\traj_filled2'+'\\'+Record.name+'__traj'+'.png',dpi=200)
    if viz==False:
        plt.close(fig10)
        

    for i in np.arange(0,len(Q),1):
        ax[0,1].plot(i,len(df[df['particle']==Q[i]])*dt,'.')
    ax[0,1].set_title('Lifetime of each particle, lifetimeMean='+str(LifeTime_Mean)+' LifetimeSTD='+str(LifeTime_Std),fontsize=fontsize)
    ax[0,1].set_xlabel('Particle',fontsize=fontsize)
    ax[0,1].set_ylabel('Lifetime in s',fontsize=fontsize)
    QMoyarray=[]
    LifetimeMoyarray=[]
    for j in np.arange(step,len(Q)-step,1):
        QMoyarray.append(np.mean([Q[k] for k in np.arange(j-step,j+step,1)]))
    for j in np.arange(step,len(Q)-step,1):
        LifetimeMoyarray.append(np.mean([LifeTime_All[k] for k in np.arange(j-step,j+step,1)]))
    ax[0,1].plot(np.arange(step,len(Q)-step,1),LifetimeMoyarray,label='running average')
    ax[0,1].legend(fontsize=fontsize)
    fig01=plt.figure(figsize=(18,13))
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    for i in np.arange(0,len(Q),1):
        plt.plot(i,len(df[df['particle']==Q[i]])*dt,'.')
    plt.title('lifetime/particle, lifetimeMean='+str(LifeTime_Mean)+' LifetimeSTD='+str(LifeTime_Std),fontsize=fontsize)
    plt.xlabel('particle',fontsize=fontsize)
    plt.ylabel('lifetime of part in s',fontsize=fontsize)
    plt.plot(np.arange(step,len(Q)-step,1),LifetimeMoyarray,label='running average')
    plt.legend(fontsize=fontsize)
    if savefig:
        if not os.path.exists(Record.path_folder+'\\lifetime_label_filled2'):
            os.mkdir(Record.path_folder+'\\lifetime_label_filled2')
        plt.savefig(Record.path_folder+'\\lifetime_label_filled2'+'\\'+Record.name+'__lifetime_label'+'.png',dpi=200)
    if viz==False:
        plt.close(fig01)



    ax[1,1].set_title('PDF(normalized lifetime) TAP WATER',fontsize=fontsize)
    ax[1,1].semilogy([AA[1][k] for k in np.arange(0,len(AA[0]),1)],AA[0],'o',label='NumPart = '+str(len(LifeTime_Normalized)))
    ax[1,1].plot(np.arange(0.001,12,0.001),[PDF_0(x) for x in np.arange(0.001,12,0.001)],label='fit Lhuissier')
    ax[1,1].set_xlim(-0.2,7)
    ax[1,1].set_ylim(0.0001,1.3)
    ax[1,1].set_xlabel('Normalized lifetime',fontsize=fontsize)
    ax[1,1].set_ylabel('PDF(normalized lifetime)',fontsize=fontsize)
    ax[1,1].legend()
    fig11=plt.figure(figsize=(18,13))
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.title('PDF(normalized lifetime) TAP WATER')
    plt.semilogy([AA[1][k] for k in np.arange(0,len(AA[0]),1)],AA[0],'o',label='NumPart = '+str(len(LifeTime_Normalized)))
    plt.plot(np.arange(0.001,12,0.001),[PDF_0(x) for x in np.arange(0.001,12,0.001)],label='fit Lhuissier')
    plt.xlim(-0.2,7)
    plt.ylim(0.0001,1.3)
    plt.xlabel('normalized lifetime',fontsize=fontsize)
    plt.ylabel('PDF(normalized lifetime)',fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    if savefig:
        if not os.path.exists(Record.path_folder+'\\lifetime_pdf_filled2'):
            os.mkdir(Record.path_folder+'\\lifetime_pdf_filled2')
        plt.savefig(Record.path_folder+'\\lifetime_pdf_filled2'+'\\'+Record.name+'__lifetime_pdf'+'.png',dpi=200)
    if viz==False:
        plt.close(fig11)
    
    if savefig:
        if not os.path.exists(Record.path_folder+'\\viz_tracked_ax_filled2'):
            os.mkdir(Record.path_folder+'\\viz_tracked_ax_filled2')
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.savefig(Record.path_folder+'\\viz_tracked_ax_filled2'+'\\'+Record.name+'__viz_tracked_ax'+'.png',dpi=200)


def viz_trackage(record,df,df_c,c,frames):
    for i in frames:
        print('frame '+str(i)+ ' sur ' +str(len(frames)))
        plt.figure(figsize=(12,12))
        plt.imshow(mask(c[i]))
        for p in df[df['frame']==i]['particle'].unique():
            plt.text(df[df['frame']==i][df[df['frame']==i]['particle']==p]['x_pix'],df[df['frame']==i][df[df['frame']==i]['particle']==p]['y_pix'],str(int(p)),color='blue',size=18,weight='bold',verticalalignment='bottom')
            plt.plot(df[df['particle']==p][df[df['particle']==p]['frame']<=i]['x_pix'],df[df['particle']==p][df[df['particle']==p]['frame']<=i]['y_pix'],label='bubble '+str(int(p)),linewidth=3.7)
        for p in df_c[df_c['frame']==i]['particle'].unique():
            plt.text(df_c[df_c['frame']==i][df_c[df_c['frame']==i]['particle']==p]['x_pix'],df_c[df_c['frame']==i][df_c[df_c['frame']==i]['particle']==p]['y_pix'],str(int(p)),color='black',size=22,weight='heavy',verticalalignment='top')
            plt.plot(df_c[df_c['particle']==p][df_c[df_c['particle']==p]['frame']<=i]['x_pix'],df_c[df_c['particle']==p][df_c[df_c['particle']==p]['frame']<=i]['y_pix'],'--',label='cluster '+str(int(p)),linewidth=3.7)
        plt.title('original image with trajectories of the present buuble and clusters')
        plt.xlabel('x in mm')
        plt.ylabel('y in mm')
        plt.axis([MaskLeft, MaskRight, MaskBottom, MaskTop])
        plt.legend()
        plt.savefig(record.path_folder+str(i)+'.png')
        plt.close()
        
        
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
        
def nb_neighboors(df,Rmean):
#    len=len(df)
    neighboors=[]
    for p in df['particle'].unique():
        print(str(p)+'....len'+str(len(df['particle'].unique())))
        for f in df[df['particle']==p]['frame']:
            xp=float(df[df['particle']==p][df[df['particle']==p]['frame']==f]['x'])
            yp=float(df[df['particle']==p][df[df['particle']==p]['frame']==f]['y'])
            nb=0
            for q in df[df['frame']==f]['particle'].unique():
                if q!=p:
                    xq=float(df[df['particle']==q][df[df['particle']==q]['frame']==f]['x'])
                    yq=float(df[df['particle']==q][df[df['particle']==q]['frame']==f]['y'])
                    if Is_inside_cercle(x=xq,y=yq,centre_x=xp,centre_y=yp,radius=(2.0*Rmean)*1.2):
                        nb=nb+1
            neighboors.append(nb)
    return neighboors
#    MaskBottom		MaskTop		MaskRight		MaskLeft

def plot_evol_nb_neigh(df,savefig=False):
    plt.figure(figsize=(12,12))
    for p in df['particle'].unique():
        plt.plot([df[df['particle']==p]['frame'].tolist()[i]-df[df['particle']==p]['frame'].tolist()[0] for i in np.arange(0,len(df[df['particle']==p]),1)],[np.mean(df[df['particle']==p]['neighboors'])]*len(df[df['particle']==p]))
        plt.xlabel('frame')
        plt.ylabel('number of neighboors')
        plt.title('Evolution of the number of neighboor as a function of time for each particle')
    if savefig:
        if not os.path.exists(Record.path_folder+'\\neighboors'):
            os.mkdir(Record.path_folder+'\\neighboors')
        plt.savefig(Record.path_folder+'\\neighboors'+'\\'+Record.name+'__neighboors'+'.png',dpi=200)

def plot_evol_nb_neigh_point(R,df,savefig=False,dt=0.02):
    plt.figure(figsize=(12,12))
    for p in df['particle'].unique():
        plt.plot([df[df['particle']==p]['frame'].tolist()[-1]*dt-df[df['particle']==p]['frame'].tolist()[0]*dt],[np.mean(df[df['particle']==p]['neighboors'])],'+')
    plt.xlabel('Proper time of the particle (0=birth)')
    plt.ylabel('Mean number of neighboors')
    plt.title('Mean nmber of neighboor as a function of proper time for each particle')
    LifeTime_All=[len(df[df['particle']==p])*dt for p in df['particle'].unique()]
    LifeTime_Mean=np.round(np.mean(LifeTime_All),3)
    LifeTime_Std=np.round(np.std(LifeTime_All),3)
    plt.plot([LifeTime_Mean,LifeTime_Mean],[0,np.max(df['neighboors'])],'-',linewidth=3,label='Mean LifeTime ALL')
    for i in np.arange(0,6,0.5):
        lifetimeint=[]
        if len(lifetimeint) == 0: print("empty")
        for p in df['particle'].unique():
            if np.mean(df[df['particle']==p]['neighboors'])>=i:
                if np.mean(df[df['particle']==p]['neighboors'])<i+1:
                    lifetimeint.append(len(df[df['particle']==p])*dt)
        P=plt.plot([np.mean(lifetimeint),np.mean(lifetimeint)],[i,i+1],'-')
        color=P[0].get_color()

        plt.plot([np.mean(lifetimeint)],[i+0.25],'o',markersize=8,label='Mean LifeTime for mean number of neigh btw '+str(i)+' and '+str(i+0.5),color=color)

    plt.legend()
    plt.xlim(-0.2,10)
    plt.ylim(0,6)

  
    if savefig:
        if not os.path.exists(R.path_folder+'\\neighboors'):
            os.mkdir(R.path_folder+'\\neighboors')
        plt.savefig(R.path_folder+'\\neighboors'+'\\'+R.name+'__neighboors'+'.png',dpi=200)

def plot_evol_nb_neigh_pointNOR(listSURF,savefig=False,dt=0.02):
    plt.figure(figsize=(12,12))
    for R in listSURF:
        df=R.df_tracked_BUBBLE_filled2
        LifeTime_All=[len(df[df['particle']==p])*dt for p in df['particle'].unique()]
        LifeTime_Mean=np.round(np.mean(LifeTime_All),3)
        LifeTime_Std=np.round(np.std(LifeTime_All),3)
        LifeTime_Normalized=[x/LifeTime_Mean for x in LifeTime_All]
        LifeTime_Normalized_mean=np.mean(LifeTime_Normalized)
        for p in df['particle'].unique():
            plt.plot([df[df['particle']==p]['frame'].tolist()[-1]*dt/LifeTime_Mean-df[df['particle']==p]['frame'].tolist()[0]*dt/LifeTime_Mean],[np.mean(df[df['particle']==p]['neighboors'])],',')
        plt.xlabel('Normalized Proper time of the particle (0=birth)')
        plt.ylabel('Mean number of neighboors')
        plt.title('Mean nmber of neighboor as a function of proper time for each particle')
        
#            plt.plot([LifeTime_Normalized_mean,LifeTime_Normalized_mean],[0,np.max(df['neighboors'])],'-',linewidth=3,label='Mean LifeTime ALL')
    
    for i in np.arange(0,6,0.25):
        lifetimeint=[]
        for R in listSURF:
            df=R.df_tracked_BUBBLE_filled2
            LifeTime_All=[len(df[df['particle']==p])*dt for p in df['particle'].unique()]
            LifeTime_Mean=np.round(np.mean(LifeTime_All),3)
            for p in df['particle'].unique():
                if np.mean(df[df['particle']==p]['neighboors'])>=i:
                    if np.mean(df[df['particle']==p]['neighboors'])<i+1:
                        lifetimeint.append(len(df[df['particle']==p])*dt/LifeTime_Mean)
        P=plt.plot([np.mean(lifetimeint),np.mean(lifetimeint)],[i,i+0.25],'-')
        color=P[0].get_color()
        plt.plot([np.mean(lifetimeint)],[i+0.125],'o',markersize=8,label='Mean LifeTime for mean number of neigh btw '+str(i)+' and '+str(i+1)+'num part='+str(lifetimeint),color=color)
#        plt.legend(fontsize=5)
        plt.xlim(-0.2,10)
        plt.ylim(0,6)
    plt.plot([1,1],[0,6],'-',color='black')
    if savefig:
        if not os.path.exists(R.path_folder+'\\neighboors'):
            os.mkdir(R.path_folder+'\\neighboors')
        plt.savefig(R.path_folder+'\\neighboors'+'\\'+R.name+'__neighboors'+'.png',dpi=200)

def plot_evol_nb_neighDIFF_pointNOR(listSURF,savefig=False,dt=0.02):
    Pi=plt.figure(figsize=(12,12))
    for R in listSURF:
        df=R.df_tracked_BUBBLE_filled2
        LifeTime_All=[len(df[df['particle']==p])*dt for p in df['particle'].unique()]
        LifeTime_Mean=np.round(np.mean(LifeTime_All),3)
        LifeTime_Std=np.round(np.std(LifeTime_All),3)
        LifeTime_Normalized=[x/LifeTime_Mean for x in LifeTime_All]
        LifeTime_Normalized_mean=np.mean(LifeTime_Normalized)
        for p in df['particle'].unique():
            a=df[df['particle']==p]['frame'].tolist()[0]*dt/LifeTime_Mean
            b=df[df['particle']==p]['frame'].tolist()[-1]*dt/LifeTime_Mean
            plt.plot([(x-a)*100.0/(float(b-a)) for x in df[df['particle']==p]['frame']*dt/LifeTime_Mean],df[df['particle']==p]['neighboorsdiff'],'.')
            Y=line.get_ydata()

        plt.xlabel('Normalized Proper time of the particle (0=birth ; 100=death)')
        plt.ylabel('Number of incresing/decreasing number of neighboors')
        plt.title('Event of increasing/decreasing number of neighboors')
    return Pi
#            plt.plot([LifeTime_Normalized_mean,LifeTime_Normalized_mean],[0,np.max(df['neighboors'])],'-',linewidth=3,label='Mean LifeTime ALL')
#    
#    for i in np.arange(0,6,0.25):
#        lifetimeint=[]
#        for R in listSURF:
#            df=R.df_tracked_BUBBLE_filled2
#            LifeTime_All=[len(df[df['particle']==p])*dt for p in df['particle'].unique()]
#            LifeTime_Mean=np.round(np.mean(LifeTime_All),3)
#            for p in df['particle'].unique():
#                if np.mean(df[df['particle']==p]['neighboors'])>=i:
#                    if np.mean(df[df['particle']==p]['neighboors'])<i+1:
#                        lifetimeint.append(len(df[df['particle']==p])*dt/LifeTime_Mean)
#        P=plt.plot([np.mean(lifetimeint),np.mean(lifetimeint)],[i,i+0.25],'-')
#        color=P[0].get_color()
#        plt.plot([np.mean(lifetimeint)],[i+0.125],'o',markersize=8,label='Mean LifeTime for mean number of neigh btw '+str(i)+' and '+str(i+1)+'num part='+str(lifetimeint),color=color)
##        plt.legend(fontsize=5)
#        plt.xlim(-0.2,10)
#        plt.ylim(0,6)
#    plt.plot([1,1],[0,6],'-',color='black')
    if savefig:
        if not os.path.exists(R.path_folder+'\\neighboors'):
            os.mkdir(R.path_folder+'\\neighboors')
        plt.savefig(R.path_folder+'\\neighboors'+'\\'+R.name+'__neighboors'+'.png',dpi=200)
'''
ax = plt.gca()
Y=[]
X=[]
for i in np.arange(0,len(ax.lines),1):
    line = ax.lines[i]
    X0=line.get_xdata()
    Y0=line.get_ydata()
    Y0=Y0.tolist()
    X0=X0.tolist()
    Y0.pop(0)
    X0.pop(0)
    X=X+X0
    Y=Y+Y0
    
X=line.get_xdata()
Y=line.get_ydata()
Y=Y.tolist()
X=X.tolist()
Y.pop(0)
X.pop(0)

indice=[]
for i in np.arange(0,len(X),1):
    if Y[i]!=0:
        indice.append(i)

X1=[X[i] for i in indice]
Y1=[Y[i] for i in indice]


interX=(np.max(X1)-np.min(X1))/30
weightsX = np.ones_like(X1)/(float(len(X1))*interX)
interY=(np.max(Y1)-np.min(Y1))/len(np.arange(-5.5,5.5,1))
weights = np.ones_like(Y1)/(float(len(Y1)*len(X1))*interY)
plt.figure()
plt.hist2d(X1,Y1,bins=[500,np.arange(-5.5,5.5,1)])
plt.xlabel('Normaliwed lifetime : 0=birth , 100=death')
plt.ylabel('Histogram of the number of increase or decrease of the number of neighboors')
plt.title('Histogram 2d Variation of the number of bubbles, Lifetime normalized')
plt.colorbar()
def bubble_per_second(df):
    nb_bubb=float(len(df['particle'].unique()))
    time=float((np.max(df['frame'])-np.min(df['frame']))*0.01)
        
    return nb_bubb/time
'''
def death_outside_mask(R,df,xmin,xmax,ymin,ymax):
    xmin=xmin+R.RmeanPIX*3
    xmax=xmax-R.RmeanPIX*3
    ymin=ymin+R.RmeanPIX*3
    ymax=ymax-R.RmeanPIX*3
    
    particle_keeped=df['particle'].unique().tolist()
    for p in df['particle'].unique():
        df_int=df[df['particle']==p]
        if np.min(df_int['x_pix'])<xmin or np.max(df_int['x_pix'])>xmax or np.min(df_int['y_pix'])<ymin or np.max(df_int['y_pix'])>ymax:
            particle_keeped.remove(p)
    print particle_keeped
    print len(particle_keeped)
    
    return df[df.particle.isin(particle_keeped)]


    
def Is_inside_cercle(x=0,y=0,centre_x=0,centre_y=0,radius=0):
    A=np.sqrt((x-centre_x)**2+(y-centre_y)**2)
    if A>radius:
        return False
    else:
        return True
    

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
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    
#G0070.df_tracked_BUBBLE=tracked(G0070.df_filled_BUBBLE,25,1)
#G0070.df_tracked_CLUSTER=tracked(G0070.df_filled_BUBBLE,25,1)

def traitement(c,thresh,g,frames,method='standard',Radius=1,MinRadius=0,MaxRadius=20000000000,MinRadius_holes=0,RadiusDiskMean=0,dt=0.01,name='',folder=''):
    
    df_all = pd.DataFrame()
    df_all_bubble = pd.DataFrame()
    df_all_cluster = pd.DataFrame()
    
    for i,f in enumerate(frames):
        print('frame '+str(f)+'.........of.'+str(len(frames))+name)
        im=mask(c[f])
        print('. get_filled and labeled')
        
        if method=='standard':
            filled = get_filled(im,thresh)
            df = filled2regionpropsdf(filled,g=g,frame=f)
            df = df[df['radius']>MinRadius]
            df = df[df['radius']<MaxRadius]
            print('... found '+str(len(df))+' objects.')
            df_all = pd.concat([df_all,df])
            if f in np.arange(0,50000,1000):
                if not os.path.exists(folder+'\\'+'excelTemp'):
                    os.mkdir(folder+'\\'+'excelTemp')
                df_all.to_excel(folder+'\\'+'excelTemp'+'\\'+name+'_'+str(f)+'_df.xlsx')
                
        elif method=='watershed':
            ws = watershed_detection(im,thresh,g,RadiusDiskMean=2,viz=False)
            df = labeled_props(ws,g,frame=f)
            df = df[df['radius']>3]
            df = df[df['radius']<200]
            print('... found '+str(len(df))+' objects.')
            df_all = pd.concat([df_all,df])
            if f in np.arange(0,50000,1000):
                df_all_bubble.to_excel(folder+'\\'+name+'_'+str(f)+'_bubble.xlsx')
                df_all_cluster.to_excel(folder+'\\'+name+'_'+str(f)+'_cluster.xlsx')
            
        elif method=='random_walker':
            rw=random_walker_detection(im,thresh,g,mode='cg',RadiusDiskMean=2,tol=0.01,viz=True)
            df = labeled_props(rw,g,frame=f)
            df = df[df['radius']>3]
            df = df[df['radius']<200]
            print('... found '+str(len(df))+' objects.')
            df_all = pd.concat([df_all,df])
            
        elif method=='random_walker_detection_cluster_dist':
            rw_cluster=random_walker_detection_cluster_dist(im,thresh,g,frame=f,MinRadius=MinRadius,MaxRadius=MaxRadius,mode='cg',RadiusDiskMean=RadiusDiskMean,tol=0.01,viz=True)
            df_bubble=rw_cluster[1]
            print('... found '+str(len(df_bubble))+' bubbles.')
            df_cluster=rw_cluster[2]
            print('... found '+str(len(df_cluster))+' clusters.')
            df_all_bubble = pd.concat([df_all_bubble,df_bubble])
            df_all_cluster = pd.concat([df_all_cluster,df_cluster])
            
        elif method=='random_walker_detection_cluster_holes':
            rw_cluster=random_walker_detection_cluster_holes(im,thresh,g,frame=f,MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,mode='bf',RadiusDiskMean=0,tol=0.01,viz_process=False,viz=True,path_folder=folder,name=name)
            df_bubble=rw_cluster[1]
            print('... found '+str(len(df_bubble))+' bubbles.')
            df_cluster=rw_cluster[2]
            print('... found '+str(len(df_cluster))+' clusters.')
            df_all_bubble = pd.concat([df_all_bubble,df_bubble])
            df_all_cluster = pd.concat([df_all_cluster,df_cluster])
            if f in np.arange(0,50000,1000):
                df_all_bubble.to_excel(folder+'\\'+name+'_'+str(f)+'_bubble.xlsx')
                df_all_cluster.to_excel(folder+'\\'+name+'_'+str(f)+'_cluster.xlsx')

        elif method=='watershed_detection_cluster_holes':
            rw_cluster=watershed_detection_cluster_holes(im,thresh,g,frame=f,MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,mode='cg_mg',RadiusDiskMean=RadiusDiskMean,tol=0.01,viz_process=True,viz=False,path_folder=folder,name=name)
            df_bubble=rw_cluster[1]
            print('... found '+str(len(df_bubble))+' bubbles.')
            df_cluster=rw_cluster[2]
            print('... found '+str(len(df_cluster))+' clusters.')
            df_all_bubble = pd.concat([df_all_bubble,df_bubble])
            df_all_cluster = pd.concat([df_all_cluster,df_cluster])
            if f in np.arange(0,50000,1000):
                df_all_bubble.to_excel(folder+'\\'+name+'_'+str(f)+'_bubble.xlsx')
                df_all_cluster.to_excel(folder+'\\'+name+'_'+str(f)+'_cluster.xlsx')            

    if method=='watershed_detection_cluster_holes':
        df_all_bubble['time'] = df_all_bubble['frame']*dt
        df_all_cluster['time'] = df_all_cluster['frame']*dt
        return df_all_bubble,df_all_cluster           
        
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


#
#tracked=tp.link_df(df_all_cluster,search_range=80,memory=3,search_range_trackage)
#
#
#df_all['time'] = df_all['frame']*dt        
#df_filtered = df_all[df_all['radius']>1]
#df_filtered = df_filtered[df_filtered['radius']<200]


''' #########################################trqckqge pqs fini###################################



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
'''
#############################    P L O T     F O L L O W I N G       B  U B B L E S  ################################
#def plotradius(A,Color):
#    for Needle in np.arange(0,np.shape(A)[0],1):
#        for Sccm in np.arange(0,len(A[Needle]),1):
##            print(Color[Needle][Sccm])
#            vizR(A[Needle][Sccm],color=Color[Needle][Sccm])
#def plotradius2(ListDataF,ColorList):
#    for i in np.arange(0,len(ListDataF),1):
#        vizR(ListDataF[i],color=ColorList[i],name=ListDataFName[i])
#
#def plottraj(ListDataF,ColorList):
#    for i in np.arange(0,len(ListDataF),1):
#        traj2(ListDataF[i],color=ColorList[i],name=ListDataFName[i])
#
#def plotVelocity(ListDataF,ColorList):
#    for i in np.arange(0,len(ListDataF),1):
#        traj(ListDataF[i],color=ColorList[i],name=ListDataFName[i])
#
#def plotYfctTimepluslabelall(ListDataF,ColorList):
#    for i in np.arange(0,len(ListDataF),1):
#        plotYfctTimepluslabel(ListDataF[i],name=ListDataFName[i])
#
#def traj2(Df,color='#CC0000',name='a'):
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    tp.plot_traj(Df,ax=ax,colorby='particle')
#    plt.title('Trajectories of the '+str(len(Df['particle'].unique()))+' bubbles of record '+name)
#    plt.xlabel('x en mm')
#    plt.ylabel('y en mm')
##    plt.savefig(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Radius\TrajFillted\\'+name+'.png')
##    plt.figure()
##    plt.hist([len(Df[Df['particle']==p]) for p in Df['particle'].unique()],histtype='step',label=name,bins=36,color=color)
##    plt.title('hist of the lifetime of the'+str(len(Df['particle'].unique()))+' bubbles of record '+name)
##    plt.savefig(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Radius\LifetimeFillted\\'+name+'.png')
#
#
#def TriRadius(ListDataF,DfR):
#    Rmean=[]
#    Rstd=[]
#    for i in np.arange(0,len(ListDataF),1):
#        ListDataF[i]=Tri(ListDataF[i],Rmin=DfR.loc[i,'Rmin'],Rmax=DfR.loc[i,'Rmax'],viz=False)
#        Rmean.append(np.round(np.mean(ListDataF[i]['radius']),4))
#        Rstd.append(np.round(np.std(ListDataF[i]['radius']),4))
#    return [Rmean,Rstd]
#
#
#def names(thing):
#    return [name for name,ref in globals().iteritems() if ref is thing]
#
#def vizR(Df,color='#CC0000',name='a'):
#    plt.figure()
#    plt.plot(Df['time'],Df['radius'],'.',color=color)
#    plt.title(names(Df))
#    plt.xlabel('time in s')
#    plt.ylabel('radius in mm')
#    plt.savefig(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Radius\RadiusFilled\\'+name+'.png')   
#    
#def Tri(Df,Rmin,Rmax,viz=False):
#    Df=Df[Df['radius']>Rmin]
#    Df=Df[Df['radius']<Rmax]
#    print('Mean Radius = '+str(np.round(np.mean(Df['radius']),5)))
#    print('Mean Stan Dev = '+str(np.round(np.std(Df['radius']),5)))
#
#    if viz==True:
#        plt.figure()
#        plt.plot(Df['frame'],Df['radius'],'.')
#    return Df
#
#def ArrayU(Df):
#    A=[Df[Df['particle']==p]['radius'].mean() for p in Df['particle'].unique()]
#    B=[((Df[Df['particle']==p]['y'].max()-Df[Df['particle']==p]['y'].min())/(Df[Df['particle']==p]['time'].max()-Df[Df['particle']==p]['time'].min())) for p in Df['particle'].unique()]
#    return [A,B]
#
##def plotVelocity(A):
##    Velo = plt.figure()
##    for Needle in np.arange(0,np.shape(A)[0],1):
##        for Sccm in np.arange(0,len(A[Needle]),1):
##            plt.plot(ArrayU(A[Needle][Sccm])[0],ArrayU(A[Needle][Sccm])[1],'+')
#            
def plotVelocity2(ListDataF,ColorList):
    Velo = plt.figure()
#    for i in np.arange(0,len(ListDataF),1):
#        plt.plot(ArrayU(ListDataF[i])[0],ArrayU(ListDataF[i])[1],'+',color=ColorList[i])
    for i in np.arange(0,len(ListDataF),1):
        plt.errorbar(np.mean(ArrayU(ListDataF[i])[0]),np.mean(ArrayU(ListDataF[i])[1]),np.std(ArrayU(ListDataF[i])[1]),np.std(ArrayU(ListDataF[i])[0]),'o',color='#000000',markerfacecolor=ColorList[i],label=ListDataFName[i]+'Vmean='+str(round(np.mean(ArrayU(ListDataF[i])[1]),3))+'  Vstd='+str(round(np.std(ArrayU(ListDataF[i])[1]),3))+'Rmean='+str(round(np.mean(ListDataF[i]['radius']),3))+' std='+str(round(np.std(ListDataF[i]['radius']),3)))
    plt.title('meanVelocity as a function of meanRadius')
    plt.xlabel('meanRadius in mm')
    plt.ylabel('meanVelocity')
    plt.legend(fontsize=5)
    
def plotVelocity2(ListDataF,ColorList):
    Velo = plt.figure()
#    for i in np.arange(0,len(ListDataF),1):
#        plt.plot(ArrayU(ListDataF[i])[0],ArrayU(ListDataF[i])[1],'+',color=ColorList[i])
    for i in np.arange(0,len(ListDataF),1):
        plt.errorbar(np.mean(ArrayU(ListDataF[i])[0]),np.mean(ArrayU(ListDataF[i])[1]),np.std(ArrayU(ListDataF[i])[1]),np.std(ArrayU(ListDataF[i])[0]),'o',color='#000000',markerfacecolor=ColorList[i],label=ListDataFName[i]+'Vmean='+str(round(np.mean(ArrayU(ListDataF[i])[1]),3))+'  Vstd='+str(round(np.std(ArrayU(ListDataF[i])[1]),3))+'Rmean='+str(round(np.mean(ListDataF[i]['radius']),3))+' std='+str(round(np.std(ListDataF[i]['radius']),3)))
    plt.title('meanVelocity as a function of meanRadius')
    plt.xlabel('meanRadius in mm')
    plt.ylabel('meanVelocity')
    plt.legend(fontsize=5)    
    
    
    

#    
#
#
#def plotHistRadius(ListDataF,ColorList):
#    bins=30
#    HRa = plt.figure()
#    ax = HRa.add_subplot(111)
#    Ww=[[0,7],[7,15],[15,21],[21,28],[28,36],[36,44],[44,55],[55,65]]
#    Needle=['A','B','C','D','E','F','G','H']
#
#    for w in np.arange(0,len(Ww),1):
#        Df = pd.DataFrame()
#        for i in np.arange(Ww[w][0],Ww[w][1],1):
#            Df= pd.concat([Df,ListDataF[i]])
#        weights = np.ones_like([np.mean(Df[Df['particle']==p]['radius']) for p in Df['particle'].unique()])/float(len([np.mean(Df[Df['particle']==p]['radius']) for p in Df['particle'].unique()]))
#        ax.hist([np.mean(Df[Df['particle']==p]['radius']) for p in Df['particle'].unique()],histtype='step', bins=bins, weights=weights,label='Needle '+Needle[w]+'  Rmean='+str(round(np.mean(Df['radius']),3))+' std='+str(round(np.std(Df['radius']),3))+'  NumPart='+str(len(Df['particle'].unique())),color=ColorList[Ww[w][1]-1])
#    
#        xt = plt.xticks()[0]  
#        xmin, xmax = min(xt), max(xt)  
#        lnspc = np.linspace(xmin, xmax, len([np.mean(Df[Df['particle']==p]['radius']) for p in Df['particle'].unique()])*1000)
#    
#    # lets try the normal distribution first
#        m, s = stats.norm.fit([np.mean(Df[Df['particle']==p]['radius']) for p in Df['particle'].unique()]) # get mean and standard deviation  
#        pdf_g = stats.norm.pdf(lnspc, m, s) # now get theoretical values in our interval  
##        plt.plot(lnspc, pdf_g,color=color, label=name+' Mean='+str(round(m,4))+' Std='+str(round(s,4)))# plot it
#        ax.plot(lnspc, pdf_g,color=ColorList[Ww[w][1]-1])# plot it
#    
#    
#    
#    
#    plt.title('Histogramme Radius for each Needle')
#    plt.xlabel('Radius in mm')
#    plt.ylabel('num')
#    plt.legend(fontsize=8)
#
#def plotHistVelocity(ListDataF,ColorList):
#    bins=20
#    HVe = plt.figure()
#    plt.title('Histogramme Vel for each Needle&FlowRate')
#    plt.xlabel('velocity in mm per sec')
#    plt.ylabel('num')
#    for i in np.arange(0,len(ListDataF),1):
#        weights = np.ones_like(ArrayU(ListDataF[i])[1])/float(len(ArrayU(ListDataF[i])[1]))
#        plt.hist(ArrayU(ListDataF[i])[1],histtype='step', bins=bins, weights=weights,label=ListDataFName[i]+'Rmean='+str(round(np.mean(ListDataF[i]['radius']),3))+' std='+str(round(np.std(ListDataF[i]['radius']),3))+'  NumPart='+str(len(ListDataF[i]['particle'].unique())),color=ColorList[i])
#
#    plt.legend(fontsize=5)
#
#
#HV = plt.figure()
#ax = HV.add_subplot(111)
#def plotVelocity(ListDataF,i1,i2,ColorList):
##    HV = plt.figure()
##    ax = HV.add_subplot(111)
#    plt.figure()
#    plt.title('Hist velocity+Fitting')
#    for i in np.arange(0,len(ListDataF),1):
##    for i in np.arange(i1,i2,1):
#        FittinHistVelocity(ArrayU(ListDataF[i])[1],ax,color=ColorList[i],name=ListDataFName[i])
#
#
#def FittinHistVelocity(Array,bins=20,ax=ax,color='#000000',name='a'):
#    weights = np.ones_like(Array)/float(len(Array))
#    plt.hist(Array,histtype='step', normed=True, bins=bins, weights=weights,color=color)
#
#    # find minimum and maximum of xticks, so we know
#    # where we should compute theoretical distribution
#    xt = plt.xticks()[0]  
#    xmin, xmax = min(xt), max(xt)  
#    lnspc = np.linspace(100, 400, len(Array)*1000)
#    
#    # lets try the normal distribution first
#    m, s = stats.norm.fit(Array) # get mean and standard deviation  
#    pdf_g = stats.norm.pdf(lnspc, m, s) # now get theoretical values in our interval  
#    plt.plot(lnspc, pdf_g,color=color, label=name+' Mean='+str(round(m,4))+' Std='+str(round(s,4)))# plot it
#    ax.plot(lnspc, pdf_g,color=color, label=name+' Mean='+str(round(m,4))+' Std='+str(round(s,4)))# plot it
#    plt.legend(fontsize=5)    
#    return [lnspc,pdf_g]
#
#def plotdist(ListDataF,ColorList):
#    for i in np.arange(0,len(ListDataF),1):
#        plotdist(ListDataF[i],name=ListDataFName[i])
#def plotdist(Df,name):
#    PD=plt.figure()
#    plt.title(name)
#    for i in np.arange(0,50,1):
#        plt.plot(Df[Df['particle']==Df['particle'].unique()[i]]['time'],Df[Df['particle']==Df['particle'].unique()[i]]['y'],linestyle='solid')
#    plt.gca().invert_yaxis()
#    plt.savefig(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Radius\yfcttime\\'+name+'.png')
#
#def plotdist2(ListDataF,ColorList):
#    for i in np.arange(0,len(ListDataF),1):
#        plotdist12(ListDataF[i],name=ListDataFName[i])
#def plotdist12(Df,name):
#    PD=plt.figure()
#    plt.xlabel('time in s')
#    plt.ylabel('distace relative btw two consecutives bubbles in mm NORMALIZED by the Rdius')
#    plt.title('NORMALIZED distance relative consecatives bubbles as function of time'+str(name))
#    for Q in np.arange(1,len(Df['particle'].unique()),1):
#        Q=int(Q)
#        timeQ=list(set(np.asarray(Df[Df['particle']==int(Df['particle'].unique()[Q])]['time'])) & set(np.asarray(Df[Df['particle']==int(Df['particle'].unique()[Q-1])]['time'])))
#        timeQ.sort()
#        Dftemp=Df[Df['time'].isin(timeQ)]
#        B0=Dftemp[Dftemp['particle']==int(Df['particle'].unique()[Q])]['y']
#        C0=Dftemp[Dftemp['particle']==int(Df['particle'].unique()[Q-1])]['y']
#        B0=np.asarray(B0)
#        C0=np.asarray(C0)
#        B=[(B0[k]-C0[k])/np.mean(Dftemp[Dftemp['particle']==int(Df['particle'].unique()[Q])]['radius']) for k in np.arange(0,min(len(B0),len(C0)),1)]
#        if len(B)!=0:
#            if B[0]<0:
#                B=map(abs, B)
#
##        if len(B0)<len(C0):
##            A0=np.asarray(Df[Df['particle']==int(Df['particle'].unique()[int(Q)])]['time'])
##            A=[k-np.min(Df[Df['particle']==int(Df['particle'].unique()[Q])]['time']) for k in A0]
##        else:
##            A0=np.asarray(Df[Df['particle']==int(Df['particle'].unique()[int(Q-1)])]['time'])
##            A=[k-np.min(Df[Df['particle']==int(Df['particle'].unique()[Q-1])]['time']) for k in A0]
#        timeQ0=[round(timeQ[k]-np.min(timeQ),5) for k in np.arange(0,len(timeQ),1)]
#        plt.plot(timeQ0,B,linestyle='solid')
#    plt.savefig(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Radius\distrealtivefcttime3superposeesNORMALIZED\\'+name+'.png')
#
#def plothistdistbtwconsbubble(ListDataF,ColorList,bins=20):
#    M=[]
#    Std=[]
#    for i in np.arange(0,len(ListDataF),1):
#        print(i)
#        S=plothistdistbtwconsbubble2(ListDataF[i],name=ListDataFName[i],bins=bins)
#        M.append(S[0])
#        Std.append(S[1])
#    plt.figure()
#    plt.title('For each needle and flow rate, Mean and std value of the \n diff btw the initial distace btw two consecutive particle and the final distance btw two consecutive particle')
#    plt.xlabel('Diff needle and flow rates')
#    plt.ylabel('Mean and std value of the diff btw \n the initial distace btw two consecutive particle \n and the final distance btw two consecutive particle',size=11)
#    for i in np.arange(0,len(ListDataF),1):
#        plt.errorbar([ListDataFName[i]],[M[i]],[Std[i]],marker='o',color='#000000',markerfacecolor=ColorList[i],label=ListDataFName[i]+' Mean='+str(round(M[i],5))+' Std'+str(round(Std[i],5)))
# #       plt.text([ListDataFName[i]],[M[i]],str(M[i]),size=9)
#        plt.legend(fontsize=5)
#    plt.plot(ListDataFName,[0]*len(ListDataFName),'--')
#    return [M,Std]
#def plothistdistbtwconsbubble2(Df,name='p',bins=20):
#    PD=plt.figure()
#    plt.xlabel('Initial Dist btw two consecutives buubles - Final one')
#    plt.ylabel('num')
#    plt.title(name)
#    Ecart=[] 
#    for Q in np.arange(1,len(Df['particle'].unique()),1):
#        Q=int(Q)
#        timeQ=list(set(np.asarray(Df[Df['particle']==int(Df['particle'].unique()[Q])]['time'])) & set(np.asarray(Df[Df['particle']==int(Df['particle'].unique()[Q-1])]['time'])))
#        timeQ.sort()
#        Dftemp=Df[Df['time'].isin(timeQ)]
#        B0=Dftemp[Dftemp['particle']==int(Df['particle'].unique()[Q])]['y']
#        C0=Dftemp[Dftemp['particle']==int(Df['particle'].unique()[Q-1])]['y']
#        B0=np.asarray(B0)
#        C0=np.asarray(C0)
#        B=[B0[k]-C0[k] for k in np.arange(0,min(len(B0),len(C0)),1)]
#        if len(B)!=0:
#            if B[0]<0:
#                B=map(abs, B)
#            Ecart.append(B[-1]-B[0])
##        if len(B0)<len(C0):
##            A0=np.asarray(Df[Df['particle']==int(Df['particle'].unique()[int(Q)])]['time'])
##            A=[k-np.min(Df[Df['particle']==int(Df['particle'].unique()[Q])]['time']) for k in A0]
##        else:
##            A0=np.asarray(Df[Df['particle']==int(Df['particle'].unique()[int(Q-1)])]['time'])
##            A=[k-np.min(Df[Df['particle']==int(Df['particle'].unique()[Q-1])]['time']) for k in A0]
#        timeQ0=[round(timeQ[k]-np.min(timeQ),5) for k in np.arange(0,len(timeQ),1)]
#    weights = np.ones_like(Ecart)/float(len(Ecart))
#    plt.hist(Ecart,histtype='step', bins=bins,weights=weights)
#    xt = plt.xticks()[0]  
#    xmin, xmax = min(xt), max(xt)  
#    lnspc = np.linspace(xmin, xmax, len(Ecart)*1000)
#    
#    # lets try the normal distribution first
#    m, s = stats.norm.fit(Ecart) # get mean and standard deviation  
#    pdf_g = stats.norm.pdf(lnspc, m, s) # now get theoretical values in our interval  
##    plt.plot(lnspc, pdf_g, label=name+' Mean='+str(round(m,4))+' Std='+str(round(s,4)))# plot it
#    plt.legend(fontsize=5)
##    plt.savefig(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Radius\HistFittedoftheDiffbtwthedistancebtwconsecutivebubbleinitiallyandfinally\\'+name+'.png')
#    return[m,s]
#
#def plothistdistbtwconsbubbleNORMALIZED(ListDataF,ColorList,bins=20):
#    M=[]
#    Std=[]
#    for i in np.arange(0,len(ListDataF),1):
#        print(i)
#        S=plothistdistbtwconsbubble2NORMALIZED(ListDataF[i],name=ListDataFName[i],bins=bins)
#        M.append(S[0])
#        Std.append(S[1])
#    plt.figure()
#    plt.title('For each needle and flow rate, Mean and std value of the \n diff btw the initial distace btw two consecutive particle and the final distance btw two consecutive particle')
#    plt.xlabel('Diff needle and flow rates')
#    plt.ylabel('Mean and std value of the diff btw \n the initial distace btw two consecutive particle \n and the final distance btw two consecutive particle',size=11)
#    for i in np.arange(0,len(ListDataF),1):
#        plt.errorbar([ListDataFName[i]],[M[i]],[Std[i]],marker='o',color='#000000',markerfacecolor=ColorList[i],label=ListDataFName[i]+' Mean='+str(round(M[i],5))+' Std'+str(round(Std[i],5)))
# #       plt.text([ListDataFName[i]],[M[i]],str(M[i]),size=9)
#        plt.legend(fontsize=5)
#    plt.plot(ListDataFName,[0]*len(ListDataFName),'--')
#    return [M,Std]
#def plothistdistbtwconsbubble2NORMALIZED(Df,name='p',bins=20):
#    PD=plt.figure()
#    plt.xlabel('Initial Dist btw two consecutives buubles - Final one, NORMALIZED by the radius')
#    plt.ylabel('num')
#    plt.title(name)
#    Ecart=[] 
#    for Q in np.arange(1,len(Df['particle'].unique()),1):
#        Q=int(Q)
#        timeQ=list(set(np.asarray(Df[Df['particle']==int(Df['particle'].unique()[Q])]['time'])) & set(np.asarray(Df[Df['particle']==int(Df['particle'].unique()[Q-1])]['time'])))
#        timeQ.sort()
#        Dftemp=Df[Df['time'].isin(timeQ)]
#        B0=Dftemp[Dftemp['particle']==int(Df['particle'].unique()[Q])]['y']
#        C0=Dftemp[Dftemp['particle']==int(Df['particle'].unique()[Q-1])]['y']
#        B0=np.asarray(B0)
#        C0=np.asarray(C0)
#        B=[(B0[k]-C0[k])/np.mean(Dftemp[Dftemp['particle']==int(Df['particle'].unique()[Q])]['radius']) for k in np.arange(0,min(len(B0),len(C0)),1)]
#        if len(B)!=0:
#            if B[0]<0:
#                B=map(abs, B)
#            Ecart.append(B[-1]-B[0])
##        if len(B0)<len(C0):
##            A0=np.asarray(Df[Df['particle']==int(Df['particle'].unique()[int(Q)])]['time'])
##            A=[k-np.min(Df[Df['particle']==int(Df['particle'].unique()[Q])]['time']) for k in A0]
##        else:
##            A0=np.asarray(Df[Df['particle']==int(Df['particle'].unique()[int(Q-1)])]['time'])
##            A=[k-np.min(Df[Df['particle']==int(Df['particle'].unique()[Q-1])]['time']) for k in A0]
#        timeQ0=[round(timeQ[k]-np.min(timeQ),5) for k in np.arange(0,len(timeQ),1)]
#    weights = np.ones_like(Ecart)/float(len(Ecart))
#    plt.hist(Ecart,histtype='step', bins=bins,weights=weights)
#    xt = plt.xticks()[0]  
#    xmin, xmax = min(xt), max(xt)  
#    lnspc = np.linspace(xmin, xmax, len(Ecart)*1000)
#    
#    # lets try the normal distribution first
#    m, s = stats.norm.fit(Ecart) # get mean and standard deviation  
#    pdf_g = stats.norm.pdf(lnspc, m, s) # now get theoretical values in our interval  
#    plt.plot(lnspc, pdf_g, label=name+' Mean='+str(round(m,4))+' Std='+str(round(s,4)))# plot it
#    plt.legend(fontsize=5)
#    plt.savefig(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Radius\HistFittedoftheDiffbtwthedistancebtwconsecutivebubbleinitiallyandfinallyNORMALIZED\\'+name+'.png')
#    return[m,s]
#
#def plotDISTINTERPART_RADIUS_VMOY(ListDataF,ColorList):
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    ax.set_title('For each buuble /n [mean Dist with the nearest above bubble,mean radius,mean Uy')
#    ax.set_xlabel('mean Dist with the nearest above bubble')
#    ax.set_ylabel('mean radius')
#    ax.set_zlabel('mean Uy')
#    for i in np.arange(0,len(ListDataF),1):
#        print(i)
#        Df=ListDataF[i]
#        for Q in np.arange(1,len(Df['particle'].unique())-1,1):
#            timeQ=list(set(np.asarray(Df[Df['particle']==int(Df['particle'].unique()[Q])]['time'])) & set(np.asarray(Df[Df['particle']==int(Df['particle'].unique()[Q-1])]['time'])))
#            timeQ.sort()
#            Dftemp=Df[Df['time'].isin(timeQ)]
#            B0=Dftemp[Dftemp['particle']==int(Df['particle'].unique()[Q])]['y']
#            C0=Dftemp[Dftemp['particle']==int(Df['particle'].unique()[Q-1])]['y']
#            B0=np.asarray(B0)
#            C0=np.asarray(C0)
#            B=[(B0[k]-C0[k])/np.mean(Dftemp[Dftemp['particle']==int(Df['particle'].unique()[Q])]['radius']) for k in np.arange(0,min(len(B0),len(C0)),1)]
#            Ecartmean=np.mean(B)
#            Radiusmean=np.mean(Df[Df['particle']==int(Df['particle'].unique()[Q])]['radius'])
#            Uymean=np.mean(Df[Df['particle']==int(Df['particle'].unique()[Q])]['Uy'])
#             
#            ax.plot([Radiusmean],[Ecartmean],[Uymean],'+',color=ColorList[i])
#
#def plotDISTINTERPART_RADIUS_VMOY(ListDataF,ColorList):
#    fig = plt.figure()
##    ax = fig.gca(projection='3d')
#    ax = fig.add_subplot(111, projection='3d')
#    ax.set_title('For each buuble /n [mean Dist with the nearest above bubble,mean radius,mean Uy')
#    ax.set_ylabel('mean Dist with the nearest above bubble')
#    ax.set_xlabel('mean radius')
#    ax.set_zlabel('mean Uy')
#
#    for i in np.arange(0,len(ListDataF),1):
#
#        print(i)
#        Df=ListDataF[i]
#        for Q in np.arange(1,len(Df['particle'].unique())-1,1):
#            timeQ=list(set(np.asarray(Df[Df['particle']==int(Df['particle'].unique()[Q])]['time'])) & set(np.asarray(Df[Df['particle']==int(Df['particle'].unique()[Q-1])]['time'])))
#            timeQ.sort()
#            Dftemp=Df[Df['time'].isin(timeQ)]
#            B0=Dftemp[Dftemp['particle']==int(Df['particle'].unique()[Q])]['y']
#            C0=Dftemp[Dftemp['particle']==int(Df['particle'].unique()[Q-1])]['y']
#            B0=np.asarray(B0)
#            C0=np.asarray(C0)
#            B=[(B0[k]-C0[k])/np.mean(Dftemp[Dftemp['particle']==int(Df['particle'].unique()[Q])]['radius']) for k in np.arange(0,min(len(B0),len(C0)),1)]
#            Ecartmean=np.mean(B)
#            Radiusmean=np.mean(Df[Df['particle']==int(Df['particle'].unique()[Q])]['radius'])
#            Uymean=np.mean(Df[Df['particle']==int(Df['particle'].unique()[Q])]['Uy'])
#             
#        ax.plot([Radiusmean],[Ecartmean],[Uymean],'o',color='#000000',markerfacecolor=ColorList[i],label=ListDataFName[i])
#        ax.plot([Radiusmean], [Uymean], 'd', color=ColorList[i], zdir='y', zs=130)
#        ax.plot([Ecartmean], [Uymean], 's', color=ColorList[i], zdir='x', zs=2.5)
#        ax.plot([Radiusmean], [Ecartmean], 'p', color=ColorList[i], zdir='z', zs=160)
#        
#        ax.legend(fontsize=5)
#        
#def plotDISTINTERPART_RADIUS_VMOY(ListDataF,ColorList):
#    Needle=['A','B','C','D','E','F','G','H']
#    Ww=[[0,7],[7,15],[15,21],[21,28],[28,36],[36,44],[44,55],[55,65]]
#    for w in np.arange(0,len(Ww),1):
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
#        ax.set_title('for needle '+Needle[w]+' and flow rates \n mean Dist with the nearest above bubble and mean Uy')
#        ax.set_xlabel('mean Dist with the nearest above bubble normalized by radius')
#        ax.set_ylabel('mean Uy')
#        EcartmeanALLPERNEEDLE=[]
#        UymeanALLPERNEEDLE=[]
#        for i in np.arange(Ww[w][0],Ww[w][1],1):
#            EcartmeanALL=[]
#            UymeanALL=[]
#            print(i)
#            Df=ListDataF[i]
#            for Q in np.arange(1,len(Df['particle'].unique())-1,1):
#                timeQ=list(set(np.asarray(Df[Df['particle']==int(Df['particle'].unique()[Q])]['time'])) & set(np.asarray(Df[Df['particle']==int(Df['particle'].unique()[Q-1])]['time'])))
#                timeQ.sort()
#                Dftemp=Df[Df['time'].isin(timeQ)]
#                B0=Dftemp[Dftemp['particle']==int(Df['particle'].unique()[Q])]['y']
#                C0=Dftemp[Dftemp['particle']==int(Df['particle'].unique()[Q-1])]['y']
#                B0=np.asarray(B0)
#                C0=np.asarray(C0)
#                B=[(B0[k]-C0[k])/np.mean(Dftemp[Dftemp['particle']==int(Df['particle'].unique()[Q])]['radius']) for k in np.arange(0,min(len(B0),len(C0)),1)]
#                Ecartmean=np.mean(B)
#                EcartmeanALL.append(Ecartmean)
#                Uymean=np.mean(Df[Df['particle']==int(Df['particle'].unique()[Q])]['Uy'])
#                UymeanALL.append(Uymean)
#                ax.plot([Ecartmean],[Uymean],'+',color=ColorList[i])
#
#            ax.plot([np.mean(EcartmeanALL)],[np.mean(UymeanALL)],'o',color='#000000',markerfacecolor=ColorList[i],label=ListDataFName[i])
#            EcartmeanALLPERNEEDLE.append(np.mean(EcartmeanALL))
#            UymeanALLPERNEEDLE.append(np.mean(UymeanALL))
#        ax.plot(EcartmeanALLPERNEEDLE,UymeanALLPERNEEDLE,color='#000000')
#
#        ax.legend(fontsize=5)    
#    
#    
#    
#def plotDISTINTERPART_VMOY2(ListDataF,ColorList):
#
#    Ww=[[0,7],[7,15],[15,21],[21,28],[28,36],[36,44],[44,55],[55,65]]
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.set_title('for needle and flow rates \n mean Dist with the nearest above bubble and mean Uy')
#    ax.set_xlabel('mean Dist with the nearest above bubble normalized by radius')
#    ax.set_ylabel('mean Uy')
#    for w in np.arange(0,len(Ww),1):
#        EcartmeanALLPERNEEDLE=[]
#        EcartstdALLPERNEEDLE=[]
#        UymeanALLPERNEEDLE=[]
#        UystdALLPERNEEDLE=[]
#        for i in np.arange(Ww[w][0],Ww[w][1],1):
#            EcartmeanALL=[]
#            UymeanALL=[]
#            print(i)
#            Df=ListDataF[i]
#            for Q in np.arange(1,len(Df['particle'].unique())-1,1):
#                timeQ=list(set(np.asarray(Df[Df['particle']==int(Df['particle'].unique()[Q])]['time'])) & set(np.asarray(Df[Df['particle']==int(Df['particle'].unique()[Q-1])]['time'])))
#                timeQ.sort()
#                Dftemp=Df[Df['time'].isin(timeQ)]
#                B0=Dftemp[Dftemp['particle']==int(Df['particle'].unique()[Q])]['y']
#                C0=Dftemp[Dftemp['particle']==int(Df['particle'].unique()[Q-1])]['y']
#                B0=np.asarray(B0)
#                C0=np.asarray(C0)
#                B=[(B0[k]-C0[k])/np.mean(Dftemp[Dftemp['particle']==int(Df['particle'].unique()[Q])]['radius']) for k in np.arange(0,min(len(B0),len(C0)),1)]
#                Ecartmean=np.mean(B)
#                EcartmeanALL.append(Ecartmean)
#                Uymean=np.mean(Df[Df['particle']==int(Df['particle'].unique()[Q])]['Uy'])
#                UymeanALL.append(Uymean)
##                ax.plot([Ecartmean],[Uymean],'+',color=ColorList[i])
#            ax.errorbar([np.mean(EcartmeanALL)],[np.mean(UymeanALL)],[np.std(UymeanALL)],[np.std(EcartmeanALL)],'o',color='#000000',markerfacecolor=ColorList[i],label=ListDataFName[i])
#            EcartmeanALLPERNEEDLE.append(np.mean(EcartmeanALL))
#            UymeanALLPERNEEDLE.append(np.mean(UymeanALL))
#        ax.plot(EcartmeanALLPERNEEDLE,UymeanALLPERNEEDLE,color='#000000')
#        ax.legend(fontsize=5)
#        
#def BrutVelocityRadiusEcart(ListDataF,ColorList):
#    df = pd.DataFrame()
#    Ww=[[0,7],[7,15],[15,21],[21,28],[28,36],[36,44],[44,55],[55,65]]
#    Needle=['A','B','C','D','E','F','G','H']
#    for i in np.arange(0,len(ListDataF),1):
#        print(str(i)+'/65')
#        Df=ListDataF[i]
#        Dfpart = pd.DataFrame()
#        Ecart=[]
#        Velocity=[]
#        Radius=[]
#        Particle=[]
#
#        for Q in np.arange(1,len(Df['particle'].unique())-1,1):
#            print(str(Q)+'/'+str(len(Df['particle'].unique()))+'part')
#            particleQ=int(Df['particle'].unique()[Q])
#            particleQ_1=int(Df['particle'].unique()[Q-1])
#            timeQ=list(set(np.asarray(Df[Df['particle']==particleQ]['time'])) & set(np.asarray(Df[Df['particle']==particleQ_1]['time'])))
#            timeQ.sort()
#            Dftemp=Df[Df['time'].isin(timeQ)]
#            B0=Dftemp[Dftemp['particle']==int(Df['particle'].unique()[Q])]['y']
#            C0=Dftemp[Dftemp['particle']==int(Df['particle'].unique()[Q-1])]['y']
#            B0=np.asarray(B0)
#            C0=np.asarray(C0)
#            B=[(B0[k]-C0[k])/np.mean(Dftemp[Dftemp['particle']==particleQ_1]['radius']) for k in np.arange(0,min(len(B0),len(C0)),1)]
#            if len(B)!=0:
#                if np.mean(B)>=0:
#                    Ecart.append(np.mean(B))
#                    Velocity.append(np.mean(Df[Df['particle']==particleQ_1]['Uy']))
#                    Radius.append(np.mean(Dftemp[Dftemp['particle']==particleQ_1]['radius']))
#                    Particle.append(particleQ_1)
#                else:
#                    Ecart.append(-np.mean(B))
#                    Velocity.append(np.mean(Df[Df['particle']==particleQ]['Uy']))
#                    Radius.append(np.mean(Dftemp[Dftemp['particle']==particleQ]['radius']))
#                    Particle.append(particleQ)
#        
#        Needle_Flowrate=[i]*len(Ecart)
#        for j in np.arange(0,len(Ww),1):
#            if i>=Ww[j][0] and i<Ww[j][1]:
#                needle=Needle[j]
#        Color=[ColorList[i]]*len(Ecart)
#        Name=[ListDataFName[i]]*len(Ecart)
#        needle=[needle]*len(Ecart)
#        
#        Dfpart['Ecart']=Ecart
#        Dfpart['Velocity']=Velocity
#        Dfpart['Radius']=Radius
#        Dfpart['Particle']=Particle
#        Dfpart['Needle_Flowrate']=Needle_Flowrate
#        Dfpart['Color']=Color
#        Dfpart['Name']=Name
#        Dfpart['needle']=needle
#        
#        df = pd.concat([df,Dfpart])
#        
#    return df
#        
#def plotDist_Vel(Df_VRE):
#    figbrut = plt.figure()
#    ax = figbrut.add_subplot(111)
#    ax.set_title('Vertical Velocity as a function of the dimensionless distance with the upper bubble.')
#    ax.set_xlabel('d/r : Distance (d) with the upper bubble divided by the radius (r) of the bubble. (dimensionless)')
#    ax.set_ylabel('Vertical Velocity (m/s)')
#    for i in Df_VRE['Needle_Flowrate'].unique():
#        print(i)
#        ax.plot(Df_VRE[Df_VRE['Needle_Flowrate']==i]['Ecart'],Df_VRE[Df_VRE['Needle_Flowrate']==i]['Velocity'],',',color=str(Df_VRE[Df_VRE['Needle_Flowrate']==i].loc[0,'Color']))
#
#def plotDist_Vel2(Df_VRE,ColorList,step,n):
#    Df_run_av=pd.DataFrame()
#    for i in Df_VRE['needle'].unique():
#        Df_run_av_needle=pd.DataFrame()
#
#        EcartMoyarray=[]
#        VelocityMoyarray=[]
#        VelocityStdarray=[]
#        Df_VRE_temp=Df_VRE[Df_VRE['needle']==i]
##        Df_VRE_temp.sort_values('Ecart', axis=0)
#        figbrut=plt.figure()
#        ax = figbrut.add_subplot(111)
#        ax.set_title('Vertical Velocity as a function of the dimensionless distance with the upper bubble.\n' +'Radius mean='+str(round(np.mean(Df_VRE_temp['Radius']),4))+'Radius std='+str(round(np.std(Df_VRE_temp['Radius']),4)) )
#        ax.set_xlabel('d/r : Distance (d) with the upper bubble divided by the radius (r) of the bubble. (dimensionless)')
#        ax.set_ylabel('Vertical Velocity (mm/s)')
#        for w in Df_VRE_temp['Needle_Flowrate'].unique():
#            ax.plot(Df_VRE_temp[Df_VRE_temp['Needle_Flowrate']==w]['Ecart'],Df_VRE_temp[Df_VRE_temp['Needle_Flowrate']==w]['Velocity'],'.',color=Df_VRE_temp[Df_VRE_temp['Needle_Flowrate']==w]['Color'].unique()[0],label=ListDataFName[int(w)])
#
#        Velocity=np.asarray(Df_VRE_temp[Df_VRE_temp['needle']==i]['Velocity'])
#        Ecart=np.asarray(Df_VRE_temp[Df_VRE_temp['needle']==i]['Ecart'])
#        for j in np.arange(5,len(Ecart)-step,(len(Ecart)-step-step)/n):
#            if i=='G':
#                Df_VRE_temp=Df_VRE_temp[Df_VRE_temp['Ecart']<60]
#                EcartMoyarray.append(j)
#                VelocityMoyarray.append(np.mean(Df_VRE_temp[(Df_VRE_temp['Ecart']>=j-step) & (Df_VRE_temp['Ecart']<=j+step)]['Velocity']))
#                VelocityStdarray.append(np.std(Df_VRE_temp[(Df_VRE_temp['Ecart']>=j-step) & (Df_VRE_temp['Ecart']<=j+step)]['Velocity']))
#            elif i=='F':
#                Df_VRE_temp=Df_VRE_temp[Df_VRE_temp['Ecart']<48]
#                EcartMoyarray.append(j)
#                VelocityMoyarray.append(np.mean(Df_VRE_temp[(Df_VRE_temp['Ecart']>=j-step) & (Df_VRE_temp['Ecart']<=j+step)]['Velocity']))
#                VelocityStdarray.append(np.std(Df_VRE_temp[(Df_VRE_temp['Ecart']>=j-step) & (Df_VRE_temp['Ecart']<=j+step)]['Velocity']))
#
#            elif i=='A':
#                Df_VRE_temp=Df_VRE_temp[Df_VRE_temp['Ecart']<68]
#                EcartMoyarray.append(j)
#                VelocityMoyarray.append(np.mean(Df_VRE_temp[(Df_VRE_temp['Ecart']>=j-step) & (Df_VRE_temp['Ecart']<=j+step)]['Velocity']))
#                VelocityStdarray.append(np.std(Df_VRE_temp[(Df_VRE_temp['Ecart']>=j-step) & (Df_VRE_temp['Ecart']<=j+step)]['Velocity']))
#
#            elif i=='H':
#                if j!=57:
#                    Df_VRE_temp=Df_VRE_temp[Df_VRE_temp['Ecart']>5]
#                    EcartMoyarray.append(j)
#                    VelocityMoyarray.append(np.mean(Df_VRE_temp[(Df_VRE_temp['Ecart']>=j-step) & (Df_VRE_temp['Ecart']<=j+step)]['Velocity']))
#                    VelocityStdarray.append(np.std(Df_VRE_temp[(Df_VRE_temp['Ecart']>=j-step) & (Df_VRE_temp['Ecart']<=j+step)]['Velocity']))
#
#            else:
#                EcartMoyarray.append(j)
#                VelocityMoyarray.append(np.mean(Df_VRE_temp[(Df_VRE_temp['Ecart']>=j-step) & (Df_VRE_temp['Ecart']<=j+step)]['Velocity']))
#                VelocityStdarray.append(np.std(Df_VRE_temp[(Df_VRE_temp['Ecart']>=j-step) & (Df_VRE_temp['Ecart']<=j+step)]['Velocity']))
#
#        ax.plot(EcartMoyarray,VelocityMoyarray,color='#000000',linewidth=3,label='Running Average')
#        ax.legend()
#        Df_run_av_needle['VelocityMoyarray']=VelocityMoyarray
#        Df_run_av_needle['VelocityMoyarrayNORM']=[VelocityMoyarray[k]/min(VelocityMoyarray) for k in np.arange(0,len(VelocityMoyarray),1)]
#        Df_run_av_needle['VelocityStdarray']=VelocityStdarray
#        Df_run_av_needle['VelocityStdarrayNORM']=[VelocityStdarray[k]/min(VelocityMoyarray) for k in np.arange(0,len(VelocityMoyarray),1)]
#
#        Df_run_av_needle['EcartMoyarray']=EcartMoyarray
#        Df_run_av_needle['needle']=[i]*len(VelocityMoyarray)
#        Df_run_av_needle['radius']=[round(np.mean(Df_VRE_temp['Radius']),4)]*len(VelocityMoyarray)
#        Df_run_av_needle['radiusstd']=[round(np.std(Df_VRE_temp['Radius']),4)]*len(VelocityMoyarray)
#        Df_run_av_needle['color']=[Df_VRE_temp[Df_VRE_temp['Needle_Flowrate']==Df_VRE_temp['Needle_Flowrate'].unique()[len(Df_VRE_temp['Needle_Flowrate'].unique())-1]]['Color'].unique()[0]]*len(VelocityMoyarray)
#        Df_run_av = pd.concat([Df_run_av,Df_run_av_needle])
#    return(Df_run_av)
#
#
#def plotDist_Vel2BISALL(Df_VRE,ColorList,step,n):
#    Df_run_av=pd.DataFrame()
#    figbrut=plt.figure()
#    ax = figbrut.add_subplot(111)
#    ax.set_title('Vertical Velocity as a function of the dimensionless distance with the upper bubble.')
#    ax.set_xlabel('d/r : Distance (d) with the upper bubble divided by the radius (r) of the bubble. (dimensionless)')
#    ax.set_ylabel('Vertical Velocity (mm/s)')
#    for i in Df_VRE['needle'].unique():
#        Df_run_av_needle=pd.DataFrame()
#
#        EcartMoyarray=[]
#        VelocityMoyarray=[]
#        VelocityStdarray=[]
#        Df_VRE_temp=Df_VRE[Df_VRE['needle']==i]
##        Df_VRE_temp.sort_values('Ecart', axis=0)
#
#        for w in Df_VRE_temp['Needle_Flowrate'].unique():
#            ax.plot(Df_VRE_temp[Df_VRE_temp['Needle_Flowrate']==w]['Ecart'],Df_VRE_temp[Df_VRE_temp['Needle_Flowrate']==w]['Velocity'],'|',color=Df_VRE_temp[Df_VRE_temp['Needle_Flowrate']==w]['Color'].unique()[0],label=ListDataFName[int(w)])
#
#        Velocity=np.asarray(Df_VRE_temp[Df_VRE_temp['needle']==i]['Velocity'])
#        Ecart=np.asarray(Df_VRE_temp[Df_VRE_temp['needle']==i]['Ecart'])
#        for j in np.arange(5,len(Ecart)-step,(len(Ecart)-step-step)/n):
#            if i=='G':
#                Df_VRE_temp=Df_VRE_temp[Df_VRE_temp['Ecart']<60]
#                EcartMoyarray.append(j)
#                VelocityMoyarray.append(np.mean(Df_VRE_temp[(Df_VRE_temp['Ecart']>=j-step) & (Df_VRE_temp['Ecart']<=j+step)]['Velocity']))
#                VelocityStdarray.append(np.std(Df_VRE_temp[(Df_VRE_temp['Ecart']>=j-step) & (Df_VRE_temp['Ecart']<=j+step)]['Velocity']))
#            elif i=='F':
#                Df_VRE_temp=Df_VRE_temp[Df_VRE_temp['Ecart']<48]
#                EcartMoyarray.append(j)
#                VelocityMoyarray.append(np.mean(Df_VRE_temp[(Df_VRE_temp['Ecart']>=j-step) & (Df_VRE_temp['Ecart']<=j+step)]['Velocity']))
#                VelocityStdarray.append(np.std(Df_VRE_temp[(Df_VRE_temp['Ecart']>=j-step) & (Df_VRE_temp['Ecart']<=j+step)]['Velocity']))
#
#            elif i=='A':
#                Df_VRE_temp=Df_VRE_temp[Df_VRE_temp['Ecart']<68]
#                EcartMoyarray.append(j)
#                VelocityMoyarray.append(np.mean(Df_VRE_temp[(Df_VRE_temp['Ecart']>=j-step) & (Df_VRE_temp['Ecart']<=j+step)]['Velocity']))
#                VelocityStdarray.append(np.std(Df_VRE_temp[(Df_VRE_temp['Ecart']>=j-step) & (Df_VRE_temp['Ecart']<=j+step)]['Velocity']))
#
#            elif i=='H':
#                if j!=57:
#                    Df_VRE_temp=Df_VRE_temp[Df_VRE_temp['Ecart']>5]
#                    EcartMoyarray.append(j)
#                    VelocityMoyarray.append(np.mean(Df_VRE_temp[(Df_VRE_temp['Ecart']>=j-step) & (Df_VRE_temp['Ecart']<=j+step)]['Velocity']))
#                    VelocityStdarray.append(np.std(Df_VRE_temp[(Df_VRE_temp['Ecart']>=j-step) & (Df_VRE_temp['Ecart']<=j+step)]['Velocity']))
#
#            else:
#                EcartMoyarray.append(j)
#                VelocityMoyarray.append(np.mean(Df_VRE_temp[(Df_VRE_temp['Ecart']>=j-step) & (Df_VRE_temp['Ecart']<=j+step)]['Velocity']))
#                VelocityStdarray.append(np.std(Df_VRE_temp[(Df_VRE_temp['Ecart']>=j-step) & (Df_VRE_temp['Ecart']<=j+step)]['Velocity']))
#
#
#        Df_run_av_needle['VelocityMoyarray']=VelocityMoyarray
#        Df_run_av_needle['VelocityMoyarrayNORM']=[VelocityMoyarray[k]/min(VelocityMoyarray) for k in np.arange(0,len(VelocityMoyarray),1)]
#        Df_run_av_needle['VelocityStdarray']=VelocityStdarray
#        Df_run_av_needle['VelocityStdarrayNORM']=[VelocityStdarray[k]/min(VelocityMoyarray) for k in np.arange(0,len(VelocityMoyarray),1)]
#
#        Df_run_av_needle['EcartMoyarray']=EcartMoyarray
#        Df_run_av_needle['needle']=[i]*len(VelocityMoyarray)
#        Df_run_av_needle['radius']=[round(np.mean(Df_VRE_temp['Radius']),4)]*len(VelocityMoyarray)
#        Df_run_av_needle['radiusstd']=[round(np.std(Df_VRE_temp['Radius']),4)]*len(VelocityMoyarray)
#        Df_run_av_needle['color']=[Df_VRE_temp[Df_VRE_temp['Needle_Flowrate']==Df_VRE_temp['Needle_Flowrate'].unique()[len(Df_VRE_temp['Needle_Flowrate'].unique())-1]]['Color'].unique()[0]]*len(VelocityMoyarray)
#        Df_run_av = pd.concat([Df_run_av,Df_run_av_needle])
#    
#    for i in Df_run_av['needle'].unique():
#        ax.plot(Df_run_av[Df_run_av['needle']==i]['EcartMoyarray'],Df_run_av[Df_run_av['needle']==i]['VelocityMoyarray'],color=Df_run_av[Df_run_av['needle']==i]['color'].unique()[0],linewidth=1.3,label='Running Average Needle '+str(i)+' Radius = '+str(round(Df_run_av[Df_run_av['needle']==i]['radius'].unique()[0],4))+' std = '+str(round(Df_run_av[Df_run_av['needle']==i]['radiusstd'].unique()[0],4)))
#    ax.legend(fontsize=4.7)
#    return(Df_run_av)
#
#fig=plt.figure()
#ax = fig.add_subplot(111)
#ax.set_title('Vertical Velocity as a function of the dimensionless distance with the upper bubble.\n')
#ax.set_xlabel('d/r : Distance (d) with the upper bubble divided by the radius (r) of the bubble. (dimensionless)')
#
#for s in Df_runn['needle'].unique():
#    
#
#    ax.plot(Df_runn[Df_runn['needle']==s]['EcartMoyarray'],Df_runn[Df_runn['needle']==s]['VelocityMoyarray'],linewidth=3,color=Df_runn[Df_runn['needle']==s]['color'].unique()[0],label='Needle '+str(Df_runn[Df_runn['needle']==s]['needle'].unique()[0])+' Radius = '+str(round(Df_runn[Df_runn['needle']==s]['radius'].unique()[0],4))+' std = '+str(round(Df_runn[Df_runn['needle']==s]['radiusstd'].unique()[0],4)))
#ax.legend()



#DF_par_pos        
#    figmoy = plt.figure()
#    ax = figmoy.add_subplot(111)
#    ax.set_title('Vertical Velocity as a function of the dimensionless distance with the upper bubble.')
#    ax.set_xlabel('d/r : Distance (d) with the upper bubble divided by the radius (r) of the bubble. (dimensionless)')
#    ax.set_ylabel('Vertical Velocity (m/s)')
#    for i in Df['Needle_Flowrate'].unique():
#        print(i)
#        ax.plot(Df[Df['Needle_Flowrate']==i]['Ecart'],Df[Df['Needle_Flowrate']==i]['Velocity'],'.',color=str(Df[Df['Needle_Flowrate']==i].loc[0,'Color']))
#
#    EcartMoyarray=[]
#    VelocityMoyarray=[]
#    Df_VRE_temp=Df_VRE[Df_VRE['needle']==i]
#    Df_VRE_temp.sort_values('Ecart', axis=0)
#    Velocity=np.asarray(Df_VRE_temp[Df_VRE_temp['needle']==i]['Velocity'])
#    Ecart=np.asarray(Df_VRE_temp[Df_VRE_temp['needle']==i]['Ecart'])
#    for j in np.arange(step,len(Ecart)-step,1):
#        EcartMoyarray.append(np.mean([Ecart[k] for k in np.arange(j-step,j+step,1)]))
#        
#    for j in np.arange(step,len(Velocity)-step,1):
#        VelocityMoyarray.append(np.mean([Velocity[k] for k in np.arange(j-step,j+step,1)]))
#    
#    
#    
#        
#plotVelocity(ListDataF,15,21,ColorList)
#plotVelocity(ListDataF,21,28,ColorList)
#plotVelocity(ListDataF,28,36,ColorList)
#plotVelocity(ListDataF,36,44,ColorList)
#plotVelocity(ListDataF,44,55,ColorList)
#plotVelocity(ListDataF,55,65,ColorList)



################### P L O T     L I F E T I M E     ########################################################################
#def LifeTime(Df00, RadiusMin=0,RadiusMax=10,LifetimeMin=0,LifetimeMax=10, style='.',viz=False):
##    tracked=tp.link_df(Df,search_range=80,memory=3)
##    Ra=plt.figure()
##########################this next ligne is used to determine Rmin and Rmax and the lifetime min
##    plotRfctTperP(Df00, style='+')
#    Df00['lifetime']=Df00['particle']
#    for p in Df00['particle'].unique():
#        index=Df00[Df00['particle']==p].index
#        lifetime=len(Df00[Df00['particle']==p])*dt
#        for i in index:
#            Df00.loc[i,'lifetime']=lifetime
#    Df00.sort_values(by='particle')
#    
#    'tri'
#    Df00=Df00[Df00['radius']>RadiusMin]
#    Df00=Df00[Df00['radius']<RadiusMax]
#    Df00=Df00[Df00['lifetime']>LifetimeMin]
#    Df00=Df00[Df00['lifetime']<LifetimeMax]
#    if viz:
#        plotRfctTperP(Df00, style=style)
#    return Df00
#
#def Histogramme(DfA,DfB,DfC,DfD,DfE,DfF,DfG):
#    H=plt.figure()
#    A=[DfA[DfA['particle']==p].iloc[0]['lifetime'] for p in DfA['particle'].unique()]
#    B=[DfB[DfB['particle']==p].iloc[0]['lifetime'] for p in DfB['particle'].unique()]
#    C=[DfC[DfC['particle']==p].iloc[0]['lifetime'] for p in DfC['particle'].unique()]
#    D=[DfD[DfD['particle']==p].iloc[0]['lifetime'] for p in DfD['particle'].unique()]
#    E=[DfE[DfE['particle']==p].iloc[0]['lifetime'] for p in DfE['particle'].unique()]+[DfF[DfF['particle']==p].iloc[0]['lifetime'] for p in DfF['particle'].unique()]
#    G=[DfG[DfG['particle']==p].iloc[0]['lifetime'] for p in DfG['particle'].unique()]
##    F=[DfF[DfF['particle']==p].iloc[0]['lifetime'] for p in DfF['particle'].unique()]
#    
#    plt.hist(A, len(list(set(A))),histtype='step',label='Needle01',normed = True)
#    plt.hist(B, len(list(set(B))),histtype='step',label='Needle02',normed = True)
#    plt.hist(C, len(list(set(C))),histtype='step',label='Needle03',normed = True)
#    plt.hist(D, len(list(set(D))),histtype='step',label='Needle04',normed = True)
#    plt.hist(E, len(list(set(E))),histtype='step',label='Needle0V',normed = True)
#    plt.hist(G, len(list(set(G))),histtype='step',label='Needle0orange',normed = True)
#
##    plt.hist(F, len(list(set(F))),histtype='step',label='Needle0Vsample02')
#    plt.title('Histogram of the life time of the bubble')
#    plt.legend()
#    return A
#
#def Histogrammeradius(DfA,DfB,DfC,DfD,DfE,DfF,DfG):
#    H=plt.figure()
#    A=[DfA[DfA['particle']==p].iloc[0]['lifetime'] for p in DfA['particle'].unique()]
#    B=[DfB[DfB['particle']==p].iloc[0]['lifetime'] for p in DfB['particle'].unique()]
#    C=[DfC[DfC['particle']==p].iloc[0]['lifetime'] for p in DfC['particle'].unique()]
#    D=[DfD[DfD['particle']==p].iloc[0]['lifetime'] for p in DfD['particle'].unique()]
#    E=[DfE[DfE['particle']==p].iloc[0]['lifetime'] for p in DfE['particle'].unique()]+[DfF[DfF['particle']==p].iloc[0]['lifetime'] for p in DfF['particle'].unique()]
#    G=[DfG[DfG['particle']==p].iloc[0]['lifetime'] for p in DfG['particle'].unique()]
##    F=[DfF[DfF['particle']==p].iloc[0]['lifetime'] for p in DfF['particle'].unique()]
#    
#    plt.hist(A, len(list(set(A))),histtype='step',label='Needle01')
#    plt.hist(B, len(list(set(B))),histtype='step',label='Needle02')
#    plt.hist(C, len(list(set(C))),histtype='step',label='Needle03')
#    plt.hist(D, len(list(set(D))),histtype='step',label='Needle04')
#    plt.hist(E, len(list(set(E))),histtype='step',label='Needle0V')
#    plt.hist(G, len(list(set(G))),histtype='step',label='Needle0orange')
#
##    plt.hist(F, len(list(set(F))),histtype='step',label='Needle0Vsample02')
#    plt.title('Histogram of the life time of the bubble')
#    plt.legend()
#    return A
#
#
#
#
#def histogramme(df_bubble,df_cluster,frames):
#    
#    'Number of bubbles per cluster as a function of time'
#    max_num_bubble_per_cluster=max(df_cluster['num_of_bubble'])
#    data=np.zeros([max_num_bubble_per_cluster+1,len(frames)],dtype=long)
#    data[0]=frames
#    for f in np.arange(0,len(frames),1):
#        for j in df_cluster[df_cluster['frame']==frames[f]]['num_of_bubble'].value_counts().index:
#            data[j][f]=df_cluster[df_cluster['frame']==frames[f]]['num_of_bubble'].value_counts().loc[j]
#    bottom=np.zeros_like(data[0])
#    hist=plt.figure()
#    for i in np.arange(1,max_num_bubble_per_cluster+1,1):
#        P=plt.bar(frames,data[i],bottom=bottom,label="num_b_in_c=%d"%(i,),width=1)
#        bottom=bottom+data[i]
#    plt.legend()
#    plt.title('Number of bubbles per cluster as a function of time')
#    plt.xlabel('frame')
#    plt.ylabel('Number of clusters')
#    
##    'Histogramme of the radius of the cluster with a given size as a function of time A REVOOOIIIIIR LOL'
##    fig,axs = plt.subplots(2,2,figsize=(17,9),sharex=True,sharey=False)
##    axs = axs.flatten()
##    datas = [df_cluster[df_cluster['num_of_bubble']==i]['radius'] for i in sorted(df_cluster['num_of_bubble'].unique())]
##    names = ['Radius of bubbles in clusters of '+str(i)+' bubbles' for i in sorted(df_cluster['num_of_bubble'].unique())]
##    for i,(d,name) in enumerate(zip(datas,names)):
##        axs[i].hist(d,bins=500)
##        axs[i].set_title(name)
##        axs[i].set_xlabel('radius')
##        axs[i].set_ylabel('num of bubble')
#
#    list_of_particle_clusters = [df_cluster[df_cluster['particle']==p] for p in df_cluster['particle'].unique()]
#    
#    'plot trajectories'
#    traj = plt.figure()
#    plt.title('trajectories of the particles y fct x')
#    ax = traj.add_subplot(111)
#    for bubble_df in list_of_particle_clusters:
#        ax.plot(bubble_df['x'],bubble_df['y'])
#        plt.legend()
#    
#def plotYfctTimepluslabel(Df,name='frd'):    
#    traj = plt.figure()
#    plt.gca().invert_yaxis()
#    plt.title('Trajectories of the '+str(len(Df['particle'].unique()))+' bubbles of record '+name)
#
#    ax = traj.add_subplot(111)
#    for p in Df['particle'].unique():
#        ax.plot(Df[Df['particle']==p]['time'],Df[Df['particle']==p]['y'])
#        ax.text(np.min(Df[Df['particle']==p]['time']),np.max(Df[Df['particle']==p]['y']),str(p))
#    
#    'Evolution num bubbles as a function of time for each cluster'
#    NBT = plt.figure()
#    plt.title('Evolution num bubbles as a function of time for each cluster')
#    axNBT = NBT.add_subplot(111)
#    for particle_clusters in list_of_particle_clusters:
#        axNBT.plot(particle_clusters['time'],particle_clusters['num_of_bubble'],label="particle=%d"%(np.mean(particle_clusters['particle'],)))
#    plt.legend()
#
#    'Evolution overall num bubbles AND overall num cluster as a function of time'
#    NOBT = plt.figure()
#    plt.title('Evolution overall num bubbles AND overall num cluster as a function of time')
#    A=df_bubble['time'].unique()
#    B=[len(df_bubble[df_bubble['time']==time]) for time in df_bubble['time'].unique()]
#    C=[len(df_cluster[df_cluster['time']==time]) for time in df_cluster['time'].unique()]
#    D=[float(len(df_bubble[df_bubble['time']==time]))/float(len(df_cluster[df_cluster['time']==time])) for time in df_cluster['time'].unique()]
#    
#    std_D = np.sqrt(abs(np.asarray(D) - np.mean(D))**2)
#    
#    axNOBT = NOBT.add_subplot(111)
#    axNOBT.plot(A,B,label='bubbles')
#    axNOBT.plot(A,C,label='clusters')
#    axNOBT.errorbar(A, D, std_D, linestyle='None', marker='x',label='num bubble per cluster')
#    plt.legend()
#
#    'Evolution overall num bubbles as a function of time'
#    NOBT = plt.figure()
#    plt.title('Evolution overall num bubbles as a function of time')
#    A=df_bubble['time'].unique()
#    B=[len(df_bubble[df_bubble['time']==time]) for time in df_bubble['time'].unique()]
#    plt.plot(A,B)  
#
#
#
#    'plot radius as a fct of time for each '
#    traj2 = plt.figure()
#    plt.title('trajectories with various linewidth of the particles y fct x')
#    ax2 = traj2.add_subplot(111)
#    for bubble_df in list_of_particle_clusters:
#        list_of_particle_cst_num_bubble = [bubble_df[bubble_df['num_of_bubble']==p] for p in bubble_df['num_of_bubble'].unique()]
##        c = "%06x" % random.randint(0, 0xFFFFFF)
#        for cst_num in list_of_particle_cst_num_bubble:
#            linewidth=int(np.mean(cst_num['num_of_bubble']))*1.5
#            ax2.plot(cst_num['x'],cst_num['y'],linewidth=linewidth)
#    
#    'Histogramme of the radius of the cluster with a given size as a function of time A REVOOOIIIIIR LOL'
#    fig = plt.plots()
#    datas = [df_cluster[df_cluster['num_of_bubble']==i]['radius'] for i in sorted(df_cluster['num_of_bubble'].unique())]
#    names = ['Radius of bubbles in clusters of '+str(i)+' bubbles' for i in sorted(df_cluster['num_of_bubble'].unique())]
#    for i,(d,name) in enumerate(zip(datas,names)):
#        axs[i].hist(d,bins=500)
#        axs[i].set_title(name)
#        axs[i].set_xlabel('radius')
#        axs[i].set_ylabel('num of bubble')
#    
#    '................plot paper.......................'
#    'Histogramme of the radius of the bubbles as a function of time'
#    fig, ax = plt.subplots(tight_layout=True)
#    hist = ax.hist2d(np.asarray(Z['time'],dtype='float64'),np.asarray(Z['radius'],dtype='float64'),bins=(2737, 60))
#    ax.set_title('Histogramme of the radius of the bubbles as a function of time')
#    ax.set_xlabel('time')
#    ax.set_ylabel('radius')
#    plt.colorbar(hist[3],ax=ax)
#
def plotLifeTimefctRmeanperPall(DfA,DfB,DfC,DfD,DfE,DfF,DfG,DfH):
    'Lifetime as a function of the radius mean of each particle for 6 needle'
    LT = plt.figure()
    plt.plot([np.mean(DfA[DfA['particle']==p]['radius']) for p in DfA['particle'].unique()],[len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()],',',label='Needle A : LifeTimeMean='+str(round(np.mean([len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfA['radius']),3))+'  std='+str(round(np.std(DfA['radius']),3))+'  NumPart='+str(len(DfA['particle'].unique())))
    plt.plot([np.mean(DfB[DfB['particle']==p]['radius']) for p in DfB['particle'].unique()],[len(DfB[DfB['particle']==p])*dt for p in DfB['particle'].unique()],',',label='Needle B : LifeTimeMean='+str(round(np.mean([len(DfB[DfB['particle']==p])*dt for p in DfB['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfB[DfB['particle']==p])*dt for p in DfB['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfB['radius']),3))+'  std='+str(round(np.std(DfB['radius']),3))+'  NumPart='+str(len(DfB['particle'].unique())))
    plt.plot([np.mean(DfC[DfC['particle']==p]['radius']) for p in DfC['particle'].unique()],[len(DfC[DfC['particle']==p])*dt for p in DfC['particle'].unique()],',',label='Needle C : LifeTimeMean='+str(round(np.mean([len(DfC[DfC['particle']==p])*dt for p in DfC['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfC[DfC['particle']==p])*dt for p in DfC['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfC['radius']),3))+'  std='+str(round(np.std(DfC['radius']),3))+'  NumPart='+str(len(DfC['particle'].unique())))
    plt.plot([np.mean(DfD[DfD['particle']==p]['radius']) for p in DfD['particle'].unique()],[len(DfD[DfD['particle']==p])*dt for p in DfD['particle'].unique()],',',label='Needle D : LifeTimeMean='+str(round(np.mean([len(DfD[DfD['particle']==p])*dt for p in DfD['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfD[DfD['particle']==p])*dt for p in DfD['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfD['radius']),3))+'  std='+str(round(np.std(DfD['radius']),3))+'  NumPart='+str(len(DfD['particle'].unique())))
    plt.plot([np.mean(DfE[DfE['particle']==p]['radius']) for p in DfE['particle'].unique()],[len(DfE[DfE['particle']==p])*dt for p in DfE['particle'].unique()],',',label='Needle E : LifeTimeMean='+str(round(np.mean([len(DfE[DfE['particle']==p])*dt for p in DfE['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfE[DfE['particle']==p])*dt for p in DfE['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfE['radius']),3))+'  std='+str(round(np.std(DfE['radius']),3))+'  NumPart='+str(len(DfE['particle'].unique())))
    plt.plot([np.mean(DfF[DfF['particle']==p]['radius']) for p in DfF['particle'].unique()],[len(DfF[DfF['particle']==p])*dt for p in DfF['particle'].unique()],',',label='Needle F : LifeTimeMean='+str(round(np.mean([len(DfF[DfF['particle']==p])*dt for p in DfF['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfF[DfF['particle']==p])*dt for p in DfF['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfF['radius']),3))+'  std='+str(round(np.std(DfF['radius']),3))+'  NumPart='+str(len(DfF['particle'].unique())))
    plt.plot([np.mean(DfG[DfG['particle']==p]['radius']) for p in DfG['particle'].unique()],[len(DfG[DfG['particle']==p])*dt for p in DfG['particle'].unique()],',',label='Needle G : LifeTimeMean='+str(round(np.mean([len(DfG[DfG['particle']==p])*dt for p in DfG['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfG[DfG['particle']==p])*dt for p in DfG['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfG['radius']),3))+'  std='+str(round(np.std(DfG['radius']),3))+'  NumPart='+str(len(DfG['particle'].unique())))
    plt.plot([np.mean(DfH[DfH['particle']==p]['radius']) for p in DfH['particle'].unique()],[len(DfH[DfH['particle']==p])*dt for p in DfH['particle'].unique()],',',label='Needle H : LifeTimeMean='+str(round(np.mean([len(DfH[DfH['particle']==p])*dt for p in DfH['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfH[DfH['particle']==p])*dt for p in DfH['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfH['radius']),3))+'  std='+str(round(np.std(DfH['radius']),3))+'  NumPart='+str(len(DfH['particle'].unique())))
    plt.title('Lifetime as a function of the radius mean of each particle Needle')
    plt.xlabel('Mean Radius of each particle in mm')
    plt.ylabel('Life time of each particle in s')
    plt.legend()




def plotLifetimelabelPart(df,step,name,folder,dt):
#    for df in df_t['variable'].tolist():
#       df.LifeTimeMean=np.mean([len(df.df_tracked[df.df_tracked['particle']==p])*df.dt for p in df.df_tracked['particle'].unique()])
#       df.LifeTimeStd=np.std([len(df.df_tracked[df.df_tracked['particle']==p])*df.dt for p in df.df_tracked['particle'].unique()])
    plt.figure(figsize=(6, 6))
    Q= df['particle'].unique().tolist()
    plt.plot(np.arange(0,len(Q),1),[len(df[df['particle']==p])*dt for p in df['particle'].unique()],'.',label='Needle'+name+' : LifeTimeMean='+str(round(np.mean([len(df[df['particle']==p])*dt for p in df['particle'].unique()]),3))+'\n'+'   std='+str(round(np.std([len(df[df['particle']==p])*dt for p in df['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(df['radius']),3))+'  std='+str(round(np.std(df['radius']),3))+'  NumPart='+str(len(df['particle'].unique())))
    plt.xlabel('label particle')
    plt.ylabel('Life time of the labeled particle in s')
    
    QMoyarray=[]
    LifetimeMoyarray=[]

    Lifetime=[len(df[df['particle']==p])*dt for p in df['particle'].unique()]
    for j in np.arange(step,len(Q)-step,1):
        QMoyarray.append(np.mean([Q[k] for k in np.arange(j-step,j+step,1)]))
        
    for j in np.arange(step,len(Q)-step,1):
        LifetimeMoyarray.append(np.mean([Lifetime[k] for k in np.arange(j-step,j+step,1)]))
    plt.plot(np.arange(step,len(Q)-step,1),LifetimeMoyarray,label='running average')
    
    plt.legend()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.savefig(folder+'\\'+name+'_plotLifetimelabelPart.png')

def plotLifetimeFCTRadius(liste,dt):
    plt.plot(figsize=(12, 12))
    for df in liste:
        color='#'+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        plt.plot([np.mean(df.df_tracked[df.df_tracked['particle']==p]['radius']) for p in df.df_tracked['particle'].unique()],[len(df.df_tracked[df.df_tracked['particle']==p])*dt for p in df.df_tracked['particle'].unique()],',',label=df.name+'for each particle',color=color)
        
        plt.errorbar([df.Rmean],[df.LifeTimeMean],[df.LifeTimeStd],marker='D',label=df.name+' mean',color=color)
    plt.legend(prop={'size':8})
    plt.xlabel('Radius in mm')
    plt.ylabel('Lifetime in s')
    plt.title('Lifetime as a function of the radius, tap water')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.savefig(liste[0].path_folder+'\\'+'ALLplotLifetimefctPart.png')

def plotLifetimeFCTsccm(liste,dt):
    plt.plot(figsize=(12, 12))
    for df in liste:
#        color='#'+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        plt.plot([df.Flow_rate_sccm]*len([len(df.df_tracked[df.df_tracked['particle']==p])*dt for p in df.df_tracked['particle'].unique()]),[len(df.df_tracked[df.df_tracked['particle']==p])*dt for p in df.df_tracked['particle'].unique()],',',label=df.name+'for each particle')
        
        plt.errorbar([df.Flow_rate_sccm],[df.LifeTimeMean],[df.LifeTimeStd],marker='D',label=df.name+' mean',color='black')
    plt.legend(prop={'size':8})
    plt.xlabel('flow rate in sccm')
    plt.ylabel('Lifetime in s')
    plt.title('Lifetime as a function of the sccm, tap water, RADIUS=0.71mm')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.savefig(liste[0].path_folder+'\\'+'ALLplotLifetimefctsccm.png')
    
def plotLifetimeFCTsurf(liste,dt):
    plt.plot(figsize=(12, 12))
    for df in liste:
#        color='#'+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        plt.plot([df.Surfactant]*len([len(df.df_tracked[df.df_tracked['particle']==p])*dt for p in df.df_tracked['particle'].unique()]),[len(df.df_tracked[df.df_tracked['particle']==p])*dt for p in df.df_tracked['particle'].unique()],',',label=df.name+'for each particle')
        
        plt.errorbar([df.Surfactant],[df.LifeTimeMean],[df.LifeTimeStd],marker='D',label=df.name+' mean   Surfactant:'+str(df.Surfactant),color='black')
    plt.legend(prop={'size':8})
    plt.xlabel('flow rate in sccm')
    plt.ylabel('Lifetime in s')
    plt.title('Lifetime as a function of the surfactant cocentration, RADIUS=1.23mm')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.savefig(liste[0].path_folder+'\\'+'ALLplotLifetimefctsccm.png')


def PDF_0(x):
#    scipy.special.gamma(7.0/4.0)
    PDF=(4.0/3.0)*(scipy.special.gamma(7.0/4.0)**(4.0/3.0))*(x**(1.0/3.0))*np.exp(-((scipy.special.gamma(7.0/4.0)*x)**(4.0/3.0)))
    return PDF

def PDF_1(x):
#    scipy.special.gamma(7.0/4.0)
    PDF=(4.0/3.0)*(0.92**(4.0/3.0))*(x**(1.0/3.0))*np.exp(-((0.92*x)**(4.0/3.0)))
    return PDF

def plotHistLifeTimeEXPFITNORM_ALLSAMETIME(liste,dt=0.02,name_all=''):

    LifeTime_Normalized_ALL=[]
    for i in np.arange(0,len(liste),1):
        Rmean=liste[i].Rmean
        
        df=liste[i].df_tracked_filled

        LifeTime_All=[len(df[df['particle']==p])*dt for p in df['particle'].unique()]

        LifeTime_Mean=np.mean(LifeTime_All)
        
        LifeTime_Normalized=[x/LifeTime_Mean for x in LifeTime_All]
        
        LifeTime_Normalized_ALL=LifeTime_Normalized_ALL+LifeTime_Normalized
    
    bins=int(len(LifeTime_Normalized_ALL)**(1.0/3.0))+3
    inter=(np.max(LifeTime_Normalized_ALL)-np.min(LifeTime_Normalized_ALL))/bins
    weights = np.ones_like(LifeTime_Normalized_ALL)/(float(len(LifeTime_Normalized_ALL))*inter)
    AA=np.histogram(LifeTime_Normalized_ALL, weights=weights,bins=bins)
    
    LTE = plt.figure(figsize=(18,13))
    plt.title('PDF(normalized lifetime) all radius TAP WATER '+name_all)

    plt.semilogy([AA[1][k] for k in np.arange(0,len(AA[0]),1)],AA[0],'o',label='NumPart = '+str(len(LifeTime_Normalized_ALL)))
    plt.plot(np.arange(0.001,12,0.001),[PDF_0(x) for x in np.arange(0.001,12,0.001)],label='fit')
    
    plt.xlim(-0.2,10)
    plt.ylim(0.0001,3)
    plt.xlabel('Normalized Lifetime ')
    plt.ylabel('PDF(normalized lifetime)')
    plt.legend()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

#def BINS(array):
##    https://en.wikipedia.org/wiki/Histogram#Number_of_bins_and_width
#    n=len(array)
#    
#    sr=int(np.sqrt(n))+1
#    print('0 : Square-root choice'+' = '+str(sr))
#        
#    sturges=int(np.log2(n))+1
#    print('1 : Sturges formula'+' = '+str(sturges))
#        
#    RiceRule=int(2.0*n**(1.0/3.0))+1
#    print('2 : Rice Rule'+' = '+str(RiceRule))
#    
#    skewness=scipy.stats.skew(array)
#    sigmag1=np.sqrt(6.0*(n-2.0)/float((n+1.0)*(n+3.0)))
#    Doane=int(1+np.log2(n)+np.log2(1.0+skewness/sigmag1))+1
#    print('3 : Doane formula'+' = '+str(Doane))
#
#    h_scott=3.5*np.std(array)/(n**(1.0/3.0))
#    scott=int((np.max(array)-np.min(array))/h_scott)+1
#    print('4 : Scott s normal reference rule'+' = '+str(scott))
#    
#    h_Freedman_Diaconi=2.0*scipy.stats.iqr(array)/(n**(1.0/3.0))
#    Freedman_Diaconi=int((np.max(array)-np.min(array))/h_Freedman_Diaconi)+1
#    print('5 : Freedman–Diaconis choice'+' = '+str(Freedman_Diaconi))
#    
#    CubeRoot=int(n**(1.0/3.0))+1
#    print('6 : CubeRoot = '+str(CubeRoot))
#    return([sr,sturges,RiceRule,Doane,scott,Freedman_Diaconi,CubeRoot])



def plotHistLifeTimeEXPFITNORM_list(liste,dt=0.02,name_all=''):
    columns = int(np.sqrt(len(liste)/0.7))
    rows = int(len(liste)/columns)+1
    fig, ax = plt.subplots(rows, columns,squeeze=False, sharey = True, sharex = True,figsize=(18,13))
    folder=liste[0].path_folder
    fig.suptitle('PDF(normalized lifetime) TAP WATER'+name_all)
    
    for i in np.arange(0,len(liste),1):
        Rmean=liste[i].Rmean
        name=liste[i].name
        df=liste[i].df_tracked_filled
        
        ax0=ax.flatten()[i]
        
        LifeTime_All=[len(df[df['particle']==p])*dt for p in df['particle'].unique()]
                
        LifeTime_Mean=np.mean(LifeTime_All)
        
        LifeTime_Normalized=[x/LifeTime_Mean for x in LifeTime_All]
        
        bins=int(len(LifeTime_Normalized)**(1.0/3.0))+3
        
        inter=(np.max(LifeTime_Normalized)-np.min(LifeTime_Normalized))/bins

        weights = np.ones_like(LifeTime_Normalized)/float(len(LifeTime_Normalized)*inter)
        

        AA=np.histogram(LifeTime_Normalized,bins=bins, weights=weights)

        ax0.semilogy([AA[1][k] for k in np.arange(0,len(AA[0]),1)],AA[0],'o',label=name+'LTMean='+str(round(LifeTime_Mean,3))+' std='+str(round(np.std(LifeTime_All)/2.0,3))+' R='+str(round(Rmean,3))+' NumPart='+str(len(df['particle'].unique())))
        ax0.plot(np.arange(0.001,12,0.001),[PDF_0(x) for x in np.arange(0.001,12,0.001)],label='fit')
        ax0.set_xlim(-0.2,10)
        ax0.set_ylim(0.0001,3)
        ax0.set_title('PDF(normalized lifetime) Rmean = '+str(Rmean), fontsize=8)
        ax0.legend(fontsize=6)






#'''
#for i in np.arange(10,120,10):
#    plotHistLifeTime(DfA,DfB,DfC,DfD,DfE,DfF,DfG,DfH,bins=i)
#'''
#
#def plotHistLifeTime(DfA,DfB,DfC,DfD,DfE,DfF,DfG,DfH,bins=15):
#    
#    'Lifetime as a function of the radius mean of each particle for 6 needle'
#    weightsA = np.ones_like([len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()])/float(len([len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()]))
#    weightsB = np.ones_like([len(DfB[DfB['particle']==p])*dt for p in DfB['particle'].unique()])/float(len([len(DfB[DfB['particle']==p])*dt for p in DfB['particle'].unique()]))
#    weightsC = np.ones_like([len(DfC[DfC['particle']==p])*dt for p in DfC['particle'].unique()])/float(len([len(DfC[DfC['particle']==p])*dt for p in DfC['particle'].unique()]))
#    weightsD = np.ones_like([len(DfD[DfD['particle']==p])*dt for p in DfD['particle'].unique()])/float(len([len(DfD[DfD['particle']==p])*dt for p in DfD['particle'].unique()]))
#    weightsE = np.ones_like([len(DfE[DfE['particle']==p])*dt for p in DfE['particle'].unique()])/float(len([len(DfE[DfE['particle']==p])*dt for p in DfE['particle'].unique()]))
#    weightsF = np.ones_like([len(DfF[DfF['particle']==p])*dt for p in DfF['particle'].unique()])/float(len([len(DfF[DfF['particle']==p])*dt for p in DfF['particle'].unique()]))
#    weightsG = np.ones_like([len(DfG[DfG['particle']==p])*dt for p in DfG['particle'].unique()])/float(len([len(DfG[DfG['particle']==p])*dt for p in DfG['particle'].unique()]))
#    weightsH = np.ones_like([len(DfH[DfH['particle']==p])*dt for p in DfH['particle'].unique()])/float(len([len(DfH[DfH['particle']==p])*dt for p in DfH['particle'].unique()]))
#
#    LTE = plt.figure()
#    AA=plt.hist([len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()],histtype='step', bins=bins, weights=weightsA,label='Needle A : LifeTimeMean='+str(round(np.mean([len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfA['radius']),3))+'  std='+str(round(np.std(DfA['radius']),3))+'  NumPart='+str(len(DfA['particle'].unique())))
#    BB=plt.hist([len(DfB[DfB['particle']==p])*dt for p in DfB['particle'].unique()],histtype='step', bins=bins, weights=weightsB,label='Needle B : LifeTimeMean='+str(round(np.mean([len(DfB[DfB['particle']==p])*dt for p in DfB['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfB[DfB['particle']==p])*dt for p in DfB['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfB['radius']),3))+'  std='+str(round(np.std(DfB['radius']),3))+'  NumPart='+str(len(DfB['particle'].unique())))
#    CC=plt.hist([len(DfC[DfC['particle']==p])*dt for p in DfC['particle'].unique()],histtype='step', bins=bins, weights=weightsC,label='Needle C : LifeTimeMean='+str(round(np.mean([len(DfC[DfC['particle']==p])*dt for p in DfC['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfC[DfC['particle']==p])*dt for p in DfC['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfC['radius']),3))+'  std='+str(round(np.std(DfC['radius']),3))+'  NumPart='+str(len(DfC['particle'].unique())))
#    DD=plt.hist([len(DfD[DfD['particle']==p])*dt for p in DfD['particle'].unique()],histtype='step', bins=bins, weights=weightsD,label='Needle D : LifeTimeMean='+str(round(np.mean([len(DfD[DfD['particle']==p])*dt for p in DfD['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfD[DfD['particle']==p])*dt for p in DfD['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfD['radius']),3))+'  std='+str(round(np.std(DfD['radius']),3))+'  NumPart='+str(len(DfD['particle'].unique())))
#    EE=plt.hist([len(DfE[DfE['particle']==p])*dt for p in DfE['particle'].unique()],histtype='step', bins=bins, weights=weightsE,label='Needle E : LifeTimeMean='+str(round(np.mean([len(DfE[DfE['particle']==p])*dt for p in DfE['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfE[DfE['particle']==p])*dt for p in DfE['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfE['radius']),3))+'  std='+str(round(np.std(DfE['radius']),3))+'  NumPart='+str(len(DfE['particle'].unique())))
#    FF=plt.hist([len(DfF[DfF['particle']==p])*dt for p in DfF['particle'].unique()],histtype='step', bins=bins, weights=weightsF,label='Needle F : LifeTimeMean='+str(round(np.mean([len(DfF[DfF['particle']==p])*dt for p in DfF['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfF[DfF['particle']==p])*dt for p in DfF['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfF['radius']),3))+'  std='+str(round(np.std(DfF['radius']),3))+'  NumPart='+str(len(DfF['particle'].unique())))
#    GG=plt.hist([len(DfG[DfG['particle']==p])*dt for p in DfG['particle'].unique()],histtype='step', bins=bins, weights=weightsG,label='Needle G : LifeTimeMean='+str(round(np.mean([len(DfG[DfG['particle']==p])*dt for p in DfG['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfG[DfG['particle']==p])*dt for p in DfG['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfG['radius']),3))+'  std='+str(round(np.std(DfG['radius']),3))+'  NumPart='+str(len(DfG['particle'].unique())))
#    HH=plt.hist([len(DfH[DfH['particle']==p])*dt for p in DfH['particle'].unique()],histtype='step', bins=bins, weights=weightsH,label='Needle H : LifeTimeMean='+str(round(np.mean([len(DfH[DfH['particle']==p])*dt for p in DfH['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfH[DfH['particle']==p])*dt for p in DfH['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfH['radius']),3))+'  std='+str(round(np.std(DfH['radius']),3))+'  NumPart='+str(len(DfH['particle'].unique())))
#    
#    LTE2=plt.figure()
#    plt.semilogy([AA[1][k] for k in np.arange(0,len(AA[0]),1)],AA[0],'o',label='Needle A : LifeTimeMean='+str(round(np.mean([len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfA['radius']),3))+'  std='+str(round(np.std(DfA['radius']),3))+'  NumPart='+str(len(DfA['particle'].unique())))
#    plt.semilogy([BB[1][k] for k in np.arange(0,len(BB[0]),1)],BB[0],'o',label='Needle B : LifeTimeMean='+str(round(np.mean([len(DfB[DfB['particle']==p])*dt for p in DfB['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfB[DfB['particle']==p])*dt for p in DfB['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfB['radius']),3))+'  std='+str(round(np.std(DfB['radius']),3))+'  NumPart='+str(len(DfB['particle'].unique())))
#    plt.semilogy([CC[1][k] for k in np.arange(0,len(CC[0]),1)],CC[0],'o',label='Needle C : LifeTimeMean='+str(round(np.mean([len(DfC[DfC['particle']==p])*dt for p in DfC['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfC[DfC['particle']==p])*dt for p in DfC['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfC['radius']),3))+'  std='+str(round(np.std(DfC['radius']),3))+'  NumPart='+str(len(DfC['particle'].unique())))
#    plt.semilogy([DD[1][k] for k in np.arange(0,len(DD[0]),1)],DD[0],'o',label='Needle D : LifeTimeMean='+str(round(np.mean([len(DfD[DfD['particle']==p])*dt for p in DfD['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfD[DfD['particle']==p])*dt for p in DfD['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfD['radius']),3))+'  std='+str(round(np.std(DfD['radius']),3))+'  NumPart='+str(len(DfD['particle'].unique())))
#    plt.semilogy([EE[1][k] for k in np.arange(0,len(EE[0]),1)],EE[0],'o',label='Needle E : LifeTimeMean='+str(round(np.mean([len(DfE[DfE['particle']==p])*dt for p in DfE['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfE[DfE['particle']==p])*dt for p in DfE['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfE['radius']),3))+'  std='+str(round(np.std(DfE['radius']),3))+'  NumPart='+str(len(DfE['particle'].unique())))
#    plt.semilogy([FF[1][k] for k in np.arange(0,len(FF[0]),1)],FF[0],'o',label='Needle F : LifeTimeMean='+str(round(np.mean([len(DfF[DfF['particle']==p])*dt for p in DfF['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfF[DfF['particle']==p])*dt for p in DfF['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfF['radius']),3))+'  std='+str(round(np.std(DfF['radius']),3))+'  NumPart='+str(len(DfF['particle'].unique())))
#    plt.semilogy([GG[1][k] for k in np.arange(0,len(GG[0]),1)],GG[0],'o',label='Needle G : LifeTimeMean='+str(round(np.mean([len(DfG[DfG['particle']==p])*dt for p in DfG['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfG[DfG['particle']==p])*dt for p in DfG['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfG['radius']),3))+'  std='+str(round(np.std(DfG['radius']),3))+'  NumPart='+str(len(DfG['particle'].unique())))
#    plt.semilogy([HH[1][k] for k in np.arange(0,len(HH[0]),1)],HH[0],'o',label='Needle H : LifeTimeMean='+str(round(np.mean([len(DfH[DfH['particle']==p])*dt for p in DfH['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfH[DfH['particle']==p])*dt for p in DfH['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfH['radius']),3))+'  std='+str(round(np.std(DfH['radius']),3))+'  NumPart='+str(len(DfH['particle'].unique())))
#
#    #ln() pour faire l interpolation
#    Ax=AA[1].tolist()
#    Bx=BB[1].tolist()
#    Cx=CC[1].tolist()
#    Dx=DD[1].tolist()
#    Ex=EE[1].tolist()
#    Fx=FF[1].tolist()
#    Gx=GG[1].tolist()
#    Hx=HH[1].tolist()
#    Ay=AA[0].tolist()
#    By=BB[0].tolist()
#    Cy=CC[0].tolist()
#    Dy=DD[0].tolist()
#    Ey=EE[0].tolist()
#    Fy=FF[0].tolist()
#    Gy=GG[0].tolist()
#    Hy=HH[0].tolist()
#    X=[Ax,Bx,Cx,Dx,Ex,Fx,Gx,Hx]
#    Y=[Ay,By,Cy,Dy,Ey,Fy,Gy,Hy]
#    for i in np.arange(0,len(X),1):
#        del X[i][len(X[i])-1]
#    for i in np.arange(0,len(X),1):
#        indices=[]
#        for k in np.arange(0,len(X[i]),1):
#            if Y[i][k]==0:
#                indices.append(k)
#        Y[i]=[Y[i][k] for k in list(set(np.arange(0,len(Y[i]),1))-set(indices))]
#        X[i]=[X[i][k] for k in list(set(np.arange(0,len(X[i]),1))-set(indices))]
#    Ax=X[0]
#    Bx=X[1]
#    Cx=X[2]
#    Dx=X[3]
#    Ex=X[4]
#    Fx=X[5]
#    Gx=X[6]
#    Hx=X[7]
#    Ay=Y[0]
#    By=Y[1]
#    Cy=Y[2]
#    Dy=Y[3]
#    Ey=Y[4]
#    Fy=Y[5]
#    Gy=Y[6]
#    Hy=Y[7]
#    lnAA=[np.log(Ay[k]) for k in np.arange(0,len(Ay),1)]
#    lnBB=[np.log(By[k]) for k in np.arange(0,len(By),1)]
#    lnCC=[np.log(Cy[k]) for k in np.arange(0,len(Cy),1)]
#    lnDD=[np.log(Dy[k]) for k in np.arange(0,len(Dy),1)]
#    lnEE=[np.log(Ey[k]) for k in np.arange(0,len(Ey),1)]
#    lnFF=[np.log(Fy[k]) for k in np.arange(0,len(Fy),1)]
#    lnGG=[np.log(Gy[k]) for k in np.arange(0,len(Gy),1)]
#    lnHH=[np.log(Hy[k]) for k in np.arange(0,len(Hy),1)]
#
#
##LinregressResult(slope=nan, intercept=nan, rvalue=nan, pvalue=nan, stderr=nan)
#    Alr = scipy.stats.linregress(Ax,lnAA)
#    Blr = scipy.stats.linregress(Bx,lnBB)
#    Clr = scipy.stats.linregress(Cx,lnCC)
#    Dlr = scipy.stats.linregress(Dx,lnDD)
#    Elr = scipy.stats.linregress(Ex,lnEE)
#    Flr = scipy.stats.linregress(Fx,lnFF)
#    Glr = scipy.stats.linregress(Gx,lnGG)
#    Hlr = scipy.stats.linregress(Hx,lnHH)
#
#    lop=plt.figure()
#    plt.semilogy(Ax,Ay,'--',linewidth=0.7,color=ColorList[6])
#    plt.semilogy(Bx,By,'--',linewidth=0.7,color=ColorList[14])
#    plt.semilogy(Cx,Cy,'--',linewidth=0.7,color=ColorList[20])
#    plt.semilogy(Dx,Dy,'--',linewidth=0.7,color=ColorList[27])
#    plt.semilogy(Ex,Ey,'--',linewidth=0.7,color=ColorList[35])
#    plt.semilogy(Fx,Fy,'--',linewidth=0.7,color=ColorList[43])
#    plt.semilogy(Gx,Gy,'--',linewidth=0.7,color=ColorList[54])
#    plt.semilogy(Hx,Hy,'--',linewidth=0.7,color=ColorList[64])
#
#    plt.semilogy(Ax,[np.exp(Alr[0]*Ax[k]+Alr[1]) for k in np.arange(0,len(Ax),1)],color=ColorList[6],label='Needle A :  Slope = '+str(round(Alr[0],3))+'  Standard error='+str(round(Alr[4],3))+'  LifeTimeMean='+str(round(np.mean([len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()]),3))+'    RadiusMean='+str(round(np.mean(DfA['radius']),3))+'  std='+str(round(np.std(DfA['radius']),3))+'    NumPart='+str(len(DfA['particle'].unique())))
#    plt.semilogy(Bx,[np.exp(Blr[0]*Bx[k]+Blr[1]) for k in np.arange(0,len(Bx),1)],color=ColorList[14],label='Needle B :  Slope = '+str(round(Blr[0],3))+'  Standard error='+str(round(Blr[4],3))+'   LifeTimeMean='+str(round(np.mean([len(DfB[DfB['particle']==p])*dt for p in DfB['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfB[DfB['particle']==p])*dt for p in DfB['particle'].unique()]),3))+'    RadiusMean='+str(round(np.mean(DfB['radius']),3))+'  std='+str(round(np.std(DfB['radius']),3))+'    NumPart='+str(len(DfB['particle'].unique())))
#    plt.semilogy(Cx,[np.exp(Clr[0]*Cx[k]+Clr[1]) for k in np.arange(0,len(Cx),1)],color=ColorList[20],label='Needle C :  Slope = '+str(round(Clr[0],3))+'  Standard error='+str(round(Clr[4],3))+'   LifeTimeMean='+str(round(np.mean([len(DfC[DfC['particle']==p])*dt for p in DfC['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfC[DfC['particle']==p])*dt for p in DfC['particle'].unique()]),3))+'    RadiusMean='+str(round(np.mean(DfC['radius']),3))+'  std='+str(round(np.std(DfC['radius']),3))+'    NumPart='+str(len(DfC['particle'].unique())))
#    plt.semilogy(Dx,[np.exp(Dlr[0]*Dx[k]+Dlr[1]) for k in np.arange(0,len(Dx),1)],color=ColorList[27],label='Needle D :  Slope = '+str(round(Dlr[0],3))+'  Standard error='+str(round(Dlr[4],3))+'   LifeTimeMean='+str(round(np.mean([len(DfD[DfD['particle']==p])*dt for p in DfD['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfD[DfD['particle']==p])*dt for p in DfD['particle'].unique()]),3))+'    RadiusMean='+str(round(np.mean(DfD['radius']),3))+'  std='+str(round(np.std(DfD['radius']),3))+'    NumPart='+str(len(DfD['particle'].unique())))
#    plt.semilogy(Ex,[np.exp(Elr[0]*Ex[k]+Elr[1]) for k in np.arange(0,len(Ex),1)],color=ColorList[35],label='Needle E :  Slope = '+str(round(Elr[0],3))+'  Standard error='+str(round(Elr[4],3))+'   LifeTimeMean='+str(round(np.mean([len(DfE[DfE['particle']==p])*dt for p in DfE['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfE[DfE['particle']==p])*dt for p in DfE['particle'].unique()]),3))+'    RadiusMean='+str(round(np.mean(DfE['radius']),3))+'  std='+str(round(np.std(DfE['radius']),3))+'    NumPart='+str(len(DfE['particle'].unique())))
#    plt.semilogy(Fx,[np.exp(Flr[0]*Fx[k]+Flr[1]) for k in np.arange(0,len(Fx),1)],color=ColorList[43],label='Needle F :  Slope = '+str(round(Flr[0],3))+'  Standard error='+str(round(Flr[4],3))+'   LifeTimeMean='+str(round(np.mean([len(DfF[DfF['particle']==p])*dt for p in DfF['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfF[DfF['particle']==p])*dt for p in DfF['particle'].unique()]),3))+'    RadiusMean='+str(round(np.mean(DfF['radius']),3))+'  std='+str(round(np.std(DfF['radius']),3))+'    NumPart='+str(len(DfF['particle'].unique())))
#    plt.semilogy(Gx,[np.exp(Glr[0]*Gx[k]+Glr[1]) for k in np.arange(0,len(Gx),1)],color=ColorList[54],label='Needle G :  Slope = '+str(round(Glr[0],3))+'  Standard error='+str(round(Glr[4],3))+'   LifeTimeMean='+str(round(np.mean([len(DfG[DfG['particle']==p])*dt for p in DfG['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfG[DfG['particle']==p])*dt for p in DfG['particle'].unique()]),3))+'    RadiusMean='+str(round(np.mean(DfG['radius']),3))+'  std='+str(round(np.std(DfG['radius']),3))+'    NumPart='+str(len(DfG['particle'].unique())))
#    plt.semilogy(Hx,[np.exp(Hlr[0]*Hx[k]+Hlr[1]) for k in np.arange(0,len(Hx),1)],color=ColorList[64],label='Needle H :  Slope = '+str(round(Hlr[0],3))+'  Standard error='+str(round(Hlr[4],3))+'   LifeTimeMean='+str(round(np.mean([len(DfH[DfH['particle']==p])*dt for p in DfH['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfH[DfH['particle']==p])*dt for p in DfH['particle'].unique()]),3))+'    RadiusMean='+str(round(np.mean(DfH['radius']),3))+'  std='+str(round(np.std(DfH['radius']),3))+'    NumPart='+str(len(DfH['particle'].unique())))
#
#    plt.title('semilogy PDF(t) for each needle and bins = '+str(bins))
#    plt.xlabel('t, Lifetime in s')
#    plt.ylabel('PDF(t)')
#    plt.legend(fontsize=8)
#    
#    
#def plotHistLifeTimeEXPFITNORM(DfA,DfB,DfC,DfD,DfE,DfF,DfG,DfH,bins=15):
#    
#    'Lifetime as a function of the radius mean of each particle for 6 needle'
#    weightsA = np.ones_like([len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()])/float(len([len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()]))
#    weightsB = np.ones_like([len(DfB[DfB['particle']==p])*dt for p in DfB['particle'].unique()])/float(len([len(DfB[DfB['particle']==p])*dt for p in DfB['particle'].unique()]))
#    weightsC = np.ones_like([len(DfC[DfC['particle']==p])*dt for p in DfC['particle'].unique()])/float(len([len(DfC[DfC['particle']==p])*dt for p in DfC['particle'].unique()]))
#    weightsD = np.ones_like([len(DfD[DfD['particle']==p])*dt for p in DfD['particle'].unique()])/float(len([len(DfD[DfD['particle']==p])*dt for p in DfD['particle'].unique()]))
#    weightsE = np.ones_like([len(DfE[DfE['particle']==p])*dt for p in DfE['particle'].unique()])/float(len([len(DfE[DfE['particle']==p])*dt for p in DfE['particle'].unique()]))
#    weightsF = np.ones_like([len(DfF[DfF['particle']==p])*dt for p in DfF['particle'].unique()])/float(len([len(DfF[DfF['particle']==p])*dt for p in DfF['particle'].unique()]))
#    weightsG = np.ones_like([len(DfG[DfG['particle']==p])*dt for p in DfG['particle'].unique()])/float(len([len(DfG[DfG['particle']==p])*dt for p in DfG['particle'].unique()]))
#    weightsH = np.ones_like([len(DfH[DfH['particle']==p])*dt for p in DfH['particle'].unique()])/float(len([len(DfH[DfH['particle']==p])*dt for p in DfH['particle'].unique()]))
#
#    meanA=np.mean([len(DfA[DfA['particle']==p]) for p in DfA['particle'].unique()])*dt
#    meanB=np.mean([len(DfB[DfB['particle']==p]) for p in DfB['particle'].unique()])*dt
#    meanC=np.mean([len(DfC[DfC['particle']==p]) for p in DfC['particle'].unique()])*dt
#    meanD=np.mean([len(DfD[DfD['particle']==p]) for p in DfD['particle'].unique()])*dt
#    meanE=np.mean([len(DfE[DfE['particle']==p]) for p in DfE['particle'].unique()])*dt
#    meanF=np.mean([len(DfF[DfF['particle']==p]) for p in DfF['particle'].unique()])*dt
#    meanG=np.mean([len(DfG[DfG['particle']==p]) for p in DfG['particle'].unique()])*dt
#    meanH=np.mean([len(DfH[DfH['particle']==p]) for p in DfH['particle'].unique()])*dt
#
#
#
#
#
#
#
#
#
#    LTE = plt.figure()
#    AA=plt.hist([len(DfA[DfA['particle']==p])*dt/meanA for p in DfA['particle'].unique()],histtype='step', bins=bins, weights=weightsA,label='Needle A : LifeTimeMean='+str(round(np.mean([len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfA['radius']),3))+'  std='+str(round(np.std(DfA['radius']),3))+'  NumPart='+str(len(DfA['particle'].unique())))
#    BB=plt.hist([len(DfB[DfB['particle']==p])*dt/meanB for p in DfB['particle'].unique()],histtype='step', bins=bins, weights=weightsB,label='Needle B : LifeTimeMean='+str(round(np.mean([len(DfB[DfB['particle']==p])*dt for p in DfB['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfB[DfB['particle']==p])*dt for p in DfB['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfB['radius']),3))+'  std='+str(round(np.std(DfB['radius']),3))+'  NumPart='+str(len(DfB['particle'].unique())))
#    CC=plt.hist([len(DfC[DfC['particle']==p])*dt/meanC for p in DfC['particle'].unique()],histtype='step', bins=bins, weights=weightsC,label='Needle C : LifeTimeMean='+str(round(np.mean([len(DfC[DfC['particle']==p])*dt for p in DfC['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfC[DfC['particle']==p])*dt for p in DfC['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfC['radius']),3))+'  std='+str(round(np.std(DfC['radius']),3))+'  NumPart='+str(len(DfC['particle'].unique())))
#    DD=plt.hist([len(DfD[DfD['particle']==p])*dt/meanD for p in DfD['particle'].unique()],histtype='step', bins=bins, weights=weightsD,label='Needle D : LifeTimeMean='+str(round(np.mean([len(DfD[DfD['particle']==p])*dt for p in DfD['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfD[DfD['particle']==p])*dt for p in DfD['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfD['radius']),3))+'  std='+str(round(np.std(DfD['radius']),3))+'  NumPart='+str(len(DfD['particle'].unique())))
#    EE=plt.hist([len(DfE[DfE['particle']==p])*dt/meanE for p in DfE['particle'].unique()],histtype='step', bins=bins, weights=weightsE,label='Needle E : LifeTimeMean='+str(round(np.mean([len(DfE[DfE['particle']==p])*dt for p in DfE['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfE[DfE['particle']==p])*dt for p in DfE['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfE['radius']),3))+'  std='+str(round(np.std(DfE['radius']),3))+'  NumPart='+str(len(DfE['particle'].unique())))
#    FF=plt.hist([len(DfF[DfF['particle']==p])*dt/meanF for p in DfF['particle'].unique()],histtype='step', bins=bins, weights=weightsF,label='Needle F : LifeTimeMean='+str(round(np.mean([len(DfF[DfF['particle']==p])*dt for p in DfF['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfF[DfF['particle']==p])*dt for p in DfF['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfF['radius']),3))+'  std='+str(round(np.std(DfF['radius']),3))+'  NumPart='+str(len(DfF['particle'].unique())))
#    GG=plt.hist([len(DfG[DfG['particle']==p])*dt/meanG for p in DfG['particle'].unique()],histtype='step', bins=bins, weights=weightsG,label='Needle G : LifeTimeMean='+str(round(np.mean([len(DfG[DfG['particle']==p])*dt for p in DfG['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfG[DfG['particle']==p])*dt for p in DfG['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfG['radius']),3))+'  std='+str(round(np.std(DfG['radius']),3))+'  NumPart='+str(len(DfG['particle'].unique())))
#    HH=plt.hist([len(DfH[DfH['particle']==p])*dt/meanH for p in DfH['particle'].unique()],histtype='step', bins=bins, weights=weightsH,label='Needle H : LifeTimeMean='+str(round(np.mean([len(DfH[DfH['particle']==p])*dt for p in DfH['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfH[DfH['particle']==p])*dt for p in DfH['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfH['radius']),3))+'  std='+str(round(np.std(DfH['radius']),3))+'  NumPart='+str(len(DfH['particle'].unique())))
#    
#    
#    
#    
#    
#    
#    
#    LTE2=plt.figure()
#    plt.semilogy([AA[1][k] for k in np.arange(0,len(AA[0]),1)],AA[0],'o',label='Needle A : LifeTimeMean='+str(round(np.mean([len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfA['radius']),3))+'  std='+str(round(np.std(DfA['radius']),3))+'  NumPart='+str(len(DfA['particle'].unique())))
#    plt.semilogy([BB[1][k] for k in np.arange(0,len(BB[0]),1)],BB[0],'o',label='Needle B : LifeTimeMean='+str(round(np.mean([len(DfB[DfB['particle']==p])*dt for p in DfB['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfB[DfB['particle']==p])*dt for p in DfB['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfB['radius']),3))+'  std='+str(round(np.std(DfB['radius']),3))+'  NumPart='+str(len(DfB['particle'].unique())))
#    plt.semilogy([CC[1][k] for k in np.arange(0,len(CC[0]),1)],CC[0],'o',label='Needle C : LifeTimeMean='+str(round(np.mean([len(DfC[DfC['particle']==p])*dt for p in DfC['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfC[DfC['particle']==p])*dt for p in DfC['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfC['radius']),3))+'  std='+str(round(np.std(DfC['radius']),3))+'  NumPart='+str(len(DfC['particle'].unique())))
#    plt.semilogy([DD[1][k] for k in np.arange(0,len(DD[0]),1)],DD[0],'o',label='Needle D : LifeTimeMean='+str(round(np.mean([len(DfD[DfD['particle']==p])*dt for p in DfD['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfD[DfD['particle']==p])*dt for p in DfD['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfD['radius']),3))+'  std='+str(round(np.std(DfD['radius']),3))+'  NumPart='+str(len(DfD['particle'].unique())))
#    plt.semilogy([EE[1][k] for k in np.arange(0,len(EE[0]),1)],EE[0],'o',label='Needle E : LifeTimeMean='+str(round(np.mean([len(DfE[DfE['particle']==p])*dt for p in DfE['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfE[DfE['particle']==p])*dt for p in DfE['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfE['radius']),3))+'  std='+str(round(np.std(DfE['radius']),3))+'  NumPart='+str(len(DfE['particle'].unique())))
#    plt.semilogy([FF[1][k] for k in np.arange(0,len(FF[0]),1)],FF[0],'o',label='Needle F : LifeTimeMean='+str(round(np.mean([len(DfF[DfF['particle']==p])*dt for p in DfF['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfF[DfF['particle']==p])*dt for p in DfF['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfF['radius']),3))+'  std='+str(round(np.std(DfF['radius']),3))+'  NumPart='+str(len(DfF['particle'].unique())))
#    plt.semilogy([GG[1][k] for k in np.arange(0,len(GG[0]),1)],GG[0],'o',label='Needle G : LifeTimeMean='+str(round(np.mean([len(DfG[DfG['particle']==p])*dt for p in DfG['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfG[DfG['particle']==p])*dt for p in DfG['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfG['radius']),3))+'  std='+str(round(np.std(DfG['radius']),3))+'  NumPart='+str(len(DfG['particle'].unique())))
#    plt.semilogy([HH[1][k] for k in np.arange(0,len(HH[0]),1)],HH[0],'o',label='Needle H : LifeTimeMean='+str(round(np.mean([len(DfH[DfH['particle']==p])*dt for p in DfH['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfH[DfH['particle']==p])*dt for p in DfH['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfH['radius']),3))+'  std='+str(round(np.std(DfH['radius']),3))+'  NumPart='+str(len(DfH['particle'].unique())))

#    #ln() pour faire l interpolation
#    Ax=AA[1].tolist()
#    Bx=BB[1].tolist()
#    Cx=CC[1].tolist()
#    Dx=DD[1].tolist()
#    Ex=EE[1].tolist()
#    Fx=FF[1].tolist()
#    Gx=GG[1].tolist()
#    Hx=HH[1].tolist()
#    Ay=AA[0].tolist()
#    By=BB[0].tolist()
#    Cy=CC[0].tolist()
#    Dy=DD[0].tolist()
#    Ey=EE[0].tolist()
#    Fy=FF[0].tolist()
#    Gy=GG[0].tolist()
#    Hy=HH[0].tolist()
#    X=[Ax,Bx,Cx,Dx,Ex,Fx,Gx,Hx]
#    Y=[Ay,By,Cy,Dy,Ey,Fy,Gy,Hy]
#    for i in np.arange(0,len(X),1):
#        del X[i][len(X[i])-1]
#    for i in np.arange(0,len(X),1):
#        indices=[]
#        for k in np.arange(0,len(X[i]),1):
#            if Y[i][k]<0.01:
#                indices.append(k)
#        Y[i]=[Y[i][k] for k in list(set(np.arange(0,len(Y[i]),1))-set(indices))]
#        X[i]=[X[i][k] for k in list(set(np.arange(0,len(X[i]),1))-set(indices))]
#    Ax=X[0]
#    Bx=X[1]
#    Cx=X[2]
#    Dx=X[3]
#    Ex=X[4]
#    Fx=X[5]
#    Gx=X[6]
#    Hx=X[7]
#    Ay=Y[0]
#    By=Y[1]
#    Cy=Y[2]
#    Dy=Y[3]
#    Ey=Y[4]
#    Fy=Y[5]
#    Gy=Y[6]
#    Hy=Y[7]
#    lnAA=[np.log(Ay[k]) for k in np.arange(0,len(Ay),1)]
#    lnBB=[np.log(By[k]) for k in np.arange(0,len(By),1)]
#    lnCC=[np.log(Cy[k]) for k in np.arange(0,len(Cy),1)]
#    lnDD=[np.log(Dy[k]) for k in np.arange(0,len(Dy),1)]
#    lnEE=[np.log(Ey[k]) for k in np.arange(0,len(Ey),1)]
#    lnFF=[np.log(Fy[k]) for k in np.arange(0,len(Fy),1)]
#    lnGG=[np.log(Gy[k]) for k in np.arange(0,len(Gy),1)]
#    lnHH=[np.log(Hy[k]) for k in np.arange(0,len(Hy),1)]
#
#
##LinregressResult(slope=nan, intercept=nan, rvalue=nan, pvalue=nan, stderr=nan)
#    Alr = scipy.stats.linregress(Ax,lnAA)
#    Blr = scipy.stats.linregress(Bx,lnBB)
#    Clr = scipy.stats.linregress(Cx,lnCC)
#    Dlr = scipy.stats.linregress(Dx,lnDD)
#    Elr = scipy.stats.linregress(Ex,lnEE)
#    Flr = scipy.stats.linregress(Fx,lnFF)
#    Glr = scipy.stats.linregress(Gx,lnGG)
#    Hlr = scipy.stats.linregress(Hx,lnHH)
#
#    lop=plt.figure()
#    plt.semilogy(Ax,Ay,'--',linewidth=0.7,color=ColorList[6])
#    plt.semilogy(Bx,By,'--',linewidth=0.7,color=ColorList[14])
#    plt.semilogy(Cx,Cy,'--',linewidth=0.7,color=ColorList[20])
#    plt.semilogy(Dx,Dy,'--',linewidth=0.7,color=ColorList[27])
#    plt.semilogy(Ex,Ey,'--',linewidth=0.7,color=ColorList[35])
#    plt.semilogy(Fx,Fy,'--',linewidth=0.7,color=ColorList[43])
#    plt.semilogy(Gx,Gy,'--',linewidth=0.7,color=ColorList[54])
#    plt.semilogy(Hx,Hy,'--',linewidth=0.7,color=ColorList[64])
#
##    plt.semilogy(Ax,[np.exp(Alr[0]*Ax[k]+Alr[1]) for k in np.arange(0,len(Ax),1)],color=ColorList[6],label='Needle A :  Slope = '+str(round(Alr[0],3))+'  Standard error='+str(round(Alr[4],3))+'  LifeTimeMean='+str(round(np.mean([len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()]),3))+'    RadiusMean='+str(round(np.mean(DfA['radius']),3))+'  std='+str(round(np.std(DfA['radius']),3))+'    NumPart='+str(len(DfA['particle'].unique())))
##    plt.semilogy(Bx,[np.exp(Blr[0]*Bx[k]+Blr[1]) for k in np.arange(0,len(Bx),1)],color=ColorList[14],label='Needle B :  Slope = '+str(round(Blr[0],3))+'  Standard error='+str(round(Blr[4],3))+'   LifeTimeMean='+str(round(np.mean([len(DfB[DfB['particle']==p])*dt for p in DfB['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfB[DfB['particle']==p])*dt for p in DfB['particle'].unique()]),3))+'    RadiusMean='+str(round(np.mean(DfB['radius']),3))+'  std='+str(round(np.std(DfB['radius']),3))+'    NumPart='+str(len(DfB['particle'].unique())))
##    plt.semilogy(Cx,[np.exp(Clr[0]*Cx[k]+Clr[1]) for k in np.arange(0,len(Cx),1)],color=ColorList[20],label='Needle C :  Slope = '+str(round(Clr[0],3))+'  Standard error='+str(round(Clr[4],3))+'   LifeTimeMean='+str(round(np.mean([len(DfC[DfC['particle']==p])*dt for p in DfC['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfC[DfC['particle']==p])*dt for p in DfC['particle'].unique()]),3))+'    RadiusMean='+str(round(np.mean(DfC['radius']),3))+'  std='+str(round(np.std(DfC['radius']),3))+'    NumPart='+str(len(DfC['particle'].unique())))
##    plt.semilogy(Dx,[np.exp(Dlr[0]*Dx[k]+Dlr[1]) for k in np.arange(0,len(Dx),1)],color=ColorList[27],label='Needle D :  Slope = '+str(round(Dlr[0],3))+'  Standard error='+str(round(Dlr[4],3))+'   LifeTimeMean='+str(round(np.mean([len(DfD[DfD['particle']==p])*dt for p in DfD['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfD[DfD['particle']==p])*dt for p in DfD['particle'].unique()]),3))+'    RadiusMean='+str(round(np.mean(DfD['radius']),3))+'  std='+str(round(np.std(DfD['radius']),3))+'    NumPart='+str(len(DfD['particle'].unique())))
##    plt.semilogy(Ex,[np.exp(Elr[0]*Ex[k]+Elr[1]) for k in np.arange(0,len(Ex),1)],color=ColorList[35],label='Needle E :  Slope = '+str(round(Elr[0],3))+'  Standard error='+str(round(Elr[4],3))+'   LifeTimeMean='+str(round(np.mean([len(DfE[DfE['particle']==p])*dt for p in DfE['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfE[DfE['particle']==p])*dt for p in DfE['particle'].unique()]),3))+'    RadiusMean='+str(round(np.mean(DfE['radius']),3))+'  std='+str(round(np.std(DfE['radius']),3))+'    NumPart='+str(len(DfE['particle'].unique())))
##    plt.semilogy(Fx,[np.exp(Flr[0]*Fx[k]+Flr[1]) for k in np.arange(0,len(Fx),1)],color=ColorList[43],label='Needle F :  Slope = '+str(round(Flr[0],3))+'  Standard error='+str(round(Flr[4],3))+'   LifeTimeMean='+str(round(np.mean([len(DfF[DfF['particle']==p])*dt for p in DfF['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfF[DfF['particle']==p])*dt for p in DfF['particle'].unique()]),3))+'    RadiusMean='+str(round(np.mean(DfF['radius']),3))+'  std='+str(round(np.std(DfF['radius']),3))+'    NumPart='+str(len(DfF['particle'].unique())))
##    plt.semilogy(Gx,[np.exp(Glr[0]*Gx[k]+Glr[1]) for k in np.arange(0,len(Gx),1)],color=ColorList[54],label='Needle G :  Slope = '+str(round(Glr[0],3))+'  Standard error='+str(round(Glr[4],3))+'   LifeTimeMean='+str(round(np.mean([len(DfG[DfG['particle']==p])*dt for p in DfG['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfG[DfG['particle']==p])*dt for p in DfG['particle'].unique()]),3))+'    RadiusMean='+str(round(np.mean(DfG['radius']),3))+'  std='+str(round(np.std(DfG['radius']),3))+'    NumPart='+str(len(DfG['particle'].unique())))
##    plt.semilogy(Hx,[np.exp(Hlr[0]*Hx[k]+Hlr[1]) for k in np.arange(0,len(Hx),1)],color=ColorList[64],label='Needle H :  Slope = '+str(round(Hlr[0],3))+'  Standard error='+str(round(Hlr[4],3))+'   LifeTimeMean='+str(round(np.mean([len(DfH[DfH['particle']==p])*dt for p in DfH['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfH[DfH['particle']==p])*dt for p in DfH['particle'].unique()]),3))+'    RadiusMean='+str(round(np.mean(DfH['radius']),3))+'  std='+str(round(np.std(DfH['radius']),3))+'    NumPart='+str(len(DfH['particle'].unique())))
#
#    plt.title('semilogy PDF(t/<t>) for each needle and bins = '+str(bins))
#    plt.xlabel('t/<t>, normalized lifetime with <t> the mean lifetime')
#    plt.ylabel('PDF(t/<t>)')
#    plt.legend(fontsize=8)
#
#
#
#def plotHistLifeTimewithFIT(DfA,DfB,DfC,DfD,DfE,DfF,DfG,DfH,bins=15):
#    
#    'Lifetime as a function of the radius mean of each particle for 6 needle'
#    weightsA = np.ones_like([len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()])/float(len([len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()]))
#    weightsB = np.ones_like([len(DfB[DfB['particle']==p])*dt for p in DfB['particle'].unique()])/float(len([len(DfB[DfB['particle']==p])*dt for p in DfB['particle'].unique()]))
#    weightsC = np.ones_like([len(DfC[DfC['particle']==p])*dt for p in DfC['particle'].unique()])/float(len([len(DfC[DfC['particle']==p])*dt for p in DfC['particle'].unique()]))
#    weightsD = np.ones_like([len(DfD[DfD['particle']==p])*dt for p in DfD['particle'].unique()])/float(len([len(DfD[DfD['particle']==p])*dt for p in DfD['particle'].unique()]))
#    weightsE = np.ones_like([len(DfE[DfE['particle']==p])*dt for p in DfE['particle'].unique()])/float(len([len(DfE[DfE['particle']==p])*dt for p in DfE['particle'].unique()]))
#    weightsF = np.ones_like([len(DfF[DfF['particle']==p])*dt for p in DfF['particle'].unique()])/float(len([len(DfF[DfF['particle']==p])*dt for p in DfF['particle'].unique()]))
#    weightsG = np.ones_like([len(DfG[DfG['particle']==p])*dt for p in DfG['particle'].unique()])/float(len([len(DfG[DfG['particle']==p])*dt for p in DfG['particle'].unique()]))
#    weightsH = np.ones_like([len(DfH[DfH['particle']==p])*dt for p in DfH['particle'].unique()])/float(len([len(DfH[DfH['particle']==p])*dt for p in DfH['particle'].unique()]))
#
#    LTE = plt.figure()
#    AA=plt.hist([len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()],histtype='step', bins=bins, weights=weightsA,label='Needle A : LifeTimeMean='+str(round(np.mean([len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfA['radius']),3))+'  std='+str(round(np.std(DfA['radius']),3))+'  NumPart='+str(len(DfA['particle'].unique())))
#    BB=plt.hist([len(DfB[DfB['particle']==p])*dt for p in DfB['particle'].unique()],histtype='step', bins=bins, weights=weightsB,label='Needle B : LifeTimeMean='+str(round(np.mean([len(DfB[DfB['particle']==p])*dt for p in DfB['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfB[DfB['particle']==p])*dt for p in DfB['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfB['radius']),3))+'  std='+str(round(np.std(DfB['radius']),3))+'  NumPart='+str(len(DfB['particle'].unique())))
#    CC=plt.hist([len(DfC[DfC['particle']==p])*dt for p in DfC['particle'].unique()],histtype='step', bins=bins, weights=weightsC,label='Needle C : LifeTimeMean='+str(round(np.mean([len(DfC[DfC['particle']==p])*dt for p in DfC['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfC[DfC['particle']==p])*dt for p in DfC['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfC['radius']),3))+'  std='+str(round(np.std(DfC['radius']),3))+'  NumPart='+str(len(DfC['particle'].unique())))
#    DD=plt.hist([len(DfD[DfD['particle']==p])*dt for p in DfD['particle'].unique()],histtype='step', bins=bins, weights=weightsD,label='Needle D : LifeTimeMean='+str(round(np.mean([len(DfD[DfD['particle']==p])*dt for p in DfD['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfD[DfD['particle']==p])*dt for p in DfD['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfD['radius']),3))+'  std='+str(round(np.std(DfD['radius']),3))+'  NumPart='+str(len(DfD['particle'].unique())))
#    EE=plt.hist([len(DfE[DfE['particle']==p])*dt for p in DfE['particle'].unique()],histtype='step', bins=bins, weights=weightsE,label='Needle E : LifeTimeMean='+str(round(np.mean([len(DfE[DfE['particle']==p])*dt for p in DfE['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfE[DfE['particle']==p])*dt for p in DfE['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfE['radius']),3))+'  std='+str(round(np.std(DfE['radius']),3))+'  NumPart='+str(len(DfE['particle'].unique())))
#    FF=plt.hist([len(DfF[DfF['particle']==p])*dt for p in DfF['particle'].unique()],histtype='step', bins=bins, weights=weightsF,label='Needle F : LifeTimeMean='+str(round(np.mean([len(DfF[DfF['particle']==p])*dt for p in DfF['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfF[DfF['particle']==p])*dt for p in DfF['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfF['radius']),3))+'  std='+str(round(np.std(DfF['radius']),3))+'  NumPart='+str(len(DfF['particle'].unique())))
#    GG=plt.hist([len(DfG[DfG['particle']==p])*dt for p in DfG['particle'].unique()],histtype='step', bins=bins, weights=weightsG,label='Needle G : LifeTimeMean='+str(round(np.mean([len(DfG[DfG['particle']==p])*dt for p in DfG['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfG[DfG['particle']==p])*dt for p in DfG['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfG['radius']),3))+'  std='+str(round(np.std(DfG['radius']),3))+'  NumPart='+str(len(DfG['particle'].unique())))
#    HH=plt.hist([len(DfH[DfH['particle']==p])*dt for p in DfH['particle'].unique()],histtype='step', bins=bins, weights=weightsH,label='Needle H : LifeTimeMean='+str(round(np.mean([len(DfH[DfH['particle']==p])*dt for p in DfH['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfH[DfH['particle']==p])*dt for p in DfH['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfH['radius']),3))+'  std='+str(round(np.std(DfH['radius']),3))+'  NumPart='+str(len(DfH['particle'].unique())))
#
#    plt.plot([AA[1][k] for k in np.arange(0,len(AA[0]),1)],AA[0],'o',color=ColorList[6],label='Needle A : LifeTimeMean='+str(round(np.mean([len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfA['radius']),3))+'  std='+str(round(np.std(DfA['radius']),3))+'  NumPart='+str(len(DfA['particle'].unique())))
#    plt.plot([BB[1][k] for k in np.arange(0,len(BB[0]),1)],BB[0],'o',color=ColorList[14],label='Needle B : LifeTimeMean='+str(round(np.mean([len(DfB[DfB['particle']==p])*dt for p in DfB['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfB[DfB['particle']==p])*dt for p in DfB['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfB['radius']),3))+'  std='+str(round(np.std(DfB['radius']),3))+'  NumPart='+str(len(DfB['particle'].unique())))
#    plt.plot([CC[1][k] for k in np.arange(0,len(CC[0]),1)],CC[0],'o',color=ColorList[20],label='Needle C : LifeTimeMean='+str(round(np.mean([len(DfC[DfC['particle']==p])*dt for p in DfC['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfC[DfC['particle']==p])*dt for p in DfC['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfC['radius']),3))+'  std='+str(round(np.std(DfC['radius']),3))+'  NumPart='+str(len(DfC['particle'].unique())))
#    plt.plot([DD[1][k] for k in np.arange(0,len(DD[0]),1)],DD[0],'o',color=ColorList[27],label='Needle D : LifeTimeMean='+str(round(np.mean([len(DfD[DfD['particle']==p])*dt for p in DfD['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfD[DfD['particle']==p])*dt for p in DfD['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfD['radius']),3))+'  std='+str(round(np.std(DfD['radius']),3))+'  NumPart='+str(len(DfD['particle'].unique())))
#    plt.plot([EE[1][k] for k in np.arange(0,len(EE[0]),1)],EE[0],'o',color=ColorList[35],label='Needle E : LifeTimeMean='+str(round(np.mean([len(DfE[DfE['particle']==p])*dt for p in DfE['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfE[DfE['particle']==p])*dt for p in DfE['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfE['radius']),3))+'  std='+str(round(np.std(DfE['radius']),3))+'  NumPart='+str(len(DfE['particle'].unique())))
#    plt.plot([FF[1][k] for k in np.arange(0,len(FF[0]),1)],FF[0],'o',color=ColorList[43],label='Needle F : LifeTimeMean='+str(round(np.mean([len(DfF[DfF['particle']==p])*dt for p in DfF['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfF[DfF['particle']==p])*dt for p in DfF['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfF['radius']),3))+'  std='+str(round(np.std(DfF['radius']),3))+'  NumPart='+str(len(DfF['particle'].unique())))
#    plt.plot([GG[1][k] for k in np.arange(0,len(GG[0]),1)],GG[0],'o',color=ColorList[54],label='Needle G : LifeTimeMean='+str(round(np.mean([len(DfG[DfG['particle']==p])*dt for p in DfG['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfG[DfG['particle']==p])*dt for p in DfG['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfG['radius']),3))+'  std='+str(round(np.std(DfG['radius']),3))+'  NumPart='+str(len(DfG['particle'].unique())))
#    plt.plot([HH[1][k] for k in np.arange(0,len(HH[0]),1)],HH[0],'o',color=ColorList[64],label='Needle H : LifeTimeMean='+str(round(np.mean([len(DfH[DfH['particle']==p])*dt for p in DfH['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfH[DfH['particle']==p])*dt for p in DfH['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfH['radius']),3))+'  std='+str(round(np.std(DfH['radius']),3))+'  NumPart='+str(len(DfH['particle'].unique())))
#
#    plt.plot([AA[1][k] for k in np.arange(0,len(AA[0]),1)],AA[0],'-',color=ColorList[6],linewidth=0.7,label='Needle A : LifeTimeMean='+str(round(np.mean([len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfA['radius']),3))+'  std='+str(round(np.std(DfA['radius']),3))+'  NumPart='+str(len(DfA['particle'].unique())))
#    plt.plot([BB[1][k] for k in np.arange(0,len(BB[0]),1)],BB[0],'-',color=ColorList[14],linewidth=0.7,label='Needle B : LifeTimeMean='+str(round(np.mean([len(DfB[DfB['particle']==p])*dt for p in DfB['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfB[DfB['particle']==p])*dt for p in DfB['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfB['radius']),3))+'  std='+str(round(np.std(DfB['radius']),3))+'  NumPart='+str(len(DfB['particle'].unique())))
#    plt.plot([CC[1][k] for k in np.arange(0,len(CC[0]),1)],CC[0],'-',color=ColorList[20],linewidth=0.7,label='Needle C : LifeTimeMean='+str(round(np.mean([len(DfC[DfC['particle']==p])*dt for p in DfC['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfC[DfC['particle']==p])*dt for p in DfC['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfC['radius']),3))+'  std='+str(round(np.std(DfC['radius']),3))+'  NumPart='+str(len(DfC['particle'].unique())))
#    plt.plot([DD[1][k] for k in np.arange(0,len(DD[0]),1)],DD[0],'-',color=ColorList[27],linewidth=0.7,label='Needle D : LifeTimeMean='+str(round(np.mean([len(DfD[DfD['particle']==p])*dt for p in DfD['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfD[DfD['particle']==p])*dt for p in DfD['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfD['radius']),3))+'  std='+str(round(np.std(DfD['radius']),3))+'  NumPart='+str(len(DfD['particle'].unique())))
#    plt.plot([EE[1][k] for k in np.arange(0,len(EE[0]),1)],EE[0],'-',color=ColorList[35],linewidth=0.7,label='Needle E : LifeTimeMean='+str(round(np.mean([len(DfE[DfE['particle']==p])*dt for p in DfE['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfE[DfE['particle']==p])*dt for p in DfE['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfE['radius']),3))+'  std='+str(round(np.std(DfE['radius']),3))+'  NumPart='+str(len(DfE['particle'].unique())))
#    plt.plot([FF[1][k] for k in np.arange(0,len(FF[0]),1)],FF[0],'-',color=ColorList[43],linewidth=0.7,label='Needle F : LifeTimeMean='+str(round(np.mean([len(DfF[DfF['particle']==p])*dt for p in DfF['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfF[DfF['particle']==p])*dt for p in DfF['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfF['radius']),3))+'  std='+str(round(np.std(DfF['radius']),3))+'  NumPart='+str(len(DfF['particle'].unique())))
#    plt.plot([GG[1][k] for k in np.arange(0,len(GG[0]),1)],GG[0],'-',color=ColorList[54],linewidth=0.7,label='Needle G : LifeTimeMean='+str(round(np.mean([len(DfG[DfG['particle']==p])*dt for p in DfG['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfG[DfG['particle']==p])*dt for p in DfG['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfG['radius']),3))+'  std='+str(round(np.std(DfG['radius']),3))+'  NumPart='+str(len(DfG['particle'].unique())))
#    plt.plot([HH[1][k] for k in np.arange(0,len(HH[0]),1)],HH[0],'-',color=ColorList[64],linewidth=0.7,label='Needle H : LifeTimeMean='+str(round(np.mean([len(DfH[DfH['particle']==p])*dt for p in DfH['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfH[DfH['particle']==p])*dt for p in DfH['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfH['radius']),3))+'  std='+str(round(np.std(DfH['radius']),3))+'  NumPart='+str(len(DfH['particle'].unique())))
#
#    plt.title('Histogramme Lifetime for each particle Needle and bins = '+str(bins))
#    plt.xlabel('Lifetime in s')
#    plt.ylabel('freqence')
#    plt.legend()
#    
#    Ax=AA[1].tolist()
#    Bx=BB[1].tolist()
#    Cx=CC[1].tolist()
#    Dx=DD[1].tolist()
#    Ex=EE[1].tolist()
#    Fx=FF[1].tolist()
#    Gx=GG[1].tolist()
#    Hx=HH[1].tolist()
#    
#    Ay=AA[0].tolist()
#    By=BB[0].tolist()
#    Cy=CC[0].tolist()
#    Dy=DD[0].tolist()
#    Ey=EE[0].tolist()
#    Fy=FF[0].tolist()
#    Gy=GG[0].tolist()
#    Hy=HH[0].tolist()
#    X=[Ax,Bx,Cx,Dx,Ex,Fx,Gx,Hx]
#    Y=[Ay,By,Cy,Dy,Ey,Fy,Gy,Hy]
#    
#    for i in np.arange(0,len(X),1):
#        del X[i][len(X[i])-1]
#    
#    
#    for i in np.arange(0,len(X),1):
#        indices=[]
#        for k in np.arange(0,len(X[i]),1):
#            if Y[i][k]==0:
#                indices.append(k)
#        Y[i]=[Y[i][k] for k in list(set(np.arange(0,len(Y[i]),1))-set(indices))]
#        X[i]=[X[i][k] for k in list(set(np.arange(0,len(X[i]),1))-set(indices))]
#
#    Ax=X[0]
#    Bx=X[1]
#    Cx=X[2]
#    Dx=X[3]
#    Ex=X[4]
#    Fx=X[5]
#    Gx=X[6]
#    Hx=X[7]
#    
#    Ay=Y[0]
#    By=Y[1]
#    Cy=Y[2]
#    Dy=Y[3]
#    Ey=Y[4]
#    Fy=Y[5]
#    Gy=Y[6]
#    Hy=Y[7]
#    
#    lnAA=[np.log(Ay[k])*np.mean([len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()]) for k in np.arange(0,len(Ay),1)]
#    lnBB=[np.log(By[k])*np.mean([len(DfB[DfB['particle']==p])*dt for p in DfB['particle'].unique()]) for k in np.arange(0,len(By),1)]
#    lnCC=[np.log(Cy[k])*np.mean([len(DfC[DfC['particle']==p])*dt for p in DfC['particle'].unique()]) for k in np.arange(0,len(Cy),1)]
#    lnDD=[np.log(Dy[k])*np.mean([len(DfD[DfD['particle']==p])*dt for p in DfD['particle'].unique()]) for k in np.arange(0,len(Dy),1)]
#    lnEE=[np.log(Ey[k])*np.mean([len(DfE[DfE['particle']==p])*dt for p in DfE['particle'].unique()]) for k in np.arange(0,len(Ey),1)]
#    lnFF=[np.log(Fy[k])*np.mean([len(DfF[DfF['particle']==p])*dt for p in DfF['particle'].unique()]) for k in np.arange(0,len(Fy),1)]
#    lnGG=[np.log(Gy[k])*np.mean([len(DfG[DfG['particle']==p])*dt for p in DfG['particle'].unique()]) for k in np.arange(0,len(Gy),1)]
#    lnHH=[np.log(Hy[k])*np.mean([len(DfH[DfH['particle']==p])*dt for p in DfH['particle'].unique()]) for k in np.arange(0,len(Hy),1)]
#
#
##LinregressResult(slope=nan, intercept=nan, rvalue=nan, pvalue=nan, stderr=nan)
#    Alr = scipy.stats.linregress(Ax,lnAA)
#    Blr = scipy.stats.linregress(Bx,lnBB)
#    Clr = scipy.stats.linregress(Cx,lnCC)
#    Dlr = scipy.stats.linregress(Dx,lnDD)
#    Elr = scipy.stats.linregress(Ex,lnEE)
#    Flr = scipy.stats.linregress(Fx,lnFF)
#    Glr = scipy.stats.linregress(Gx,lnGG)
#    Hlr = scipy.stats.linregress(Hx,lnHH)
#
#    Ln = plt.figure()
#    plt.plot(Ax,lnAA,'--',linewidth=0.7,color=ColorList[6],label='Needle A : Slope = '+str(round(Alr[0],3))+'Standard error='+str(round(Alr[4],3))+'LifeTimeMean='+str(round(np.mean([len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfA['radius']),3))+'  std='+str(round(np.std(DfA['radius']),3))+'  NumPart='+str(len(DfA['particle'].unique())))
#    plt.plot(Bx,lnBB,'--',linewidth=0.7,color=ColorList[14],label='Needle B : Slope = '+str(round(Blr[0],3))+'Standard error='+str(round(Blr[4],3))+' LifeTimeMean='+str(round(np.mean([len(DfB[DfB['particle']==p])*dt for p in DfB['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfB[DfB['particle']==p])*dt for p in DfB['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfB['radius']),3))+'  std='+str(round(np.std(DfB['radius']),3))+'  NumPart='+str(len(DfB['particle'].unique())))
#    plt.plot(Cx,lnCC,'--',linewidth=0.7,color=ColorList[20],label='Needle C : Slope = '+str(round(Clr[0],3))+'Standard error='+str(round(Clr[4],3))+' LifeTimeMean='+str(round(np.mean([len(DfC[DfC['particle']==p])*dt for p in DfC['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfC[DfC['particle']==p])*dt for p in DfC['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfC['radius']),3))+'  std='+str(round(np.std(DfC['radius']),3))+'  NumPart='+str(len(DfC['particle'].unique())))
#    plt.plot(Dx,lnDD,'--',linewidth=0.7,color=ColorList[27],label='Needle D : Slope = '+str(round(Dlr[0],3))+'Standard error='+str(round(Dlr[4],3))+' LifeTimeMean='+str(round(np.mean([len(DfD[DfD['particle']==p])*dt for p in DfD['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfD[DfD['particle']==p])*dt for p in DfD['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfD['radius']),3))+'  std='+str(round(np.std(DfD['radius']),3))+'  NumPart='+str(len(DfD['particle'].unique())))
#    plt.plot(Ex,lnEE,'--',linewidth=0.7,color=ColorList[35],label='Needle E : Slope = '+str(round(Elr[0],3))+'Standard error='+str(round(Elr[4],3))+' LifeTimeMean='+str(round(np.mean([len(DfE[DfE['particle']==p])*dt for p in DfE['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfE[DfE['particle']==p])*dt for p in DfE['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfE['radius']),3))+'  std='+str(round(np.std(DfE['radius']),3))+'  NumPart='+str(len(DfE['particle'].unique())))
#    plt.plot(Fx,lnFF,'--',linewidth=0.7,color=ColorList[43],label='Needle F : Slope = '+str(round(Flr[0],3))+'Standard error='+str(round(Flr[4],3))+' LifeTimeMean='+str(round(np.mean([len(DfF[DfF['particle']==p])*dt for p in DfF['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfF[DfF['particle']==p])*dt for p in DfF['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfF['radius']),3))+'  std='+str(round(np.std(DfF['radius']),3))+'  NumPart='+str(len(DfF['particle'].unique())))
#    plt.plot(Gx,lnGG,'--',linewidth=0.7,color=ColorList[54],label='Needle G : Slope = '+str(round(Glr[0],3))+'Standard error='+str(round(Glr[4],3))+' LifeTimeMean='+str(round(np.mean([len(DfG[DfG['particle']==p])*dt for p in DfG['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfG[DfG['particle']==p])*dt for p in DfG['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfG['radius']),3))+'  std='+str(round(np.std(DfG['radius']),3))+'  NumPart='+str(len(DfG['particle'].unique())))
#    plt.plot(Hx,lnHH,'--',linewidth=0.7,color=ColorList[64],label='Needle H : Slope = '+str(round(Hlr[0],3))+'Standard error='+str(round(Hlr[4],3))+' LifeTimeMean='+str(round(np.mean([len(DfH[DfH['particle']==p])*dt for p in DfH['particle'].unique()]),3))+'   std='+str(round(np.std([len(DfH[DfH['particle']==p])*dt for p in DfH['particle'].unique()]),3))+'  RadiusMean='+str(round(np.mean(DfH['radius']),3))+'  std='+str(round(np.std(DfH['radius']),3))+'  NumPart='+str(len(DfH['particle'].unique())))
#
#    plt.plot(Ax,[Alr[0]*Ax[k]+Alr[1] for k in np.arange(0,len(Ax),1)],color=ColorList[6])
#    plt.plot(Bx,[Blr[0]*Bx[k]+Blr[1] for k in np.arange(0,len(Bx),1)],color=ColorList[14])
#    plt.plot(Cx,[Clr[0]*Cx[k]+Clr[1] for k in np.arange(0,len(Cx),1)],color=ColorList[20])
#    plt.plot(Dx,[Dlr[0]*Dx[k]+Dlr[1] for k in np.arange(0,len(Dx),1)],color=ColorList[27])
#    plt.plot(Ex,[Elr[0]*Ex[k]+Elr[1] for k in np.arange(0,len(Ex),1)],color=ColorList[35])
#    plt.plot(Fx,[Flr[0]*Fx[k]+Flr[1] for k in np.arange(0,len(Fx),1)],color=ColorList[43])
#    plt.plot(Gx,[Glr[0]*Gx[k]+Glr[1] for k in np.arange(0,len(Gx),1)],color=ColorList[54])
#    plt.plot(Hx,[Hlr[0]*Hx[k]+Hlr[1] for k in np.arange(0,len(Hx),1)],color=ColorList[64])
#
#    
#
#    Ww=[[0,7],[7,15],[15,21],[21,28],[28,36],[36,44],[44,55],[55,65]]
#
#
#
#
#
#
#    plt.title('LifeTimeMean * ln(PDF(Lifetime)) Needle')
#    plt.xlabel('Lifetime in s')
#    plt.ylabel('Lifetime Mean * ln(PDF)')
#    plt.legend(fontsize=8)
#
#
#
#
##a voir
#
#
#
#def plotLifeTimefctRmeanperPone(Df00):
#    'Lifetime as a function of the radius mean of each particle'
#    LT = plt.figure()
#    plt.plot([np.mean(DfA[DfA['particle']==p]['radius']) for p in DfA['particle'].unique()],[len(DfA[DfA['particle']==p])*dt for p in DfA['particle'].unique()],'.',label='Needle01')
#    plt.title('Lifetime as a function of the radius mean of each particle Needle ')
#    plt.xlabel('Mean Radius of each particle in mm')
#    plt.ylabel('Life time of each particle in s')
#    plt.legend()
#    
#def plotRfctTperP(Df, numframe=50000, style='+'):
#    'Plot radius fct of time per particle'
#    plt.title('Plot radius fct of time per particle')
#    plt.xlabel('frame')
#    plt.ylabel('radius in mm')
#    Df_all=Df[Df['frame']<numframe]
#    for p in Df_all['particle'].unique():
#        plt.plot(Df_all[Df_all['particle']==p]['frame'],Df_all[Df_all['particle']==p]['radius'],style,label=p)
#    
#def Histogramme2DRadius(DfA,DfB,DfC,DfD,DfE,DfF,DfG,DfH):
#    DfALL=pd.concat([DfA,DfB,DfC,DfD,DfE,DfF,DfG,DfH])
#    plt.figure()
#    plt.hist2d(np.asarray(DfALL['time'],dtype='float64'),np.asarray(DfALL['radius'],dtype='float64'),bins=(60, 60))
#
#def HistogrammeRadius(DfA,DfB,DfC,DfD,DfE,DfF,DfG,DfH):
#    DfALL=pd.concat([DfA/len(DfA),DfB/len(DfB),DfC/len(DfC),DfD/len(DfD),DfE/len(DfE),DfF/len(DfF),DfG/len(DfG),DfH/len(DfH)])
#    plt.figure()
#    plt.hist(np.asarray(DfALL['radius'],dtype='float64'),bins=30)
#    
##    'Velocity'
##    vel,axs = plt.subplots(2,2,figsize=(17,9),sharex=True,sharey=False)
##    axs = axs.flatten()
##    datas = [U[1][U[1]['num_of_bubble']==i][U[1]['U']<0.5] for i in sorted(U[1]['num_of_bubble'].unique())]
##    names = ['Mean velocity of cluster in clusters of '+str(i)+' bubbles as fct of Mean radius' for i in sorted(U[1]['num_of_bubble'].unique())]
##    for i,(d,name) in enumerate(zip(datas,names)):
##        trackedmean=pd.DataFrame(index=d['particle'].unique(),columns=d.columns)
##        for p in d['particle'].unique():
##            for col in ('radius','U'):
##                trackedmean[col][p]=d[d['particle']==p].mean(axis=0)[col]
##        axs[i].plot(d['radius'],d['U'],'bo')
##        axs[i].set_title(name)
##        axs[i].set_xlabel('radius')
##        axs[i].set_ylabel('U')
#    
#    return data
#
#
#
#def radiusbubble(df_bubble,df_cluster):
#    df_cluster['list_rad_bu']=df_cluster['bubble']
#    for c in df_cluster['frame'].index:
#        f=df_cluster.loc[c]['frame']
#        
#        
#def VelocityRadius(DfA,DfB,DfC,DfD,DfE,DfF):
#    'Velocity as a fct of radius'
#    LT = plt.figure()
#    plt.plot(DfA['radius'],DfA['U'],'*',label='NeedleOrange')
#    plt.plot(DfB['radius'],DfB['U'],'.',label='Needle01')
#    plt.plot(DfC['radius'],DfC['U'],'x',label='Needle02')
#    plt.plot(DfD['radius'],DfD['U'],'<',label='Needle03')
#    plt.plot(DfE['radius'],DfE['U'],'d',label='Needle04')
#    plt.plot(DfF['radius'],DfF['U'],'s',label='Needle0Bigone')
#
#
#
#    plt.title('Mean Velocity as a function of the mean radius')
#    plt.xlabel('Radius')
#    plt.ylabel('Velocity')
#    plt.legend()
#
#def verticalVelocityRadius(DfA,DfB,DfC,DfD,DfE,DfF):
#    'vertical Velocity as a fct of radius'
#    LT = plt.figure()
#    plt.plot(DfA['radius'],DfA['Uy'],',',label='NeedleOrange')
#    plt.plot(DfB['radius'],DfB['Uy'],',',label='Needle01')
#    plt.plot(DfC['radius'],DfC['Uy'],',',label='Needle02')
#    plt.plot(DfD['radius'],DfD['Uy'],',',label='Needle03')
#    plt.plot(DfE['radius'],DfE['Uy'],',',label='Needle04')
#    plt.plot(DfF['radius'],DfF['Uy'],',',label='Needle0Bigone')
#
#
#
#    plt.title('Mean vertical Velocity as a function of the mean radius')
#    plt.xlabel('Radius')
#    plt.ylabel('Velocity')
#    plt.legend()
#
#def xVelocityRadius(DfA,DfB,DfC,DfD,DfE,DfF):
#    'vertical Velocity as a fct of radius'
#    LT = plt.figure()
#    plt.plot(DfA['radius'],DfA['Ux'],'*',label='NeedleOrange')
#    plt.plot(DfB['radius'],DfB['Ux'],'.',label='Needle01')
#    plt.plot(DfC['radius'],DfC['Ux'],'x',label='Needle02')
#    plt.plot(DfD['radius'],DfD['Ux'],'<',label='Needle03')
#    plt.plot(DfE['radius'],DfE['Ux'],'d',label='Needle04')
#    plt.plot(DfF['radius'],DfF['Ux'],'s',label='Needle0Bigone')
#    plt.title('Mean vertical Velocity as a function of the mean radius')
#    plt.xlabel('Radius')
#    plt.ylabel('Velocity')
#    plt.legend()
#
#'''
#plt.figure()
#for df in Df:
#    for p in df['particle'].unique():
#        plt.plot([np.mean(df[df['particle']==p]['radius'])],[np.mean(df[df['particle']==p]['Uy'])],',',color='r')
d=[0.5,0.6,0.7,0.8,0.9,1,1.1,1.3,1.4,2,3,4,6]
r=[d[i]/2.0 for i in np.arange(0,len(d),1)]
v=[60,70,85,115,150,180,220,340,323,290,276,257,238]
plt.plot(r,v,'-',color='b',label='Maxworthy and Al.')
plt.legend()
#plt.title('Vertical Velocity as a function of the Radius')
#plt.xlabel('Radius of each Bubble in mm')
#plt.ylabel('Vertical Velocity in mm/s')

plt.figure()

red = Color("red")
colors = list(red.range_to(Color("green"),280))

for i in np.arange(0,len(VRD),1):
    plt.plot([VRD['Radius'].tolist()[i]],[VRD['Velocity'].tolist()[i]],',',color=colors[int(VRD['Ecart'].tolist()[i])])
    print(i)
#tracked=tp.link_df(df_filtered,search_range=80,memory=3)
#tracked_filtred=tp.filter_stubs(df, threshold=15)
#
#Df_Afill=tp.filter_stubs(Df_Afi, threshold=8)
#Df_Bfill=tp.filter_stubs(Df_Bfi, threshold=8)
#Df_Cfill=tp.filter_stubs(Df_Cfi, threshold=8)
#Df_Dfill=tp.filter_stubs(Df_Dfi, threshold=8)
#Df_Efill=tp.filter_stubs(Df_Efi, threshold=8)
#Df_Ffill=tp.filter_stubs(Df_Ffi, threshold=8)
#Df_Gfill=tp.filter_stubs(Df_Gfi, threshold=8)
#Df_Hfill=tp.filter_stubs(Df_Hfi, threshold=8)
#'''
#def RadiusperPart(Df):
#    J = plt.figure()
#    Ax = J.add_subplot(111)
#    for p in Df['particle'].unique():
#        Ax.plot(Df[Df['particle']==p]['time'],Df[Df['particle']==p]['radius'])
#
#def LenperPArt(Df):
#    TY = plt.figure()
#    plt.plot(np.asarray(Df['particle'].unique()),np.asarray([len(Df[Df['particle']==p]) for p in Df['particle'].unique()]),'.')
#
#def YfctT(Df):
#    J = plt.figure()
#    Ax = J.add_subplot(111)
#    for p in Df['particle'].unique():
#        Ax.plot(Df[Df['particle']==p]['time'],Df[Df['particle']==p]['y'])
#    
#def YfctTNEW(Df):
#    J = plt.figure()
#    Ax = J.add_subplot(111)
#    for p in Df['particle'].unique():
#        Ax.plot(Df[Df['particle']==p]['radius'].mean(),((Df[Df['particle']==p]['y'].max()-Df[Df['particle']==p]['y'].min())/(Df[Df['particle']==p]['time'].max()-Df[Df['particle']==p]['time'].min())),'o')
#    
#def ArrayU0o(Df):
#    A=[]
#    B=[]
#    for p in Df['particle'].unique():
#        if ((Df[Df['particle']==p]['y'].max()-Df[Df['particle']==p]['y'].min())/(Df[Df['particle']==p]['time'].max()-Df[Df['particle']==p]['time'].min()))<300:
#            if ((Df[Df['particle']==p]['y'].max()-Df[Df['particle']==p]['y'].min())/(Df[Df['particle']==p]['time'].max()-Df[Df['particle']==p]['time'].min()))>200:
#                A.append(Df[Df['particle']==p]['radius'].mean())
#                B.append(((Df[Df['particle']==p]['y'].max()-Df[Df['particle']==p]['y'].min())/(Df[Df['particle']==p]['time'].max()-Df[Df['particle']==p]['time'].min())))
#    return [A,B]
#
#def REGVelocityRadius(DfA,DfB,DfC,DfD,DfE,DfF):
#    'vertical Velocity as a fct of radius'
#    LT = plt.figure()
#    plt.plot(ArrayU(DfA)[0],ArrayU(DfA)[1],'*',label='NeedleOrange')
#    plt.plot(ArrayU0o(DfB)[0],ArrayU0o(DfB)[1],'.',label='Needle01')
#    plt.plot(ArrayU(DfC)[0],ArrayU(DfC)[1],'x',label='Needle02')
#    plt.plot(ArrayU(DfD)[0],ArrayU(DfD)[1],'<',label='Needle03')
##    plt.plot(ArrayU(DfE)[0],ArrayU(DfE)[1],'d',label='Needle04')
#    plt.plot(ArrayU(DfF)[0],ArrayU(DfF)[1],'s',label='Needle0Bigone')
#    plt.title('REGRESSIOn Mean vertical Velocity as a function of the mean radius')
#    plt.xlabel('Radius in mm')
#    plt.ylabel('Velocity in mm per sec')
#    plt.legend()
#
#def histogrammeRadius(Df,ax,needle):
#    A=[Df[Df['particle']==p]['radius'].mean() for p in Df['particle'].unique()]
#    ax.hist(A,label=needle+'mean='+str(np.mean(A))+'stand.dev='+str(np.std(A)),normed=True)
#    
#def negatif(x):
#    if x<=0 :
#        return 
#    elif x>0.6:
#        return
#    else :
#        return x
#
#def traj(Df,needle='00'):
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    tp.plot_traj(Df,ax=ax,colorby='particle')
#    plt.title('Trajectories of the '+str(len(Df['particle'].unique()))+' bubbles of needle '+str(needle))
#    plt.xlabel('x en mm')
#    plt.ylabel('y en mm')
#    plt.figure()
#    plt.hist([len(Df[Df['particle']==p]) for p in Df['particle'].unique()],histtype='step',label='Needle0'+str(needle),bins=36)
##ax.invert_yaxis()
#
#def tri(Df,min,max):
#    A=[]
#    for p in Df['particle'].unique():
#        if len(Df[Df['particle']==p])<min:
#            Df=Df[Df['particle']!=p]
#        if len(Df[Df['particle']==p])>max:
#            Df=Df[Df['particle']!=p]
#
#    return Df
#
#
#
#def Dfmean(Df):
#    trackedmean=pd.DataFrame(index=Df['particle'].unique(),columns=Df.columns)
#    for p in Df['particle'].unique():
#        for col in Df.columns:
#            trackedmean[col][p]=Df[Df['particle']==p].mean(axis=0)[col]
#    return trackedmean
    
'''
list_of_bubbles = [tracked[tracked['particle']==p] for p in tracked['particle'].unique()]

#################PLOTS####################################

fig = plt.figure()
plt.title('y fct x')
ax = fig.add_subplot(111)
for bubble_df in list_of_bubbles:
    ax.plot(bubble_df['x'],bubble_df['y'])

'''


#####################################################################################################################
'''
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

DfA=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\5_22_18\trackedDf_A.xlsx')
DfB=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\5_22_18\trackedDf_B.xlsx')
DfC=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\5_22_18\trackedDf_C.xlsx')
DfD=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\5_22_18\trackedDf_D.xlsx')
DfE=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\5_22_18\trackedDf_E.xlsx')
DfF=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\5_22_18\trackedDf_F.xlsx')
DfG=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\5_22_18\trackedDf_G.xlsx')
DfH=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\5_22_18\trackedDf_H.xlsx')



Df_Gfi.to_excel(r'C:\Users\Luc Deike\Nicolas\StudySurface\5_24_18\Df_Gfi.xlsx')
Df_G.to_excel(r'C:\Users\Luc Deike\Nicolas\StudySurface\5_24_18\Df_G.xlsx')


DfA0200.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfA0200.xlsx')
DfA0100.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfA0100.xlsx')
DfA0080.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfA0080.xlsx')
DfA0070.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfA0070.xlsx')
DfA0060.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfA0060.xlsx')
DfA0050.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfA0050.xlsx')
DfA0030.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfA0030.xlsx')

DfB1000.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfB1000.xlsx')
DfB0300.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfB0300.xlsx')
DfB0100.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfB0100.xlsx')
DfB0070.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfB0070.xlsx')
DfB0065.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfB0065.xlsx')
DfB0060.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfB0060.xlsx')
DfB0050.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfB0050.xlsx')
DfB0035.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfB0035.xlsx')

DfC1500.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfC1500.xlsx')
DfC0500.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfC0500.xlsx')
DfC0300.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfC0300.xlsx')
DfC0100.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfC0100.xlsx')
DfC0070.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfC0070.xlsx')
DfC0020.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfC0020.xlsx')

DfD2000.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfD2000.xlsx')
DFD1500.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfD1500.xlsx')
DfD1000.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfD1000.xlsx')
DfD0400.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfD0400.xlsx')
DfD0200.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfD0200.xlsx')
DfD0100.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfD0100.xlsx')
DfD0060.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfD0060.xlsx')
DfD0040.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfD0040.xlsx')

DfE2000.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfE2000.xlsx')
DfE2500.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfE2500.xlsx')
DfE1500.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfE1500.xlsx')
DfE1000.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfE1000.xlsx')
DfE0700.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfE0700.xlsx')
DfE0500.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfE0500.xlsx')
DfE0300.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfE0300.xlsx')
DfE0200.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfE0200.xlsx')
DfE0100.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfE0100.xlsx')

DfF3500.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfF3500.xlsx')
DfF3000.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfF3000.xlsx')
DfF2500.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfF2500.xlsx')
DfF2000.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfF2000.xlsx')
DfF1500.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfF1500.xlsx')
DfF1000.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfF1000.xlsx')
DfF0700.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfF0700.xlsx')
DfF0500.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfF0500.xlsx')

DfG5000.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfG5000.xlsx')
DfG4500.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfG4500.xlsx')
DfG4000.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfG4000.xlsx')
DfG3500.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfG3500.xlsx')
DfG3000.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfG3000.xlsx')
DfG2500.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfG2500.xlsx')
DfG2000.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfG2000.xlsx')
DfG1500.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfG1500.xlsx')
DfG1000.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfG1000.xlsx')
DfG0800.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfG0800.xlsx')
DfG0500.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfG0500.xlsx')

DfH5000.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfH5000.xlsx')
DfH4500.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfH4500.xlsx')
DfH4000.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfH4000.xlsx')
DfH3500.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfH3500.xlsx')
DfH3000.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfH3000.xlsx')
DfH2500.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfH2500.xlsx')
DfH2000.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfH2000.xlsx')
DfH1500.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfH1500.xlsx')
DfH1000.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfH1000.xlsx')
DfH0500.to_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfH0500.xlsx')


DfA0200=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfA0200.xlsx')
DfA0100=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfA0100.xlsx')
DfA0080=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfA0080.xlsx')
DfA0070=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfA0070.xlsx')
DfA0060=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfA0060.xlsx')
DfA0050=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfA0050.xlsx')
DfA0030=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfA0030.xlsx')

DfB1000=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfB1000.xlsx')
DfB0300=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfB0300.xlsx')
DfB0100=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfB0100.xlsx')
DfB0070=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfB0070.xlsx')
DfB0065=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfB0065.xlsx')
DfB0060=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfB0060.xlsx')
DfB0050=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfB0050.xlsx')
DfB0035=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfB0035.xlsx')

DfC1500=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfC1500.xlsx')
DfC0500=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfC0500.xlsx')
DfC0300=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfC0300.xlsx')
DfC0100=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfC0100.xlsx')
DfC0070=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfC0070.xlsx')
DfC0020=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfC0020.xlsx')

DfD2000=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfD2000.xlsx')
DFD1500=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfD1500.xlsx')
DfD1000=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfD1000.xlsx')
DfD0400=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfD0400.xlsx')
DfD0200=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfD0200.xlsx')
DfD0100=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfD0100.xlsx')
DfD0060=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfD0060.xlsx')
DfD0040=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfD0040.xlsx')

#DfE2000=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfE2000.xlsx')
DfE2500=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfE2500.xlsx')
DfE1500=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfE1500.xlsx')
DfE1000=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfE1000.xlsx')
DfE0700=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfE0700.xlsx')
DfE0500=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfE0500.xlsx')
DfE0300=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfE0300.xlsx')
DfE0200=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfE0200.xlsx')
DfE0100=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfE0100.xlsx')

DfF3500=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfF3500.xlsx')
DfF3000=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfF3000.xlsx')
DfF2500=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfF2500.xlsx')
DfF2000=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfF2000.xlsx')
DfF1500=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfF1500.xlsx')
DfF1000=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfF1000.xlsx')
DfF0700=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfF0700.xlsx')
DfF0500=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfF0500.xlsx')

DfG5000=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfG5000.xlsx')
DfG4500=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfG4500.xlsx')
DfG4000=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfG4000.xlsx')
DfG3500=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfG3500.xlsx')
DfG3000=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfG3000.xlsx')
DfG2500=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfG2500.xlsx')
DfG2000=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfG2000.xlsx')
DfG1500=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfG1500.xlsx')
DfG1000=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfG1000.xlsx')
DfG0800=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfG0800.xlsx')
DfG0500=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfG0500.xlsx')

DfH5000=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfH5000.xlsx')
DfH4500=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfH4500.xlsx')
DfH4000=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfH4000.xlsx')
DfH3500=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfH3500.xlsx')
DfH3000=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfH3000.xlsx')
DfH2500=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfH2500.xlsx')
DfH2000=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfH2000.xlsx')
DfH1500=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfH1500.xlsx')
DfH1000=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfH1000.xlsx')
DfH0500=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\Traitement\DfH0500.xlsx')

###################################STUDY RISE FOLLOZING BUBBLE

H5000 = pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleH\H5000_P2170_T2907_Tlab203Hum52_100fps_70exptime.cine')
H4500 = pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleH\H4500_P2170_T2907_Tlab203Hum52_100fps_70exptime.cine')
H4000 = pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleH\H4000_P2170_T2907_Tlab203Hum52_100fps_70exptime.cine')
H3500 = pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleH\H3500_P2170_T2907_Tlab203Hum52_100fps_70exptime.cine')
H3000 = pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleH\H3000_P2170_T2907_Tlab203Hum52_100fps_70exptime.cine')
H2500 = pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleH\H2500_P2170_T2907_Tlab203Hum52_100fps_70exptime.cine')
H2000 = pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleH\H2000_P2170_T2907_Tlab203Hum52_100fps_70exptime.cine')
H1500 = pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleH\H1500_P2070_T2907_Tlab217Hum49_100fps_70exptime.cine')
H1000 = pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleH\H1000_P2070_T2907_Tlab217Hum49_100fps_70exptime.cine')
H0500 = pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleH\H0500_P2070_T2907_Tlab217Hum49_100fps_70exptime.cine')

DfH5000=traitement(H5000,thresh,g,np.arange(0,len(H5000),1),method='standard',Radius=Radius[7],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfH4500=traitement(H4500,thresh,g,np.arange(0,len(H4500),1),method='standard',Radius=Radius[7],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfH4000=traitement(H4000,thresh,g,np.arange(0,len(H4000),1),method='standard',Radius=Radius[7],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfH3500=traitement(H3500,thresh,g,np.arange(0,len(H3500),1),method='standard',Radius=Radius[7],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfH3000=traitement(H3000,thresh,g,np.arange(0,len(H3000),1),method='standard',Radius=Radius[7],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfH2500=traitement(H2500,thresh,g,np.arange(0,len(H2500),1),method='standard',Radius=Radius[7],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfH2000=traitement(H2000,thresh,g,np.arange(0,len(H2000),1),method='standard',Radius=Radius[7],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfH1500=traitement(H1500,thresh,g,np.arange(0,len(H1500),1),method='standard',Radius=Radius[7],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfH1000=traitement(H1000,thresh,g,np.arange(0,len(H1000),1),method='standard',Radius=Radius[7],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfH0500=traitement(H0500,thresh,g,np.arange(0,len(H0500),1),method='standard',Radius=Radius[7],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)


#NeedleG
G5000 = pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleG\G5000_P2145_T2607_Tlab219Hum49_100fps_70exptime.cine')
G4500 = pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleG\G4500_P2145_T2707_Tlab223Hum49_100fps_70exptime.cine')
G4000 = pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleG\G4000_P2145_T2707_Tlab225Hum49_100fps_70exptime.cine')
G3500 = pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleG\G3500_P2145_T2707_Tlab225Hum49_100fps_70exptime.cine')
G3000 = pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleG\G3000_P2145_T2707_Tlab225Hum49_100fps_70exptime.cine')
G2500 = pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleG\G2500_P2145_T2707_Tlab225Hum49_100fps_70exptime.cine')
G2000 = pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleG\G2000_P2145_T2707_Tlab225Hum49_100fps_70exptime.cine')
G1500 = pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleG\G1500_P2145_T2707_Tlab225Hum49_100fps_70exptime.cine')
G1000 = pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleG\G1000_P2145_T2707_Tlab225Hum49_100fps_70exptime.cine')
G0800 = pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleG\G0800_P2145_T2707_Tlab225Hum49_100fps_70exptime.cine')
G0500 = pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleG\G0500_P2145_T2707_Tlab225Hum49_100fps_70exptime.cine')

DfG5000=traitement(G5000,thresh,g,np.arange(0,len(G5000),1),method='standard',Radius=Radius[6],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfG4500=traitement(G4500,thresh,g,np.arange(0,len(G4500),1),method='standard',Radius=Radius[6],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfG4000=traitement(G4000,thresh,g,np.arange(0,len(G4000),1),method='standard',Radius=Radius[6],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfG3500=traitement(G3500,thresh,g,np.arange(0,len(G3500),1),method='standard',Radius=Radius[6],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfG3000=traitement(G3000,thresh,g,np.arange(0,len(G3000),1),method='standard',Radius=Radius[6],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfG2500=traitement(G2500,thresh,g,np.arange(0,len(G2500),1),method='standard',Radius=Radius[6],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfG2000=traitement(G2000,thresh,g,np.arange(0,len(G2000),1),method='standard',Radius=Radius[6],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfG1500=traitement(G1500,thresh,g,np.arange(0,len(G1500),1),method='standard',Radius=Radius[6],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfG1000=traitement(G1000,thresh,g,np.arange(0,len(G1000),1),method='standard',Radius=Radius[6],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfG0800=traitement(G0800,thresh,g,np.arange(0,len(G0800),1),method='standard',Radius=Radius[6],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfG0500=traitement(G0500,thresh,g,np.arange(0,len(G0500),1),method='standard',Radius=Radius[6],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)

#NeedleF
F3500= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleF\F3500_P2145_T2707_Tlab226Hum49_100fps_70exptime.cine')
F3000= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleF\F3000_P2145_T2707_Tlab226Hum49_100fps_70exptime.cine')
F2500= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleF\F2500_P2145_T2707_Tlab226Hum49_100fps_70exptime.cine')
F2000= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleF\F2000_P2145_T2707_Tlab226Hum49_100fps_70exptime.cine')
F1500= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleF\F1500_P2145_T2707_Tlab226Hum49_100fps_70exptime.cine')
F1000= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleF\F1000_P2145_T2707_Tlab226Hum49_100fps_70exptime.cine')
F0700= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleF\F0700_P2145_T2707_Tlab226Hum49_100fps_70exptime.cine')
F0500= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleF\F0500_P2145_T2707_Tlab226Hum49_100fps_70exptime.cine')

DfF3500= traitement(F3500,thresh,g,np.arange(0,len(F3500),1),method='standard',Radius=Radius[5],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfF3000= traitement(F3000,thresh,g,np.arange(0,len(F3000),1),method='standard',Radius=Radius[5],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfF2500= traitement(F2500,thresh,g,np.arange(0,len(F2500),1),method='standard',Radius=Radius[5],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfF2000= traitement(F2000,thresh,g,np.arange(0,len(F2000),1),method='standard',Radius=Radius[5],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfF1500= traitement(F1500,thresh,g,np.arange(0,len(F1500),1),method='standard',Radius=Radius[5],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfF1000= traitement(F1000,thresh,g,np.arange(0,len(F1000),1),method='standard',Radius=Radius[5],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfF0700= traitement(F0700,thresh,g,np.arange(0,len(F0700),1),method='standard',Radius=Radius[5],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfF0500= traitement(F0500,thresh,g,np.arange(0,len(F0500),1),method='standard',Radius=Radius[5],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)

#NeedleE
E2500= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleE\E2500_P2145_T2707_Tlab226Hum49_100fps_70exptime.cine')
E2000= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleE\E2000_P2145_T2707_Tlab226Hum49_100fps_70exptime.cine')
E1500= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleE\E1500_P2145_T2707_Tlab226Hum49_100fps_70exptime.cine')
E1000= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleE\E1000_P2145_T2707_Tlab226Hum49_100fps_70exptime.cine')
E0700= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleE\E0700_P2145_T2707_Tlab226Hum49_100fps_70exptime.cine')
E0500= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleE\E0500_P2145_T2707_Tlab226Hum49_100fps_70exptime.cine')
E0300= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleE\E0300_P2145_T2707_Tlab226Hum49_100fps_70exptime.cine')
E0200= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleE\E0200_P2145_T2707_Tlab226Hum49_100fps_70exptime.cine')
E0100= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleE\E0100_P2145_T2707_Tlab226Hum49_100fps_70exptime.cine')

DfE2500= traitement(E2500,thresh,g,np.arange(0,len(E2500),1),method='standard',Radius=Radius[4],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DFE2000= traitement(E2000,thresh,g,np.arange(0,len(E2000),1),method='standard',Radius=Radius[4],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfE1500= traitement(E1500,thresh,g,np.arange(0,len(E1500),1),method='standard',Radius=Radius[4],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfE1000= traitement(E1000,thresh,g,np.arange(0,len(E1000),1),method='standard',Radius=Radius[4],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfE0700= traitement(E0700,thresh,g,np.arange(0,len(E0700),1),method='standard',Radius=Radius[4],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfE0500= traitement(E0500,thresh,g,np.arange(0,len(E0500),1),method='standard',Radius=Radius[4],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfE0300= traitement(E0300,thresh,g,np.arange(0,len(E0300),1),method='standard',Radius=Radius[4],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfE0200= traitement(E0200,thresh,g,np.arange(0,len(E0200),1),method='standard',Radius=Radius[4],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfE0100= traitement(E0100,thresh,g,np.arange(0,len(E0100),1),method='standard',Radius=Radius[4],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)

#NeedleD
D2000= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleD\D2000_P1999_T2988_Tlab234Hum47_100fps_70exptime.cine')
D1500= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleD\D1500_P1999_T2988_Tlab234Hum47_100fps_70exptime.cine')
D1000= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleD\D1000_P1999_T2988_Tlab234Hum47_100fps_70exptime.cine')
D0400= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleD\D0400_P1999_T2988_Tlab234Hum47_100fps_70exptime.cine')
D0200= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleD\D0200_P1999_T2988_Tlab234Hum47_100fps_70exptime.cine')
D0100= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleD\D0100_P1999_T2988_Tlab234Hum47_100fps_70exptime.cine')
D0060= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleD\D0060_P1999_T2988_Tlab234Hum47_100fps_70exptime.cine')
D0040= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleD\D0040_P1999_T2988_Tlab234Hum47_100fps_70exptime.cine')

DfD2000= traitement(D2000,thresh,g,np.arange(0,len(D2000),1),method='standard',Radius=Radius[3],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DFD1500= traitement(D1500,thresh,g,np.arange(0,len(D1500),1),method='standard',Radius=Radius[3],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfD1000= traitement(D1000,thresh,g,np.arange(0,len(D1000),1),method='standard',Radius=Radius[3],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfD0400= traitement(D0400,thresh,g,np.arange(0,len(D0400),1),method='standard',Radius=Radius[3],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfD0200= traitement(D0200,thresh,g,np.arange(0,len(D0200),1),method='standard',Radius=Radius[3],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfD0100= traitement(D0100,thresh,g,np.arange(0,len(D0100),1),method='standard',Radius=Radius[3],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfD0060= traitement(D0060,thresh,g,np.arange(0,len(D0060),1),method='standard',Radius=Radius[3],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfD0040= traitement(D0040,thresh,g,np.arange(0,len(D0040),1),method='standard',Radius=Radius[3],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)

#NeedleC
C1500= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleC\C1500_P1999_T2988_Tlab236Hum47_100fps_70exptime.cine')
C0500= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleC\C0500_P1999_T2988_Tlab236Hum47_100fps_70exptime.cine')
C0300= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleC\C0300_P1999_T2988_Tlab236Hum47_100fps_70exptime.cine')
C0100= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleC\C0100_P1999_T2988_Tlab236Hum47_100fps_70exptime.cine')
C0070= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleC\C0070_P1999_T2988_Tlab236Hum47_100fps_70exptime.cine')
C0020= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleC\C0020_P1999_T2988_Tlab236Hum47_100fps_70exptime.cine')

DfC1500= traitement(C1500,thresh,g,np.arange(0,len(C1500),1),method='standard',Radius=Radius[2],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfC0500= traitement(C0500,thresh,g,np.arange(0,len(C0500),1),method='standard',Radius=Radius[2],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfC0300= traitement(C0300,thresh,g,np.arange(0,len(C0300),1),method='standard',Radius=Radius[2],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfC0100= traitement(C0100,thresh,g,np.arange(0,len(C0100),1),method='standard',Radius=Radius[2],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfC0070= traitement(C0070,thresh,g,np.arange(0,len(C0070),1),method='standard',Radius=Radius[2],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfC0020= traitement(C0020,thresh,g,np.arange(0,len(C0020),1),method='standard',Radius=Radius[2],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)

#NeedleB
B1000= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleB\B1000_P1999_T2988_Tlab236Hum47_100fps_70exptime.cine')
B0300= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleB\B0300_P1999_T2988_Tlab236Hum47_100fps_70exptime.cine')
B0100= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleB\B0100_P1999_T2988_Tlab236Hum47_100fps_70exptime.cine')
B0070= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleB\B0070_P1999_T2988_Tlab236Hum47_100fps_70exptime.cine')
B0065= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleB\B0065_P1999_T2988_Tlab236Hum47_100fps_70exptime.cine')
B0060= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleB\B0060_P1999_T2988_Tlab236Hum47_100fps_70exptime.cine')
B0050= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleB\B0050_P1999_T2988_Tlab236Hum47_100fps_70exptime.cine')
B0035= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleB\B0035_P1999_T2988_Tlab236Hum47_100fps_70exptime.cine')

DfB1000= traitement(B1000,thresh,g,np.arange(0,len(B1000),1),method='standard',Radius=Radius[1],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfB0300= traitement(B0300,thresh,g,np.arange(0,len(B0300),1),method='standard',Radius=Radius[1],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfB0100= traitement(B0100,thresh,g,np.arange(0,len(B0100),1),method='standard',Radius=Radius[1],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfB0070= traitement(B0070,thresh,g,np.arange(0,len(B0070),1),method='standard',Radius=Radius[1],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfB0065= traitement(B0065,thresh,g,np.arange(0,len(B0065),1),method='standard',Radius=Radius[1],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfB0060= traitement(B0060,thresh,g,np.arange(0,len(B0060),1),method='standard',Radius=Radius[1],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfB0050= traitement(B0050,thresh,g,np.arange(0,len(B0050),1),method='standard',Radius=Radius[1],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfB0035= traitement(B0035,thresh,g,np.arange(0,len(B0035),1),method='standard',Radius=Radius[1],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)

#NeedleA
A0200= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleA\A0200_P1999_T2988_Tlab236Hum47_100fps_70exptime.cine')
A0100= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleA\A0100_P1999_T2988_Tlab236Hum47_100fps_70exptime.cine')
A0080= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleA\A0080_P1999_T2988_Tlab236Hum47_100fps_70exptime.cine')
A0070= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleA\A0070_P1999_T2988_Tlab236Hum47_100fps_70exptime.cine')
A0060= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleA\A0060_P1999_T2988_Tlab236Hum47_100fps_70exptime.cine')
A0050= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleA\A0050_P1999_T2988_Tlab236Hum47_100fps_70exptime.cine')
A0030= pims.open(r'C:\Users\Luc Deike\Nicolas\StudyRaise\30_5_18_following_bubbles\NeedleA\A0030_P1999_T2988_Tlab236Hum47_100fps_70exptime.cine')

DfA0200= traitement(A0200,thresh,g,np.arange(0,len(A0200),1),method='standard',Radius=Radius[0],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfA0100= traitement(A0100,thresh,g,np.arange(0,len(A0100),1),method='standard',Radius=Radius[0],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfA0080= traitement(A0080,thresh,g,np.arange(0,len(A0080),1),method='standard',Radius=Radius[0],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfA0070= traitement(A0070,thresh,g,np.arange(0,len(A0070),1),method='standard',Radius=Radius[0],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfA0060= traitement(A0060,thresh,g,np.arange(0,len(A0060),1),method='standard',Radius=Radius[0],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfA0050= traitement(A0050,thresh,g,np.arange(0,len(A0050),1),method='standard',Radius=Radius[0],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)
DfA0030= traitement(A0030,thresh,g,np.arange(0,len(A0030),1),method='standard',Radius=Radius[0],MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=RadiusDiskMean)

A=[[DfA0030,DfA0050,DfA0060,DfA0070,DfA0080,DfA0100,DfA0200],
   [DfB0035,DfB0050,DfB0060,DfB0065,DfB0070,DfB0100,DfB0300,DfB1000],
   [DfC0020,DfC0070,DfC0100,DfC0300,DfC0500,DfC1500],
   [DfD0040,DfD0060,DfD0100,DfD0200,DfD0400,DfD1000,DFD1500,DfD2000],
   [DfE0100,DfE0200,DfE0300,DfE0500,DfE0700,DfE1000,DfE1500,DfE2500],
   [DfF0500,DfF0700,DfF1000,DfF1500,DfF2000,DfF2500,DfF3000,DfF3500],
   [DfG0500,DfG0800,DfG1000,DfG1500,DfG2000,DfG2500,DfG3000,DfG3500,DfG4000,DfG4500,DfG5000],
   [DfH0500,DfH1000,DfH1500,DfH2000,DfH2500,DfH3000,DfH3500,DfH4000,DfH4500,DfH5000]]

Color=[['#330000','#660000','#990000','#CC0000','#FF0000','#FF3333','#FF6666','#FF9999','#FFCCCC'],
['#331900','#663300','#994C00','#CC6600','#FF8000','#FF9933','#FFB266','#FFCC99','#FFE5CC'],
['#333300','#666600','#999900','#CCCC00','#FFFF00','#FFFF33','#FFFF66','#FFFF99','#FFFFCC'],
['#193300','#336600','#4C9900','#66CC00','#80FF00','#99FF33','#B2FF66','#CCFF99','#E5FFCC'],
['#003333','#006666','#009999','#00CCCC','#00FFFF','#33FFFF','#66FFFF','#99FFFF','#CCFFFF'],
['#001933','#003366','#004C99','#0066CC','#0080FF','#3399FF','#66B2FF','#99CCFF','#CCE5FF'],
['#190033','#330066','#4C0099','#6600CC','#7F00FF','#9933FF','#B266FF','#CC99FF','#E5CCFF','#FFCCFF','#FFE5FF'],
['#330019','#660033','#99004C','#CC0066','#FF007F','#FF3399','#FF66B2','#FF99CC','#FFCCE5','#FFDFEF']]

DfA=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudySurface\5_24_18\Df_Afi.xlsx')
DfB=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudySurface\5_24_18\Df_Bfi.xlsx')
DfC=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudySurface\5_24_18\Df_Cfi.xlsx')
DfD=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudySurface\5_24_18\Df_Dfi.xlsx')
DfE=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudySurface\5_24_18\Df_Efi.xlsx')
DfF=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudySurface\5_24_18\Df_Ffi.xlsx')
DfG=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudySurface\5_24_18\Df_Gfi.xlsx')
DfH=pd.read_excel(r'C:\Users\Luc Deike\Nicolas\StudySurface\5_24_18\Df_Hfi.xlsx')




ListDataF=[DfA0200,
DfA0100,
DfA0080,
DfA0070,
DfA0060,
DfA0050,
DfA0030,
DfB1000,
DfB0300,
DfB0100,
DfB0070,
DfB0065,
DfB0060,
DfB0050,
DfB0035,
DfC1500,
DfC0500,
DfC0300,
DfC0100,
DfC0070,
DfC0020,
DfD2000,
DfD1000,
DfD0400,
DfD0200,
DfD0100,
DfD0060,
DfD0040,
DfE2500,
DfE1500,
DfE1000,
DfE0700,
DfE0500,
DfE0300,
DfE0200,
DfE0100,
DfF3500,
DfF3000,
DfF2500,
DfF2000,
DfF1500,
DfF1000,
DfF0700,
DfF0500,
DfG5000,
DfG4500,
DfG4000,
DfG3500,
DfG3000,
DfG2500,
DfG2000,
DfG1500,
DfG1000,
DfG0800,
DfG0500,
DfH5000,
DfH4500,
DfH4000,
DfH3500,
DfH3000,
DfH2500,
DfH2000,
DfH1500,
DfH1000,
DfH0500]
ListDataFName=['DfA0200',
'DfA0100',
'DfA0080',
'DfA0070',
'DfA0060',
'DfA0050',
'DfA0030',
'DfB1000',
'DfB0300',
'DfB0100',
'DfB0070',
'DfB0065',
'DfB0060',
'DfB0050',
'DfB0035',
'DfC1500',
'DfC0500',
'DfC0300',
'DfC0100',
'DfC0070',
'DfC0020',
'DfD2000',
'DfD1000',
'DfD0400',
'DfD0200',
'DfD0100',
'DfD0060',
'DfD0040',
'DfE2500',
'DfE1500',
'DfE1000',
'DfE0700',
'DfE0500',
'DfE0300',
'DfE0200',
'DfE0100',
'DfF3500',
'DfF3000',
'DfF2500',
'DfF2000',
'DfF1500',
'DfF1000',
'DfF0700',
'DfF0500',
'DfG5000',
'DfG4500',
'DfG4000',
'DfG3500',
'DfG3000',
'DfG2500',
'DfG2000',
'DfG1500',
'DfG1000',
'DfG0800',
'DfG0500',
'DfH5000',
'DfH4500',
'DfH4000',
'DfH3500',
'DfH3000',
'DfH2500',
'DfH2000',
'DfH1500',
'DfH1000',
'DfH0500']
'''
ColorList=['#FFCCCC',
'#FF9999',
'#FF6666',
'#FF3333',
'#FF0000',
'#CC0000',
'#990000',
'#FFE5CC',
'#FFCC99',
'#FFB266',
'#FF9933',
'#FF8000',
'#CC6600',
'#994C00',
'#663300',
'#FFFFCC',
'#FFFF99',
'#FFFF66',
'#FFFF33',
'#FFFF00',
'#CCCC00',
'#E5FFCC',
'#CCFF99',
'#B2FF66',
'#99FF33',
'#80FF00',
'#66CC00',
'#4C9900',
'#CCFFFF',
'#99FFFF',
'#66FFFF',
'#33FFFF',
'#00FFFF',
'#00CCCC',
'#009999',
'#006666',
'#CCE5FF',
'#99CCFF',
'#66B2FF',
'#3399FF',
'#0080FF',
'#0066CC',
'#004C99',
'#003366',
'#FFE5FF',
'#FFCCFF',
'#E5CCFF',
'#CC99FF',
'#B266FF',
'#9933FF',
'#7F00FF',
'#6600CC',
'#4C0099',
'#330066',
'#190033',
'#FFDFEF',
'#FFCCE5',
'#FF99CC',
'#FF66B2',
'#FF3399',
'#FF007F',
'#CC0066',
'#99004C',
'#660033',
'#330019']

'''

####################TEST CLASSE#############
#class Record:
#    #constructeur, appelee invariablement quand on souhaite cree un objet depuis la classe
#    #tous les constructeurs s apellent __init__
#    #le premier parametre doit etre self
#    def __init__(self,
#                 name='name',
#                 needle='A',
#                 name_record_file='',
#                 path_folder=r'',
#                 
#                 fps='',
#                 dx='',
#                 images='',
#                 thresh='',
#                 date='no_data',
#                 study='no_data',
#                 camera='no_data',
#                 objectif='no_data',
#                 obectif_extension='no_data',
#                 
#                 MaskBottom='no_data',
#                 MaskTop='no_data',
#                 MaskRight='no_data',
#                 MaskLeft='no_data',
#                 
#                 MinRadius='no_data',
#                 MaxRadius='no_data',
#                 MinRadius_holes='no_data',
#                 
#                 search_range='no_data',
#                 memory='no_data',
#                 
#                 Flow_rate='no_data',
#                 Inlet_pressure='no_data',
#                 T_controler='no_data',
#                 T_lab='no_data',
#                 Humidity='no_data',
#                 Exp_time='no_data',
#                 Surfactant='no_data',
#                 Age_of_the_water='no_data',
#                 path_metric='no_data',
#                 
#                 color='no_data',
#                 df_Brut='no_data',
#                 df_filled='no_data',
#                 df_tracked='no_data',
#                 df_tracked_filled='no_data',
#                 
#                 df_Brut_BUBB='no_data',
#                 df_filled_BUBBLE='no_data',
#                 df_tracked_BUBBLE='no_data',
#                 df_tracked_BUBBLEfilled='no_data',
#                 
#                 df_Brut_CLUSTER='no_data',
#                 df_filled_CLUSTER='no_data',
#                 df_tracked_CLUSTER='no_data',
#                 df_tracked_filled_CLUSTER='no_data',
#                 ):
#        #ID
#        self.name=name
#        self.date=date
#        self.study=study
#        
#        
#        self.needle=needle
#        self.name_record_file=name_record_file
#
#        self.path_folder=path_folder
#        
#        #record
#        self.images=pims.open(path_folder+name_record_file)
#        self.nb_images=len(self.images)
#        self.shape_images=(np.shape(self.images[0]))
#        self.frames_all=np.arange(0,self.nb_images,1)
#        self.camera=camera
#        self.objectifs=objectifs
#        
#        #geometry+time+processing
#        self.fps=fps
#        self.dx=dx
#        self.g=fluids2d.geometry.GeometryScaler(dx=self.dx,im_shape=self.shape_images,origin_pos=(0,0),origin_units='pix')
#        self.frames=frames = np.arange(0,self.nb_images,1)
#        self.thresh=thresh
#        
#        self.MaskBottom=MaskBottom
#        self.MaskTop=MaskTop
#        self.MaskRight=MaskRight
#        self.MaskLeft=MaskLeft
#        
#        self.MinRadius=MinRadius
#        self.MaxRadius=MaxRadius
#        self.MinRadius_holes=MinRadius_holes
#        
#        self.search_range=search_range
#        self.memory=memory
#        
#        
#        #Experimental stuffs
#        self.Flow_rate=Flow_rate
#        self.Inlet_pressure=Inlet_pressure
#        self.T_controler=T_controler
#        self.T_lab=T_lab
#        self.Humidity=Humidity
#        self.Exp_time=Exp_time
#        self.Surfactant=Surfactant
#        self.Age_of_the_water=Age_of_the_water
#        self.path_metric=path_metric
#
#        #plots
#        self.color=color
#        
#        #Data_Frames
##        for df in (df_Brut,df_filled,df_tracked,df_tracked_filled,
##                   df_Brut_BUBB,df_filled_BUBBLE,df_tracked_BUBBLE,df_tracked_BUBBLEfilled,
##                   df_Brut_CLUSTER,df_filled_CLUSTER,df_tracked_CLUSTER,df_tracked_filled_CLUSTER):
#        
#        if df_Brut=='no_data':
#            self.df_Brut=pd.DataFrame()
#        elif type(df_Brut)==str and df_Brut!='no_data':
#            self.df_Brut=pd.read_excel(self.path_folder+df_Brut)
#        else:
#            self.df_Brut=df_Brut
#        
#        if df_filled=='no_data':
#            self.df_filled=pd.DataFrame()
#        elif type(df_filled)==str and df_filled!='no_data':
#            self.df_filled=pd.read_excel(self.path_folder+df_filled)
#        else:
#            self.df_filled=df_filled
#        
#        if df_tracked=='no_data':
#            self.df_tracked=pd.DataFrame()
#        elif type(df_tracked)==str and df_tracked!='no_data':
#            self.df_tracked=pd.read_excel(self.path_folder+df_tracked)
#        else:
#            self.df_tracked=df_tracked
#        
#        if df_tracked_filled=='no_data':
#            self.df_tracked_filled=pd.DataFrame()
#        elif type(df_tracked_filled)==str and df_tracked_filled!='no_data':
#            self.df_tracked_filled=pd.read_excel(self.path_folder+df_tracked_filled)
#        else:
#            self.df_tracked_filled=df_tracked_filled
#        
#        if df_Brut_BUBB=='no_data':
#            self.df_Brut_BUBB=pd.DataFrame()
#        elif type(df_Brut_BUBB)==str and df_Brut_BUBB!='no_data':
#            self.df_Brut_BUBB=pd.read_excel(self.path_folder+df_Brut_BUBB)
#        else:
#            self.df_Brut_BUBB=df_Brut_BUBB
#        
#        if df_filled_BUBBLE=='no_data':
#            self.df_filled_BUBBLE=pd.DataFrame()
#        elif type(df_filled_BUBBLE)==str and df_filled_BUBBLE!='no_data':
#            self.df_filled_BUBBLE=pd.read_excel(self.path_folder+df_filled_BUBBLE)
#        else:
#            self.df_filled_BUBBLE=df_filled_BUBBLE
#        
#        if df_tracked_BUBBLE=='no_data':
#            self.df_tracked_BUBBLE=pd.DataFrame()
#        elif type(df_tracked_BUBBLE)==str and df_tracked_BUBBLE!='no_data':
#            self.df_tracked_BUBBLE=pd.read_excel(self.path_folder+df_tracked_BUBBLE)
#        else:
#            self.df_tracked_BUBBLE=df_tracked_BUBBLE
#        
#        if df_tracked_BUBBLEfilled=='no_data':
#            self.df_tracked_BUBBLEfilled=pd.DataFrame()
#        elif type(df_tracked_BUBBLEfilled)==str and df_tracked_BUBBLEfilled!='no_data':
#            self.df_tracked_BUBBLEfilled=pd.read_excel(self.path_folder+df_tracked_BUBBLEfilled)
#        else:
#            self.df_tracked_BUBBLEfilled=df_tracked_BUBBLEfilled
#        
#        if df_Brut_CLUSTER=='no_data':
#            self.df_Brut_CLUSTER=pd.DataFrame()
#        elif type(df_Brut_CLUSTER)==str and df_Brut_CLUSTER!='no_data':
#            self.df_Brut_CLUSTER=pd.read_excel(self.path_folder+df_Brut_CLUSTER)
#        else:
#            self.df_Brut_CLUSTER=df_Brut_CLUSTER
#        
#        if df_filled_CLUSTER=='no_data':
#            self.df_filled_CLUSTER=pd.DataFrame()
#        elif type(df_filled_CLUSTER)==str and df_filled_CLUSTER!='no_data':
#            self.df_filled_CLUSTER=pd.read_excel(self.path_folder+df_filled_CLUSTER)
#        else:
#            self.df_filled_CLUSTER=df_filled_CLUSTER
#        
#        if df_tracked_CLUSTER=='no_data':
#            self.df_Brut=pd.DataFrame()
#        elif type(df_tracked_CLUSTER)==str and df_tracked_CLUSTER!='no_data':
#            self.df_tracked_CLUSTER=pd.read_excel(self.path_folder+df_tracked_CLUSTER)
#        else:
#            self.df_tracked_CLUSTER=df_tracked_CLUSTER
#        
#        if df_tracked_filled_CLUSTER=='no_data':
#            self.df_tracked_filled_CLUSTER=pd.DataFrame()
#        elif type(df_tracked_filled_CLUSTER)==str and df_tracked_filled_CLUSTER!='no_data':
#            self.df_tracked_filled_CLUSTER=pd.read_excel(self.path_folder+df_tracked_filled_CLUSTER)
#        else:
#            self.df_tracked_filled_CLUSTER=df_tracked_filled_CLUSTER
#        
#
#
#        
##                df_Brut='no_data',
##                df_filled='no_data',
##                df_tracked='no_data',
##                df_tracked_filled='no_data',
##                 
##                df_Brut_BUBB='Bubble_NeedleG0970sccm_1000.xlsx',
##                df_filled_BUBBLE='no_data',
##                df_tracked_BUBBLE='no_data',
##                df_tracked_BUBBLEfilled='no_data',
##                
##                df_Brut_CLUSTER='no_data',
##                df_filled_CLUSTER='no_data',
##                df_tracked_CLUSTER='no_data',
##                df_tracked_filled_CLUSTER='no_data'
#
#    def Process(self,frames,method='random_walker_detection_cluster_holes',MinRadius=0,MaxRadius=10000,MinRadius_holes=0):
#        if method=='random_walker_detection_cluster_holes':
#            self.df_Brut_BUBB=traitement(self.images,self.thresh,self.g,frames=frames,method=method,Radius=1,MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=2)[0]
#            self.df_Brut_CLUSTER=traitement(self.images,self.thresh,self.g,frames=frames,method=method,Radius=1,MinRadius=MinRadius,MaxRadius=MaxRadius,MinRadius_holes=MinRadius_holes,RadiusDiskMean=2)[0]
#            self.df_Brut_BUBB.to_excel(self.path_folder+name+str(df_Brut_BUBB)+'.xlsx')
#            self.df_Brut_CLUSTER.to_excel(self.path_folder+name+str(df_Brut_CLUSTER)+'.xlsx')
#    
#    def tri_radius_b(self,Rmin,Rmax):
#         self.df_filled_BUBBLE=self.df_Brut_BUBB[self.df_Brut_BUBB['radius']>Rmin]
#         self.df_filled_BUBBLE=self.df_Brut_BUBB[self.df_Brut_BUBB['radius']<Rmax]
#
#    def tri_radius_c(self,Rmin,Rmax):
#         self.df_filled_CLUSTER=self.df_Brut_CLUSTER[self.df_Brut_CLUSTER['radius']>Rmin]
#         self.df_filled_CLUSTER=self.df_Brut_CLUSTER[self.df_Brut_CLUSTER['radius']<Rmax]
#         plt.plot(self.df_filled_CLUSTER['time'],self.df_filled_CLUSTER['radius'])
#
##    def combien(cls):
##        #methode de classe
##        print(objets_crees)
'''

