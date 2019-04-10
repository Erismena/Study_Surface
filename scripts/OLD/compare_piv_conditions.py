# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 17:30:19 2017

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import fluids2d.piv as piv
import pickle
import fluids2d.geometry
import pandas as pd

parent_folder = r'C:\Users\danjr\Documents\Fluids Research\Data\misc\PIV\with_cap\\'

#name_dict = {3:'PIV_sv_fps4k_grid3x3x10_withCap_topRegion_Cam_20861_Cine7',
#            5:'PIV_sv_fps4k_grid3x3x10_withCap_topRegion_Cam_20861_Cine8',
#            8:'PIV_sv_fps4k_grid3x3x10_withCap_topRegion_Cam_20861_Cine9',
#            10:'PIV_sv_fps4k_grid3x3x10_withCap_topRegion_Cam_20861_Cine10'}
#
#c_dict = {3:'b',5:'g',8:'orange',10:'r'}


name_list = ['PIV_sv_fps4k_grid3x3x10_withCap_topRegion_Cam_20861_Cine3',
             'PIV_sv_fps4k_grid3x3x10_withCap_topRegion_Cam_20861_Cine1',
             'PIV_sv_fps4k_grid3x3x10_withCap_topRegion_Cam_20861_Cine4',
             'PIV_sv_fps4k_grid3x3x10_withCap_topRegion_Cam_20861_Cine5',
             'PIV_sv_fps4k_grid3x3x10_withCap_topRegion_Cam_20861_Cine6',
             'PIV_sv_fps4k_grid3x3x10_withCap_midRegion_Cam_20861_Cine3',
             'PIV_sv_fps4k_grid3x3x10_withCap_midRegion_Cam_20861_Cine1',
             'PIV_sv_fps4k_grid3x3x10_withCap_midRegion_Cam_20861_Cine4',
             'PIV_sv_fps4k_grid3x3x10_withCap_midRegion_Cam_20861_Cine5',
             'PIV_sv_fps4k_grid3x3x10_withCap_midRegion_Cam_20861_Cine6']

#name_list = ['PIV_sv_fps4k_grid3x3x10_withCap_topRegion_Cam_20861_Cine3',
#             'PIV_sv_fps4k_grid3x3x10_withCap_topRegion_Cam_20861_Cine7',
#             'PIV_sv_fps4k_grid3x3x10_withCap_topRegion_Cam_20861_Cine8',
#             'PIV_sv_fps4k_grid3x3x10_withCap_topRegion_Cam_20861_Cine9',
#             'PIV_sv_fps4k_grid3x3x10_withCap_topRegion_Cam_20861_Cine10',
#             'PIV_sv_fps4k_grid3x3x10_withCap_topRegion_Cam_20861_Cine3',
#             'PIV_sv_fps4k_grid3x3x10_withCap_midRegion_Cam_20861_Cine7',
#             'PIV_sv_fps4k_grid3x3x10_withCap_midRegion_Cam_20861_Cine8',
#             'PIV_sv_fps4k_grid3x3x10_withCap_midRegion_Cam_20861_Cine9',
#             'PIV_sv_fps4k_grid3x3x10_withCap_midRegion_Cam_20861_Cine10']

A_list = [1,2,3,4,5,1,2,3,4,5]
region_list = ['top','top','top','top','top','mid','mid','mid','mid','mid']

meta = pd.DataFrame(index=name_list)
meta['A'] = A_list
meta['region'] = region_list

name_df = pd.DataFrame(index=np.arange(1,6),columns=['mid','top'])
name_df.loc[:,'top'] = name_list[0:5]
name_df.loc[:,'mid'] = name_list[5:11]


c_dict = {1:'b',2:'g',3:'orange',4:'r',5:'k'}

p = {}
ff = {}
g = {}

vmin = -0.1
vmax = 0.1


for k in name_list:
    p[k] = pickle.load(open(parent_folder+k+'.pkl'))
    p[k].parent_folder = parent_folder
    p[k].associate_flowfield()
    
    ff[k] = p[k].data.ff
    
    if k == 'PIV_sv_fps4k_grid3x3x10_withCap_topRegion_Cam_20861_Cine3':
        ff[k] = ff[k][:,:,18:18+69,:]
        
    ff[k][ff[k]>0.5] = 0.5
    ff[k][ff[k]<-0.5] = -0.5
        
    g[k] = fluids2d.geometry.GeometryScaler(dx=p[k].dx*(p[k].window_size-p[k].overlap),origin_pos=0)
    g[k].calibrate_image(ff[k][0,:,:,0])

figu,axu = plt.subplots(2,5)
figv,axv = plt.subplots(2,5)
figtke,axtke = plt.subplots(2,5)

figmid,axmid = plt.subplots(2,1,sharex=True)

for k in name_list:
    print(k)
    print(np.shape(ff[k]))

for ai,A in enumerate(name_df.index):
    
    for ri,region in enumerate(['top','mid']):
    
        k = name_df.loc[A,region]
        
        left = int(g[k].get_coords([np.nan,0.07])[1])
        right = int(g[k].get_coords([np.nan,0.08])[1])
        cols_to_use = np.arange(left,right)
        
        meanu = np.nanmean(ff[k][:,:,cols_to_use,0],axis=2)
        meanv = np.nanmean(ff[k][:,:,cols_to_use,1],axis=2)
        
        meanspeed = np.nanmean(np.sqrt( ff[k][:,:,:,0]**2 + ff[k][:,:,:,1]**2 ),axis=0)
        
        axu[ri,ai].imshow(np.nanmean(ff[k],axis=0)[:,:,0],vmin=vmin,vmax=vmax,cmap='PuOr',extent=g[k].im_extent)
        axv[ri,ai].imshow(np.nanmean(ff[k],axis=0)[:,:,1],vmin=vmin,vmax=vmax,cmap='seismic',extent=g[k].im_extent)
        
        meanflow = np.nanmean(ff[k],axis=0)
        tke = np.sqrt(np.nanmean( (ff[k] - meanflow)[:,:,:,0]**2, axis=0) + np.nanmean( (ff[k] - meanflow)[:,:,:,1]**2, axis=0) )
        
        im=axtke[ri,ai].imshow(tke,vmin=0,vmax=0.2,extent=g[k].im_extent)
        
        [a.set_title('A = '+str(A)+' mm') for a in [axu[ri,ai],axv[ri,ai],axtke[ri,ai]] if region=='top']
        [a.set_ylabel(region) for a in [axu[ri,ai],axv[ri,ai],axtke[ri,ai]] if A==1]
        
        if ri==1:
            label = str(A)+' mm'
        else:
            label=None
        
        axmid[ri].plot(np.nanmean(tke[:,cols_to_use],axis=1),np.flipud(g[k].y),color=c_dict[A],label=label)
        
        axmid[ri].plot(np.nanmean(meanspeed[:,cols_to_use],axis=1),np.flipud(g[k].y),'--',color=c_dict[A],alpha=0.5)
        #axmid[ri,1].plot(np.nanmean(tke,axis=0),g[k].y,color=c_dict[A])
        
axmid[1].legend()

axmid[0].plot([],[],'-',color='k',label=r'''$\sqrt{\overline{{{u'}^2}} + \overline{{{v'}^2}}}$''')
axmid[0].plot([],[],'--',color='k',alpha=0.5,label=r'''$\overline{\sqrt{U^2 + V^2}}$''')

axmid[0].legend()


axmid[0].set_ylabel('top portion \nvertical position  [m]')
axmid[1].set_ylabel('middle portion \nvertical position  [m]')
axmid[1].set_xlabel('[m/s]')

figmid.suptitle('Vertical profiles with f = 10 Hz')



figtke.suptitle('TKE with forcing at f = 10 Hz',y = 0.93,fontsize=24)
figtke.subplots_adjust(right=0.88)
cbar_ax = figtke.add_axes([0.93, 0.4, 0.03, 0.2])
cbar_ax.set_title('TKE\n'+r'''$\sqrt{\overline{{{u'}^2}} + \overline{{{v'}^2}}}$ [m/s]'''+'\n')
figtke.colorbar(im, cax=cbar_ax)