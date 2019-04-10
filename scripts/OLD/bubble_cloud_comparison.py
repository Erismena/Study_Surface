# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 19:03:44 2017

@author: danjr
"""

import matplotlib.pyplot as plt
import numpy as np
import fluids2d.geometry
import fluids2d.backlight
import pims

folder = r'C:\Users\Luc Deike\Documents\high_speed_data\171028\bubble_cloud\\'

file_list = [r'backlight_sv_grid3tx2x20_fps2000_cloud_25mL_fast_noTurbulence_b',
             r'backlight_sv_grid3tx2x20_fps2000_cloud_25mL_fast_f10Hz_A4mm_b',
             r'backlight_sv_grid3tx2x20_fps2000_cloud_25mL_fast_f5Hz_A10mm_b']

title_list = ['no turbulence','10 Hz, 4 mm','5 Hz, 10 mm']

frame_start = [0,0,0]



g = fluids2d.geometry.GeometryScaler(dx=0.000126336189127,im_shape=(1280,800),origin_pos=(0.096,0),origin_units='m')

def mask(im):
    im = im.astype(float)
    #im[0:30,:] = 0
    #im[380:400,:] = 0
    
    #im[:,254:422] = 0
    #im[:,814:965] = 0
    
    im = np.rot90(im)
    
    return im

cine_list = [pims.open(folder+f+'.cine') for f in file_list]
bgs = [mask(c[0]) for c in cine_list]

len_cines = [len(c)-frame_start[ci] for ci,c in enumerate(cine_list)]
time = np.arange(min(len_cines)) * 1./2000.

composite_img_list = [np.zeros((np.shape(bgs[0])[0],min(len_cines))) for _ in range(len(file_list))]

for i in range(0,min(len_cines),1):
    #fig,axs = plt.subplots(1,3,figsize=(13,7))
    #axs = axs.flatten()
    for ci,c in enumerate(cine_list):
        f = i+frame_start[ci]
        im = mask(c[f]) - bgs[ci]
        
        composite_img_list[ci][:,i] = np.mean(np.abs(im),axis=1)
        
#        axs[ci].clear()
#        axs[ci].imshow(np.rot90(c[f]),vmin=0,vmax=800,cmap='gray',extent=g.im_extent)
#        axs[ci].set_title(title_list[ci])
#        
#    plt.tight_layout()
#    fig.savefig(folder+r'comparison_closeup_b_frames\frame_'+str(i)+'.png')
#    plt.close('all')
        
fig,axs = plt.subplots(1,3,figsize=(13,5))
axs = axs.flatten()

[axs[i].imshow(composite_img_list[i],vmin=0,vmax=100,cmap='gray_r',aspect='auto',extent=[time[0],time[-1],g.im_extent[2],g.im_extent[3]]) for i in range(0,3)]
[axs[ci].set_title(title_list[ci]) for ci in range(3)]
[axs[ci].set_xlabel('time [s]') for ci in range(3)]
[ax.grid(True,linestyle='--',color='k',alpha=0.5) for ax in axs]
axs[0].set_ylabel('vertical position [m]')
plt.tight_layout()
fig.savefig(folder+'comparisons_closeup_b.png')


fig= plt.figure()
ax = fig.add_subplot(111)

import matplotlib
cmap = matplotlib.cm.get_cmap('inferno')
rgba = cmap(0.5)

#base_colors = [cmap(i)[0:3] for i in np.linspace(0,1,len(file_list))]
base_colors = [[1,0,0],[0,1,0],[0,0,1]]

for ci in np.arange(len(file_list)):
    comp = composite_img_list[ci]
    colors4 = np.zeros((np.shape(comp)[0],np.shape(comp)[1],4))
    
    for i in [0,1,2]:
        #im = (base_colors[fi][i] * (crop(c[f]) + 500)) / scale
        im = base_colors[ci][i] * np.ones(np.shape(comp))
        colors4[:,:,i] = im
        
    alpha = (comp -10) / 70
    alpha[alpha<0] = 0
    alpha[alpha>0.8] = 0.8
    colors4[:,:,3] = alpha
    
    ax.imshow(colors4)