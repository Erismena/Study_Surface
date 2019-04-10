# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 23:28:34 2017

@author: danjr
"""

import pims
import numpy as np
import openpiv.process
import openpiv.validation
import openpiv.filters
import matplotlib.pyplot as plt

import fluids2d.piv as piv

parent_folder = r'E:\Experiments_Stephane\Grid column\PIV_measurements\Hollow_grid_3\20170816\Cines_B\\'

keys=('center_05','center_10','bottom_05','bottom_10')
res_to_load = {'center_05':r'PIV_svCloseUpCenterThird_makro100mmzeiss_X0mm_Y15mm_fps2500_A15mm_f05Hz_grid3x4_10cycles',
               'center_10':r'PIV_svCloseUpCenterThird_makro100mmzeiss_X0mm_Y15mm_fps2500_A15mm_f10Hz_grid3x4_10cycles',
               'bottom_05':r'PIV_svCloseUpBottomThird_makro100mmzeiss_X0mm_Y05mm_fps2500_A15mm_f05Hz_grid3x4_10cycles',
               'bottom_10':r'PIV_svCloseUpBottomThird_makro100mmzeiss_X0mm_Y05mm_fps2500_A15mm_f10Hz_grid3x4_10cycles'}
#keys = list(res_to_load.keys())

p = {key:piv.load_processed_PIV_data(parent_folder,res_to_load[key]) for key in list(res_to_load.keys())}

# load the data

dx_bottom = 0.000122917675934896 ## m
dx_center = 0.00012066365007541

'''
Correct the calibrations
'''
if True:
    p['bottom_05'].dx_orig = dx_bottom
    p['bottom_10'].dx_orig = dx_bottom
    p['center_05'].dx_orig = dx_center
    p['center_10'].dx_orig = dx_center


vmin=0
vmax=0.2

fig = plt.figure(figsize=(16,9))
ax = {k:fig.add_subplot(2,2,i+1) for i,k in enumerate(keys)}

import scipy.ndimage
filtered_flowfield={}
for k in keys:
    filtered_flowfield[k] = scipy.ndimage.median_filter(p[k].data.ff,size=[3,1,1,1])[0:50,:,:,:]
    mean_flow = np.nanmean(filtered_flowfield[k],axis=0)
    #mean_speed = np.sqrt( mean_flow[:,:,0]**2 + mean_flow[:,:,1]**2 )
    mean_speed=np.nanmean(np.sqrt(filtered_flowfield[k][:,:,:,0]**2+filtered_flowfield[k][:,:,:,1]**2),axis=0)
    
    speed_inst=np.sqrt(filtered_flowfield[k][:,:,:,0]**2+filtered_flowfield[k][:,:,:,1]**2)
    u_rms = np.sqrt(np.nanmean(speed_inst**2,axis=0))
    
    ax[k].imshow(u_rms,vmin=vmin,vmax=vmax)
    ax[k].quiver(mean_flow[:,:,0],mean_flow[:,:,1])
    ax[k].set_title(k)



for fi in np.arange(0,500,1):
    fig = plt.figure(figsize=(16,9))
    ax = {k:fig.add_subplot(2,2,i+1) for i,k in enumerate(keys)}

    plt.tight_layout()
    for k in keys:
        ax[k].clear()
        speed = np.sqrt(filtered_flowfield[k][fi,:,:,0]**2+filtered_flowfield[k][fi,:,:,1]**2)
        ax[k].matshow(speed,vmin=vmin,vmax=vmax)
        ax[k].quiver(filtered_flowfield[k][fi,:,:,0],filtered_flowfield[k][fi,:,:,1])
        
        ax[k].xaxis.set_ticks([]) 
        ax[k].yaxis.set_ticks([]) 
        ax[k].set_title(k)

    
    
    plt.show()
    plt.pause(0.2)
    fig.savefig(r'E:\Experiments_Stephane\Grid column\PIV_measurements\Hollow_grid_3\20170816\Cines_B\amp_freq_comparison\\'+format(fi,'05')+'.png')
    plt.close(fig)
    
    
    
    
    
    
fig=plt.figure()
ax=fig.add_subplot(111)


    
for fi in np.arange(0,np.shape(ff_phase_averaged)[0],1):
    ax.clear()
    ax.matshow(speed[fi,:,:],vmin=vmin,vmax=vmax)
    ax.quiver(ff_phase_averaged[fi,:,:,0],ff_phase_averaged[fi,:,:,1])
    
    plt.show()
    plt.pause(0.1)
    
maxspeed=3000
speed_inst = np.sqrt(ff_mean[:,:,:,0]**2+ff_mean[:,:,:,1]**2)
speed_inst[speed_inst>maxspeed] = maxspeed
speed_inst[speed_inst<-1*maxspeed] = -1*maxspeed

fig=plt.figure()
ax=fig.add_subplot(111)
vmin=np.nanmean(speed_inst)-np.nanstd(speed_inst)
vmax=np.nanmean(speed_inst)+np.nanstd(speed_inst)
for fi in np.arange(0,np.shape(ff_mean)[0],1):
    ax.clear()
    ax.matshow(speed_inst[fi,:,:],vmin=vmin,vmax=vmax)
    ax.quiver(ff_mean[fi,:,:,0],ff_mean[fi,:,:,1])
    
    plt.show()
    plt.pause(0.1)
    
'''
Animation
'''

fig=plt.figure()
ax=fig.add_subplot(111)
vmin=np.nanmean(speed_inst)-np.nanstd(speed_inst)
vmax=np.nanmean(speed_inst)+np.nanstd(speed_inst)
for fi in np.arange(0,np.shape(ff)[0],50):
    ax.clear()
    ax.matshow(speed_inst[fi,:,:],vmin=vmin,vmax=vmax)
    ax.quiver(ff[fi,:,:,0],ff[fi,:,:,1])
    
    plt.show()
    plt.pause(0.1)

uxx = np.gradient(np.gradient(ff[:,:,:,0],axis=1),axis=1)
vyy = np.gradient(np.gradient(ff[:,:,:,1],axis=0),axis=0)

shear = 0.5 * (uxx**2 + vyy**2)

plt.figure()
plt.imshow(np.nanmean(shear,axis=0),vmax=1e5)


speed_point = ff[:,5,24,0]

plt.figure()
plt.plot(u_point)

fig=plt.figure(); ax=fig.add_subplot(111)
freq=np.fft.fftfreq(len(u_point),0.004)
f = np.fft.fft(u_point)

ax.semilogy(freq[freq>0],np.absolute(f[freq>0]),color='k')

#piv2 = fluids2d.piv.PIVData(parent_folder,cine_name2)


#from multiprocessing import Process
#p1=Process(target=piv1.run_analysis,args=(np.arange(400,403,1),))
#p2=Process(target=piv2.run_analysis,args=(np.arange(100,500,100),))

#p1.start()
#out2=p2.start()

#stophere
#
#fluids2d.piv.show_frame(flowfield[0,:,:,0],flowfield[0,:,:,1],bg='shear')
#        
#u_mean = np.nanmean(flowfield[:,:,:,0],axis=0)
#v_mean = np.nanmean(flowfield[:,:,:,1],axis=0)
#
#fluids2d.piv.show_frame(u_mean,v_mean)