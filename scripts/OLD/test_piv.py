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

import fluids2d.piv

parent_folder = r'C:\Users\danjr\Documents\Fluids Research\Data\misc\PIV\\'

ff_to_load = [r'PIV_svCloseUpCenterThird_makro100mmzeiss_X0mm_Y15mm_fps2500_A15mm_f10Hz_grid3x4_10cycles_flowfield.npy',]

# load the data
ff_all = [np.load(r'C:\Users\danjr\Documents\Fluids Research\Data\misc\PIV\\'+f) for f in ff_to_load]

dx_bottom = 0.000122917675934896 ## m
dx_center = 0.00012066365007541

dx = [dx_center,dx_center,dx_bottom,dx_bottom]
dt = (1./1000) * 2500 # s, 1000 is to account for the .001 assumed for dt in extended_search_area_piv

# m/s = (pix/s1) * (s1/frame) * (frame/s)  # s1 is the assumed dt

#fpc = [1000,500,1000,500]
fpc = [500,250,500,250]


#vmean = np.mean(ff[:,:,:,0],axis=0)

'''
Get the phase average of each condition
'''

phase_averages = []
for i,ff in enumerate(ff_all):
    
    ff = ff[0:fpc[i]*5,:,:,:]
    
    # first reshape the array: [cycle,frame,row,col,dir]
    ff_shape = np.shape(ff)
    frames_per_cycle=fpc[i]
    #frames_per_cycle = ff_shape[0]
    num_cycles=ff_shape[0]//frames_per_cycle
    ff_by_cycle = ff.reshape((num_cycles,frames_per_cycle,ff_shape[1],ff_shape[2],2))
    ff_by_cycle[np.abs(ff_by_cycle)>10*np.nanmean(np.abs(ff_by_cycle))] = np.nan
    ff_phase_averaged = np.nanmean(ff_by_cycle,axis=0) * dx[i] / dt
    
    '''
    Double the data for the 10 Hz cases so the duration is the same as for the
    5 Hz cases
    '''
    if (i==1) or (i==3):
        ff_phase_averaged = np.concatenate((ff_phase_averaged,ff_phase_averaged),axis=0)
    
    phase_averages.append(ff_phase_averaged)
    

vmin=0
vmax=0.004

fig = plt.figure(figsize=(16,9))
ax = [fig.add_subplot(2,2,i+1) for i in range(4)]

for i in range(4):
    ff_phase_averaged = phase_averages[i]
    speed_inst = np.sqrt(ff_all[i][:,:,:,0]**2+ff_all[i][:,:,:,1]**2)
    speed_avg = np.nanmean(speed_inst,axis=0)
    
    fluc = speed_inst - speed_avg
    
    rms = np.sqrt( np.nanmean(fluc**2,axis=0))
    
    ax[i].matshow(speed_avg,vmin=vmin,vmax=vmax)
    ax[i].quiver(np.nanmean(ff_phase_averaged[:,:,:,0],axis=0),np.nanmean(ff_phase_averaged[:,:,:,1],axis=0))
    
    ax[i].xaxis.set_ticks([]) 
    ax[i].yaxis.set_ticks([]) 


for fi in np.arange(0,500,1):
    fig = plt.figure(figsize=(16,9))
    ax = [fig.add_subplot(2,2,i+1) for i in range(4)]
    plt.tight_layout()
    for i in range(4):
        ax[i].clear()
        #ax[i].set_title(i)
        ff_phase_averaged = phase_averages[i]
        speed = np.sqrt(ff_phase_averaged[fi,:,:,0]**2+ff_phase_averaged[fi,:,:,1]**2)
        ax[i].matshow(speed,vmin=vmin,vmax=vmax)
        ax[i].quiver(ff_phase_averaged[fi,:,:,0],ff_phase_averaged[fi,:,:,1])
        
        ax[i].xaxis.set_ticks([]) 
        ax[i].yaxis.set_ticks([]) 

    
    ax[0].set_title('f = 5 Hz')
    ax[1].set_title('f = 10 Hz')
    ax[0].set_ylabel('A = 5 mm')
    ax[2].set_ylabel('A = 10 mm')
    
    plt.show()
    plt.pause(0.2)
    fig.savefig(r'C:\Users\danjr\Documents\Fluids Research\Data\misc\PIV\amp_freq_comparison\\'+format(fi,'05')+'.png')
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