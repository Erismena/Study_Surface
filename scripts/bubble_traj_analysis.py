# -*- coding: utf-8 -*-
"""
Created on Fri Mar 02 10:09:11 2018

@author: user
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trackpy as tp
import fluids2d.spectra as spectra
import fluids2d.backlight as backlight


#folder = r'\\Mae-deike-lab3\c\Users\Luc Deike\data_comp3_C\180302\\'
folder = r'\\Mae-deike-lab3\c\Users\Luc Deike\data_comp3_C\180323\\'
cine_name = r'backlight_topDown_bubblesRising_smallNotTinyNeedle_2x2x2_on100_off400_fps1000_withCap'


dt = 0.001
#dx = 0.00020049103322
#dx = 0.000222484436389
#dx = 0.000176940825543




'''
Load the data
'''
tracked = pd.read_pickle(folder+cine_name+'_tracked.pkl')

'''
Plot all the trajectories
'''
#fig = plt.figure()
#ax = fig.add_subplot(111)
#tp.plot_traj(tracked,colorby='particle',ax=ax)
#ax.invert_yaxis()
#ax.set_aspect('equal')

'''
Make a list of dfs each corresponding to one bubble
'''
bubbles = backlight.make_list_of_bubbles(tracked,min_length=30)
bubbles = [b for b in bubbles if np.sqrt((b['y'].max()-b['y'].min())**2+(b['x'].max()-b['x'].min())**2)>0.01]
print('starting terminate_once_static')
bubbles = [backlight.terminate_once_static(b,100,0.002) for b in bubbles]
bubbles = [backlight.interp_df_by_frame(b,dt) for b in bubbles]


meta = pd.DataFrame(columns=['med_radius','med_u','med_v'])

fig = plt.figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312,sharex=ax1)
ax3 = fig.add_subplot(313,sharex=ax1)


fig=plt.figure()
ax=fig.add_subplot(111)

C_list = []

f_list = []
freq_list = []

for bi,b in enumerate(bubbles):
    print(bi)

    b = backlight.compute_transient_quantities(b,roll_vel=3)
    
    ax1.plot(b.index,b['u'])
    ax2.plot(b.index,b['v'])
    ax3.plot(b.index,b['perimeter'])
    
    meta.loc[bi,'med_radius'] = b['radius'].median()
    meta.loc[bi,'med_u'] = b['u'].abs().median()
    meta.loc[bi,'med_v'] = b['v'].median()
    meta.loc[bi,'std_u'] = b['u'].std()
    meta.loc[bi,'std_v'] = b['v'].std()
    meta.loc[bi,'len'] = len(b)    
    
    ax.plot(b['orientation'],b['y'])
    
    #f,freq = spectra.abs_fft(b['u'].values/meta.loc[bi,'med_v'],b.index*meta.loc[bi,'med_v']/meta.loc[bi,'med_radius'])
    f,freq = spectra.abs_fft(b['u'].values,b.index)
    f_list.append(f)
    freq_list.append(freq)
    
fig = plt.figure()
ax = fig.add_subplot(111)
[ax.plot(b['x'],b['y'],alpha=0.3) for b in bubbles]
ax.set_aspect('equal')
    
mean_displacement_list = []
lags_list = []
for bi,b in enumerate(bubbles):
    print(bi)
    lags = np.arange(0,min(len(b),1000)-2,1)
    mean_displacement = np.zeros(np.shape(lags))
    
    for li,lag in enumerate(lags):
        mean_displacement[li] = np.sqrt(np.nanmean((b['x'].values[0:len(b)-lag]-b['x'].values[lag:len(b)])**2 + (b['y'].values[0:len(b)-lag]-b['y'].values[lag:len(b)])**2))
        
    lags_list.append(lags)
    mean_displacement_list.append(mean_displacement)
    
fig = plt.figure()
ax = fig.add_subplot(111)
urms = 0.1
[ax.loglog(l*dt,md**2/urms**2) for l,md in zip(lags_list,mean_displacement_list)]
ax.plot([.01,.1],[10**-3,10**-1],'--',color='k',alpha=0.5)
ax.set_xlabel('lag [s]')
ax.set_ylabel('MSD / u_rms [s^2]')
fig.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(111)
[ax.plot(l*dt,np.gradient(md)/(dt*(l[1]-l[0]))) for l,md in zip(lags_list,mean_displacement_list)]
    
fig = plt.figure()
ax=fig.add_subplot(111)
[ax.loglog(x,y,'x',alpha=0.3) for x,y in zip(freq_list,f_list)]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(meta['std_u'],meta['std_v'],c=[min(0.01,r) for r in meta['med_radius']],alpha=0.3)
ax.plot([0,0.25],[0,0.25],color='r')
ax.set_xlabel('std u [m/s]')
ax.set_ylabel('std v [m/s]')
fig.tight_layout()
fig.savefig(folder+'bubbles_velocity_stds.pdf')

#for freq,f in zip(freq_list,f_list):
    #

#df_f = pd.DataFrame(columns=np.arange(len(bubbles)))
#for bi,(freq,f) in enumerate(zip(freq_list,f_list)):
#    s = pd.Series(index=freq,data=np.absolute(f))
#    df_f[bi] = s

#    

all_freq = []
[all_freq.append(freq.tolist()) for freq in freq_list]
all_freq = [item for sublist in all_freq for item in sublist]

all_f = []
[all_f.append(f.tolist()) for f in f_list]
all_f = [item for sublist in all_f for item in sublist]

all_s = pd.Series(index=all_freq,data=all_f).sort_index()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(all_s.index,all_s,alpha=0.5)
ax.plot(all_s.rolling(window=100,center=True,min_periods=0).mean(),color='r')
ax.set_xscale('log')
ax.set_yscale('log')

all_s.to_pickle(folder+cine_name+'_all_s.pkl')


all_points = pd.concat(bubbles)
all_points['speed'] = np.sqrt(all_points['u']**2+all_points['v']**2)
fig = plt.figure()
ax = fig.add_subplot(111,projection='polar')
#ax.set_theta_zero_location("N")
#ax.set_theta_direction(-1)
ax.hist(all_points['direction']/180.*np.pi,bins=100) # ,weights=all_points['speed']
fig.tight_layout()
fig.savefig(folder+'bubbles_direction_histogram.pdf')


stophere

import pims

fig = plt.figure(figsize=(14,12))
ax = fig.add_subplot(111)
dx = 0.000216023661265

figfolder = r'\\Mae-deike-lab3\c\Users\Luc Deike\data_comp3_C\180323\backlight_topDown_frames\\'

skip = 10
c = pims.open(folder+cine_name+'.cine')
frames = range(0,len(c),skip)

n_rows = np.shape(c[0])[0]
n_cols = np.shape(c[0])[1]

for fi,f in enumerate(frames):
    print(f)
    ax.clear()
    ax.imshow(c[f],extent=[0,n_cols*dx,0,n_rows*dx],cmap='gray',vmin=50,vmax=500)
    
    for b in [b for b in bubbles if np.min(np.abs(b['frame']-f))<100]:
        df_this_frame = b[b['frame']==f]
        for idx in df_this_frame.index:
            ax.plot(df_this_frame['x'],df_this_frame['y'],'o',color='r',alpha=0.5)
        df_up_to_frame = b[b['frame']<=f]
        ax.plot(df_up_to_frame['x'],df_up_to_frame['y'],color='b',alpha=0.5)
        
    fig.savefig(figfolder+'frame_'+str(f)+'.png')