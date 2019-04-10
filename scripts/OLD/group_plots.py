# -*- coding: utf-8 -*-
"""
Created on Wed Oct 04 17:55:39 2017

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import fluids2d.piv as piv
import fluids2d.geometry
import pandas as pd
import pickle

csv_filepath = r'C:\Users\user\Documents\2d-fluids-analysis\piv_data_organizer.csv'
cases = pd.read_csv(csv_filepath)

cases = cases.loc[cases['use_now']==1]

amplitudes = cases['A'].unique()
freqs = cases['freq'].unique()

a_dict = {a:ai for ai,a in enumerate(np.sort(amplitudes))}
f_dict = {f:fi for fi,f in enumerate(np.sort(freqs))}


fig,ax = plt.subplots(len(freqs),len(amplitudes))
ax = np.atleast_2d(ax)

for case in cases.index:
    
    case_name = cases.loc[case,'file_name']
    
    ai = a_dict[cases.loc[case,'A']]
    fi = f_dict[cases.loc[case,'freq']]
    origin_y = cases.loc[case,'origin_y']
    
    parent_folder = cases.loc[case,'parent_folder']+'\\'
    
    need2rotate=bool(cases.loc[case,'need2rotate'])
    
    '''
    Load the data and make the scalers
    '''
    
    p = pickle.load(open(parent_folder+case_name+'.pkl'))
    p.parent_folder = parent_folder
    p.associate_flowfield()
    
    if need2rotate:
        p.data.ff=piv.rotate_data_90(p.data.ff)    
    ff = p.data.ff
    
    ff=piv.clip_flowfield(ff,0.5)
    
    g_orig = fluids2d.geometry.GeometryScaler(dx=p.dx,im_shape=(800,1280),origin_pos=(origin_y,0),origin_units='m')
    g = fluids2d.geometry.create_piv_scaler(p,g_orig)
    
    time = np.arange(0,np.shape(ff)[0]) * p.dt_frames
    
    print([ai,fi])
    print(cases.loc[case,'A'])
        
    piv.add_fieldimg_to_ax(np.nanmean(ff[:,:,:,1],axis=0),g,ax[fi,ai],time=None,slice_dir=None,vel_dir='vertical',vmin=-0.2,vmax=0.2)
    
    ax[fi,ai].set_title(cases.loc[case,'A'])
    
[a.set_ylim([0.0,0.42]) for a in ax.flatten()]
    