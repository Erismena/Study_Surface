# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 10:10:49 2018

@author: Luc Deike
"""

import fluids2d.backlight as backlight
import fluids2d.geometry
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import pims
import trackpy as tp
import pandas as pd

'''
Set stuff up
'''



folder = r'E:\Stephane\171114\\'
cine_name = r'balloon_breakup_pumps_fps10000_backlight_D400minch_d28mm'
df_filepath = folder+cine_name+'_bubbles.pkl'

c = pims.open(folder+cine_name+'.cine')
thresh = 480
dt = 1./10000
#dx= 9.4884764743E-05
#dx = 0.00020049103322
#dx = 0.000216023661265
dx = 0.000112656993755

def mask(im):
    return im

g = fluids2d.geometry.GeometryScaler(dx=dx,im_shape=np.shape(mask(c[0])),origin_pos=(0,0),origin_units='pix')

num_jobs = 16
ji = range(num_jobs)
#per_job = len(c)/num_jobs
#frames = [np.arange((i)*per_job,(i+1)*per_job,1) for i in ji]


all_frames = np.arange(2000,22000,20)
per_job = len(all_frames)/num_jobs

# Don't distribute the number of frames in each job evenly since the initial
# frames (with more bubbles) take longer to process
factor = 0.5
frames_spacing = np.geomspace(len(all_frames)*factor,len(all_frames)*(factor+1),num_jobs+1).astype(int)-int(len(all_frames)*factor)
#frames = [all_frames[i*per_job:(i+1)*per_job] for i in ji]
frames = [all_frames[max(0,frames_spacing[i]):frames_spacing[i+1]] for i in range(num_jobs)]

def worker(i):
    name_for_save = cine_name+'_job'+str(i)
    backlight.run_bubble_detection(c,thresh,g,frames=frames[i],mask=None,viz=False,filepath=folder+name_for_save+'.pkl',method='standard')
    return

def concat_jobs(name_list,cine_name,df_filepath):
    
    df_all = pd.DataFrame()
    for name in name_list:
        job_df = pd.read_pickle(folder+name+r'.pkl')
        df_all = pd.concat([df_all,job_df])
        
    df_all.index = np.arange(len(df_all))
    df_all = df_all.sort_values('frame')
    df_all.to_pickle(df_filepath)

    return df_all

if __name__ == '__main__':
    
    if True:
        jobs = []
        for i in ji:
            p = mp.Process(target=worker,args=(i,))
            jobs.append(p)
            p.start()
            
    if False:
                    
        name_list = [cine_name+r'_job'+str(i) for i in range(num_jobs)]
        
        df_all = concat_jobs(name_list,cine_name,df_filepath)
        
        df_filtered = df_all[df_all['radius']>0.0005]
        df_filtered = df_filtered[df_filtered['radius']<0.02]
        tracked = tp.link_df(df_filtered,search_range=0.002,memory=20,adaptive_stop=0.001)
        
        tracked.to_pickle(folder+cine_name+r'_tracked.pkl')