# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 13:12:39 2017

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dt = 1/2000.

parent_folder = r'E:\Experiments_Stephane\Grid column\Backlight_bubbles\20171014\\'


filenames = [r'Backlight_bubble_fps2000_svCloseUp_Needle_grid2tx20_noCap_Cam_20861_Cine1',
             r'Backlight_bubble_fps2000_svCloseUp_Needle_grid2tx20_noCap_Cam_20861_Cine2',
             r'Backlight_bubble_fps2000_svCloseUp_Needle_grid2tx20_noCap_Cam_20861_Cine3',
             r'Backlight_bubble_fps2000_svCloseUp_Needle_grid2tx20_noCap_Cam_20861_Cine4',
             r'Backlight_bubble_fps2000_svCloseUp_Needle_grid2tx20_noCap_Cam_20861_Cine5',
             r'Backlight_bubble_fps2000_svCloseUp_Needle_grid2tx20_noCap_Cam_20861_Cine6',
             r'Backlight_bubble_fps2000_svCloseUp_Needle_grid2tx20_noCap_Cam_20861_Cine7',
             r'Backlight_bubble_fps2000_svCloseUp_Needle_grid2tx20_noCap_Cam_20861_Cine8',
             r'Backlight_bubble_fps2000_svCloseUp_Needle_grid2tx20_noCap_Cam_20861_Cine9',
             r'Backlight_bubble_fps2000_svCloseUp_Needle_grid2tx20_noCap_Cam_20861_Cine10',
             r'Backlight_bubble_fps2000_svCloseUp_Needle_grid2tx20_noCap_Cam_20861_Cine11',
             r'Backlight_bubble_fps2000_svCloseUp_Needle_grid2tx20_noCap_Cam_20861_Cine12',
             r'Backlight_bubble_fps2000_svCloseUp_Needle_grid2tx20_noCap_Cam_20861_Cine13',
             r'Backlight_bubble_fps2000_svCloseUp_Needle_grid2tx20_noCap_Cam_20861_Cine14',
             r'Backlight_bubble_fps2000_svCloseUp_Needle_grid2tx20_noCap_Cam_20861_Cine15',
             r'Backlight_bubble_fps2000_svCloseUp_Needle_grid2tx20_noCap_Cam_20861_Cine16',
             r'Backlight_bubble_fps2000_svCloseUp_Needle_grid2tx20_noCap_Cam_20861_Cine17',
             r'Backlight_bubble_fps2000_svCloseUp_Needle_grid2tx20_noCap_Cam_20861_Cine18',
             r'Backlight_bubble_fps2000_svCloseUp_Needle_grid2tx20_noCap_Cam_20861_Cine19',
             r'Backlight_bubble_fps2000_svCloseUp_Needle_grid2tx20_noCap_Cam_20861_Cine20',
             r'Backlight_bubble_fps2000_svCloseUp_Needle_grid2tx20_noCap_Cam_20861_Cine21',
             r'Backlight_bubble_fps2000_svCloseUp_Needle_grid2tx20_noCap_Cam_20861_Cine22',
             r'Backlight_bubble_fps2000_svCloseUp_Needle_grid2tx20_noCap_Cam_20861_Cine23',
             r'Backlight_bubble_fps2000_svCloseUp_Needle_grid2tx20_noCap_Cam_20861_Cine24',
             r'Backlight_bubble_fps2000_svCloseUp_Needle_grid2tx20_noCap_Cam_20861_Cine25',
             r'Backlight_bubble_fps2000_svCloseUp_Needle_grid2tx20_noCap_Cam_20861_Cine26',
             r'Backlight_bubble_fps2000_svCloseUp_Needle_grid2tx20_noCap_Cam_20861_Cine27',
             r'Backlight_bubble_fps2000_svCloseUp_Needle_grid2tx20_noCap_Cam_20861_Cine28',
             r'Backlight_bubble_fps2000_svCloseUp_Needle_grid2tx20_noCap_Cam_20861_Cine29',
             r'Backlight_bubble_fps2000_svCloseUp_Needle_grid2tx20_noCap_Cam_20861_Cine30']

fig = plt.figure()
ax = fig.add_subplot(111)

for fi,f in enumerate(filenames):
    
    if fi<10:
        c = 'k'
    elif fi<20:
        c = 'g'
    elif fi<30:
        c = 'r'
    
    dfs = pd.read_pickle(parent_folder+f+r'_trackedParticles.pkl')
    
    for df in list(dfs.values()):
        
        ref_indx = df.index[df['y']>=0.01]
        
        
        if ref_indx.empty==False:
            time = (df.index-ref_indx[0]) * dt
            ax.plot(time,df['y'],color=c)
            
        #ax.plot(time[df['y']>0.02],df[df['y']>0.02]['orientation'],color=c)