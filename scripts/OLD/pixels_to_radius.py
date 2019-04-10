# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 13:12:39 2017

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dt = 1/500.
dx = 0.000126336189127

parent_folder = r'E:\Experiments_Stephane\Grid column\Backlight_bubbles\20171028\\'
filenames = [r'backlight_bubbles_sv_grid3tx2x20_171028_noTurbulence_Cam_20861_Cine'+str(i) for i in range(1,61)]
        
def area_to_radius(area_pix):
    '''
    A = pi*r^2
    r = sqrt(A/pi)
    
    A = A_pix * dx^2
    '''    
    return np.sqrt(area_pix*dx**2/np.pi) * 1000
    
df_dict = {}

for f in filenames:
    
    
    dfs = pd.read_pickle(parent_folder+f+r'_trackedParticles.pkl')

        
    for di in list(dfs.keys()):
        
        df=dfs[di]
        
        if ('indx_orig' in df.columns)==False:
            df['indx_orig'] = df.index
        
        #time = df.index * dt
        df['time'] = df['indx_orig']*dt
        df = df.set_index(['time'])
        
        df['radius'] = area_to_radius(df['filled_area'])
        
        dfs[di] = df
            
    import pickle
    pickle.dump(dfs, open(parent_folder+f+'_trackedParticles.pkl', "wb"))