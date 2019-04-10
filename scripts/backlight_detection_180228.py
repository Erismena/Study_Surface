# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 19:03:44 2017

@author: danjr
"""

import matplotlib.pyplot as plt
import numpy as np
import fluids2d.geometry
import fluids2d.backlight as backlight
import pims
import pandas as pd
import trackpy as tp

    
#folder = r'\\Mae-deike-lab3\c\Users\Luc Deike\data_comp3_C\180302\\'
#folder = r'\\Mae-deike-lab3\c\Users\Luc Deike\data_comp3_C\180323\\'
#cine_name = r'backlight_topDown_bubblesRising_smallNotTinyNeedle_2x2x2_on100_off400_fps1000_withCap'
#df_filepath = folder+cine_name+'_bubbles.pkl'
#
#c = pims.open(folder+cine_name+'.cine')

c = pims.open(r'C:\Users\Luc Deike\Nicolas\StudySurface\Trials\3\Following_Cluster\\*.tiff')

thresh = 370
dt = 1./1000
#dx= 9.4884764743E-05
#dx = 0.00020049103322
dx = 0.000225319221818

# CHANGES MADE BY NICOLAS

def mask(im):
    return im

g = fluids2d.geometry.GeometryScaler(dx=dx,im_shape=np.shape(mask(c[0])),origin_pos=(0,0),origin_units='pix')

frames = np.arange(0,46888,1)

df_all = backlight.run_bubble_detection(c,thresh,g,frames=frames,mask=None,viz=True,filepath=df_filepath)

df_all = pd.read_pickle(df_filepath)
df_filtered = df_all[df_all['radius']>0.0001]
df_filtered = df_filtered[df_filtered['radius']<0.02]
tracked = tp.link_df(df_filtered,search_range=0.002,memory=20)

tracked.to_pickle(folder+cine_name+r'_tracked.pkl')

'''
Make a list of dfs each corresponding to one bubble
'''
bubbles = backlight.make_list_of_bubbles(tracked,min_length=30)
bubbles = [backlight.interp_df_by_frame(b,dt) for b in bubbles]


fig = plt.figure()
ax = fig.add_subplot(111)
tp.plot_traj(tracked,ax=ax,colorby='particle')
ax.invert_yaxis()

fig = plt.figure()
ax = fig.add_subplot(111)
[ax.plot(b.index,b['eccentricity']) for b in bubbles]