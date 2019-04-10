# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 10:09:51 2018
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pims
import scipy.ndimage
import pandas as pd
from scipy.ndimage.morphology import binary_fill_holes
import skimage.measure
import skimage.filters
import skimage.morphology
from matplotlib import cm
import trackpy as tp
import skimage
import cv2
from skimage.filters import threshold_local
import fluids2d.backlight as backlight
import fluids2d.geometry
from datetime import datetime

thresh=500


folder = r'E:\Stephane\171114\\'
cine_name = r'balloon_breakup_pumps_fps10000_backlight_D800minch_d20mm'

c = pims.open(folder+cine_name+'.cine')
im = c[12000][400:800,20:600]
dx = 0.000112656993755
g = fluids2d.geometry.GeometryScaler(dx=dx,im_shape=np.shape(im),origin_pos=(0,0),origin_units='m')

bg = backlight.find_bg_image(c)
plt.figure()
plt.imshow(bg)

stophere


print('--- random walker (red) ---')
start_rw = datetime.now()
rw = backlight.random_walker_detection(im,thresh,g,viz=False)
end_rw = datetime.now()
dur_rw = end_rw - start_rw
print(dur_rw)
df = backlight.labeled_props(rw,g)
vf = backlight.estimate_void_fraction(df)
print(vf)

print('--- watershed (green) ---')
start_ws = datetime.now()
ws = backlight.watershed_detection(im,thresh,g,RadiusDiskMean=1,viz=False) #[550:750,400:600]
end_ws = datetime.now()
dur_ws = end_ws - start_ws
print(dur_ws)
df_ws = backlight.labeled_props(ws,g)
vf = backlight.estimate_void_fraction(df_ws)
print(vf)

print('--- standard (blue) ---')
start_standard = datetime.now()
filled = backlight.get_filled(im,thresh)
df_standard = backlight.filled2regionpropsdf(filled,g=g)
end_standard = datetime.now()
dur_standard = end_standard - start_standard
print(dur_standard)
vf = backlight.estimate_void_fraction(df_standard)
print(vf)

#
#df = backlight.labeled_props(rw,g)
#vf = backlight.estimate_void_fraction(df)
#print(vf)
ax = backlight.show_and_annotate(im,g,df,ax=None,vmin=0,vmax=600)
backlight.add_ellipses_to_ax(df_ws,ax,color=[0,1,0,0.5])
backlight.add_ellipses_to_ax(df_standard,ax,color=[0,0,1,0.5])


#df_all = backlight.run_bubble_detection(c,thresh,g,frames=frames,method='watershed')
