# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 23:28:34 2017

@author: danjr
"""

'''
Run multiple PIV simultaneously using the multiprocessing package
'''

import fluids2d.piv as piv
import fluids2d.masking
from fluids2d.masking import MovingRectMasks
import numpy as np

parent_folder = r'E:\Experiments_Stephane\Grid column\PIV_measurements\Hollow_grid_3\20171015\\'

files = [r'PIV_sv_fps2k_grid3tx2x20_noCap_Cam_20861_Cine5',
         r'PIV_sv_fps2k_grid3tx2x20_noCap_Cam_20861_Cine10',
         r'PIV_sv_fps2k_grid3tx2x20_noCap_Cam_20861_Cine19',
         r'PIV_sv_fps2k_grid3tx2x20_noCap_Cam_20861_Cine20']

dx_topView = 0.0001132022
dx_bottomView = 0.000113469584153

dx = [dx_topView,dx_topView,dx_bottomView,dx_bottomView]

crop_lims = [None,None,None,None]

#crop_lims = [0,-1,200,1040]

piv_list = []
for fi,f in enumerate(files):
    #masker = fluids2d.masking.load_masker(parent_folder+f+'.masker')
    p = piv.PIVDataProcessing(parent_folder,f,dx=dx[fi],dt_orig=1./2000,frame_diff=1,crop_lims=crop_lims[fi],maskers=None,window_size=24,overlap=12,search_area_size=24)
    piv_list.append(p)
    
def run_analysis(p):
    return p.run_analysis(a_frames=np.arange(0,2482,2))
    
from multiprocessing.dummy import Pool as ThreadPool 
pool = ThreadPool(4) 
results = pool.map(run_analysis, piv_list)