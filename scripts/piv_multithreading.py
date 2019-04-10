# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 10:10:49 2018

@author: Luc Deike
"""

import fluids2d.piv as piv
import multiprocessing as mp
import time
import numpy as np
import matplotlib.pyplot as plt


import sys

num_jobs=16

#dx = 0.000203704906811 # 180121
#dx = 0.000204237510653 # 180121
#dx = 0.000188330837785 # 180201
#dx = 0.000211159351906 # 180320, center plane
#dx = 0.00020808761871 # 180320, half back
#dx = 0.000204940026359 # 180320, quarter back
# dx = 0.000211518252846 # 180322, three quarters back
#dx = 0.0002133401873 # 180322, full back
dx_a = 0.000201818276556 # 180323, top plane
dx_b = 0.000215974661611 # 180323, middle plane
dx = (dx_a+dx_b)/2.
parent_folder = r'\\Mae-deike-lab3\c\Users\Luc Deike\data_comp3_C\180323\\'
cine_name = r'piv_sunbathing_topDown_halfUpFromMiddlePlane_on100_off400'

ji = range(num_jobs)

per_job = 49784/num_jobs
a_frames = [np.arange((i)*per_job,(i+1)*per_job,2) for i in ji]

def worker(i):
    """worker function"""
    print 'Worker'
    
    name_for_save = cine_name+'_job'+str(i)
    #crop_lims = [0,2160,750,3800]
    crop_lims=None
    #name_for_save = cine_name+'_job'+str(i)
    processing = piv.PIVDataProcessing(parent_folder,cine_name,name_for_save=name_for_save,dx=dx,dt_orig=1./1000,frame_diff=1,crop_lims=crop_lims,maskers=None,window_size=32,overlap=16,search_area_size=32)
    processing.run_analysis(a_frames=a_frames[i],save=True,s2n_thresh=1.3)
    return

def concat_jobs(piv_list,cine_name):
    
    p0=piv_list[0]
    p0.associate_flowfield()
    p = piv.PIVDataProcessing(p0.parent_folder,cine_name,dx=p0.dx,dt_orig=p0.dt_orig,frame_diff=p0.frame_diff,crop_lims=p0.crop_lims,maskers=p0.maskers,window_size=p0.window_size,overlap=p0.overlap,search_area_size=p0.search_area_size)

    p.dt_frames = p0.dt_frames
    p.dt_ab = p0.dt_ab
    p.window_coordinates = p0.window_coordinates
    p.s2n_thresh = p0.s2n_thresh
    p.flow_field_res_filepath = p.parent_folder+p.cine_name+'_flowfield.npy'
    p.name_for_save = cine_name
    
    '''
    Initialize values that will be updated with each job
    '''
    p.a_frames = np.array([])
    p.a_frame_times = []
    
    p0.associate_flowfield()
    ffshape = list(np.shape(p0.data.ff))
    ffshape[0] = 0
    #per_job = ffshape[0]
    #ffshape[0]=per_job*len(piv_list)
    
    ff = np.zeros(ffshape)
    for pi,pj in enumerate(piv_list):
        p.a_frames = np.concatenate([p.a_frames,pj.a_frames])
        p.a_frame_times.append(pj.a_frame_times)
        
        print(np.shape(ff))
        print(np.shape(pj.data.ff))

        pj_ff = np.load(pj.parent_folder+pj.name_for_save+'_flowfield.npy')
        ff = np.concatenate([ff.copy(),pj_ff.copy()],axis=0)

        
    np.save(p.parent_folder+p.name_for_save+'_flowfield.npy',ff)
    p.save()
    
    return p

if __name__ == '__main__':
        
    if False:
        jobs = []
        for i in ji:
            p = mp.Process(target=worker,args=(i,))
            jobs.append(p)
            p.start()
            
    if True:
                    
        name_list = [cine_name+r'_job'+str(i) for i in range(num_jobs)]
        
        piv_list = [piv.load_processed_PIV_data(parent_folder,name) for name in name_list]
        p = concat_jobs(piv_list,cine_name)