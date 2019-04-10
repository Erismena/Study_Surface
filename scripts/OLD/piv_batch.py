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
import os
import pickle
#import fluids2d
import fluids2d.masking

parent_folder = r'E:\Experiments_Stephane\Grid column\PIV_measurements\Hollow_grid_3\20170816\Cines_B\\'
#parent_folder = r'C:\Users\danjr\Documents\Fluids Research\Data\misc\PIV\\'
#cine_name1 = r'PIV_svCloseUpCenterThird_makro100mmzeiss_X0mm_Y15mm_fps2500_A15mm_f10Hz_grid3x4_10cycles'
#cine_name2 = r'PIV_svCloseUpCenterThird_makro100mmzeiss_X0mm_Y15mm_fps2500_A05mm_f10Hz_grid3x4_10cycles'

#cine_files = [fn for fn in os.listdir(parent_folder) if fn.endswith('.cine')]

cine_files = ['PIV_svCloseUpCenterThird_makro100mmzeiss_X0mm_Y15mm_fps2500_A15mm_f10Hz_grid3x4_10cycles',]

import fluids2d.piv

cine = pims.open(parent_folder+cine_files[0]+'.cine')

masker = fluids2d.masking.MovingRectMasks(250,motion_direction='vertical')
masker.calibrate_from_images('1',cine,frames_to_calibrate=11)
masker.calibrate_from_images('2',cine,frames_to_calibrate=11)

piv_list = [fluids2d.piv.PIVData(parent_folder,cine_name,maskers=[masker,]) for cine_name in cine_files]

def process_to_run(target,kwargs):
    
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            print(kwargs)
            target(**kwargs)
        except:
            []
        #except Warning as e:
        #    print('error found:', e)

P = piv_list[0]
P.run_analysis()


'''
from multiprocessing import Process
#process_list = [Process(target=piv.run_analysis,kwargs={'save':True,'crop_lims':None,'a_frames':None}) for piv in piv_list]
#kws = {'save':True,'crop_lims':[80,700,0,1280],'a_frames':np.arange(150,168)}
#process_list = [Process(target=process_to_run,args=(piv.run_analysis,kws,)) for piv in piv_list]
def func(save=True,crop_lims=None,a_frames=None):
    return piv_list[0].run_analysis(save=save,crop_lims=crop_lims,a_frames=a_frames)
#piv_list[0].run_analysis(save=True,crop_lims=[80,700,0,1280],a_frames=None)
p=Process(target=func,kwargs={'save':True,'crop_lims':None,'a_frames':None})
p.start()
'''

#piv_list[0].run_analysis(save=True,crop_lims=[80,700,0,1280],a_frames=np.arange(150,168))

#[p.start() for p in process_list]

##process_list[0].start()
#process_list[1].start()
#
#while True:
#    for p in process_list:
#        print(p)
#    plt.pause(1)