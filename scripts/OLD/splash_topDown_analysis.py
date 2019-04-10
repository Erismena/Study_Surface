# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 21:44:55 2017

@author: Luc Deike
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pims

folder = r'D:\high_speed_data\171128\\'

heights_mm = [229,346,497,631]
cine_nums = np.arange(1,6)

def cine_name(height,num):
    return r'milkDrop_topDown_H'+str(height)+r'mm_pump0050uLMin_Cam_20861_Cine'+str(num)


meta = pd.DataFrame(columns=['filepath','height','num'])
for height in heights_mm:
    for num in cine_nums:
        name=cine_name(height,num)
        print(name)
        meta.loc[name,'filepath'] = folder+name+r'.cine'
        meta.loc[name,'height'] = height
        meta.loc[name,'num'] = num
        
        c = pims.open(meta.loc[name,'filepath'])
        
        i0=np.mean(c[0])
        f=0
        found_first_frame = False
        while found_first_frame==False:
            f = f+1
            i0 = np.max(c[f])
            if i0 > 300:
                found_first_frame=True
                
        meta.loc[name,'first_frame'] = f
            
        

frames = np.arange(50,2000,1)

for fi,f in enumerate(frames):
    
    plt.close('all')
    
    fig,axs = plt.subplots(len(heights_mm),len(cine_nums),figsize=(16,9))
    [ax.set_axis_off() for ax in axs.flatten()]
    
    for hi,h in enumerate(heights_mm):
        for ni,n in enumerate(cine_nums):
            name = cine_name(h,n)
            c = pims.open(meta.loc[name,'filepath'])
            axs[hi,ni].imshow(c[f+meta.loc[name,'first_frame']],cmap='gray',vmin=100,vmax=400)
            
            if ni==0:
                axs[hi,ni].text(50,50,'h = '+str(h)+' mm',color=[0,1,0])
            
    plt.tight_layout()
    fig.savefig(folder+r'frames\\frame_'+str(f)+'.png')