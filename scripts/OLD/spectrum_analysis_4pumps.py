# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 12:20:35 2018

@author: Luc Deike
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import fluids2d.spectra as spectra

figfolder = r'C:\Users\Luc Deike\Documents\dan_turbulence_project\figures\\'

C_arr = pickle.load(open(figfolder+'corr_T0250_onThird.pkl','rb'))

num_lags = 16000
keep_first=num_lags/2
favg= np.zeros((keep_first,2))

C_arr = C_arr 
#C_avg = np.nanmean(np.nanmean(C_arr,axis=1),axis=1)


for d in [0,1]: 
    C_avg[:,d] = np.nanmean(np.nanmean(C_arr[:,:,:,d]* np.nanmean(ff[:,center_rows[0]:center_rows[1]+1,center_cols[0]:center_cols[1]+1,0]**2,axis=0),axis=1),axis=1)
    favg[:,d],freq = spectra.temporal_spectrum_from_autocorr(C_avg[:,d],lags)
    
freq = freq*2*np.pi
    
fig=plt.figure()
ax = fig.add_subplot(111)
ax.loglog(freq,favg)

imin = np.argmin(np.abs(freq-10.))
imax = np.argmin(np.abs(freq-100.))

xuse = freq[imin:imax]
yuse = favg[imin:imax,1]

import scipy

def func(x,A): return A*x**(-5./3)

a,_=scipy.optimize.curve_fit(func,  xuse,  yuse)

x_show = [10.,100.]
y_show = a[0]*np.array(x_show)**(-5./3)

ax.loglog(x_show,y_show)

u_rms = 0.195

#epsilon = (a[0]/u_rms**(2./3))**(3./2)
epsilon = a[0]**(3./2) / u_rms