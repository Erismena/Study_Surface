# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 16:12:12 2018

@author: user
"""

import matplotlib.pyplot as plt
import numpy as np
import fluids2d.backlight as backlight
import fluids2d.geometry as geometry
import fluids2d.spectra as spectra
import pandas as pd
import scipy.interpolate

n_groups = 5

figfolder = r'\\Mae-deike-lab3\c\Users\Luc Deike\Documents\multi_orientation_figures\\'

#folders = [r'\\Mae-deike-lab3\c\Users\Luc Deike\data_comp3_C\180302\\']*2 + [r'\\Mae-deike-lab3\c\Users\Luc Deike\data_comp3_C\180309\\']*2
folders = [r'\\Mae-deike-lab3\c\Users\Luc Deike\data_comp3_C\180320\\'] + [r'\\Mae-deike-lab3\c\Users\Luc Deike\data_comp3_C\180319\\'] + [r'\\Mae-deike-lab3\c\Users\Luc Deike\data_comp3_C\180322\\']*2 + [r'\\Mae-deike-lab3\d\data_comp3_D\180305\\']*2 + [r'\\Mae-deike-lab3\c\Users\Luc Deike\data_comp3_C\180309\\']*2 

names = [r'backlight_bubblesRising_tinyNeedle_noTurbulence_fps1000',
         r'backlight_bubblesRising_tinyNeedle_2x2x2_on100_off400_fps1000_withCap',
         r'backlight_bubblesRising_smallNotTinyNeedle_noTurbulence_fps1000',
         r'backlight_bubblesRising_smallNotTinyNeedle_2x2x2_on100_off400_fps1000_withCap',
         r'backlight_bubblesRising_noTurbulence_fps1000',
         r'backlight_bubblesRising_2x2x2_on100_off400_fps1000_withCap',
         r'backlight_bubblesRising_largeNeedle_noTurbulence_fps1000',
         r'backlight_bubblesRising_largeNeedle_2x2x2_on100_off400_fps1000_withCap']

dts = [0.001]*8

dxs = [0.000176940825543]*2 + [0.000225319221818]*2 + [0.00020049103322]*2 + [0.000222484436389]*2

offsets = [(0,0)]*8

fig,axs = plt.subplots(4,2,figsize=(17,9),sharex=True,sharey=True); axs = axs.flatten()
fig_md,axs_md = plt.subplots(4,2,figsize=(9,7),sharex=True,sharey=True); axs_md = axs_md.flatten()
fig_md_x,axs_md_x = plt.subplots(4,2,figsize=(9,7),sharex=True,sharey=True); axs_md_x = axs_md_x.flatten()
fig_dist_v,axs_dist_v = plt.subplots(4,2,figsize=(9,7),sharex=True,sharey=True); axs_dist_v = axs_dist_v.flatten()
fig_dist_u,axs_dist_u = plt.subplots(4,2,figsize=(9,7),sharex=True,sharey=True); axs_dist_u = axs_dist_u.flatten()
fig_dist_r,axs_dist_r = plt.subplots(4,2,figsize=(9,7),sharex=True,sharey=True); axs_dist_r = axs_dist_r.flatten()
fig_spatial_v,axs_spatial_v = plt.subplots(4,2,figsize=(9,7),sharex=True,sharey=True); axs_spatial_v = axs_spatial_v.flatten()



pumps=[False,True,False,True,False,True,False,True]
needle = [0,0,0.25,0.25,0.5,0.5,1,1]

fft_res = []

fft_quantities = ['u','v','orientation','eccentricity','direction','slip']

descriptions = ['tiny','tiny, turbulence','small','small, turbulence','med','med, turbulence','large','large, turbulence',]

all_meta = pd.DataFrame()

bi = 0

urms = 0.1

groups_of_bubbles = []

bins = np.linspace(0,.2,201)
dy_bins = bins[1]-bins[0]

min_length = 150

for i,(folder,name,dt,dx) in enumerate(zip(folders,names,dts,dxs)):
    tracked = pd.read_pickle(folder+name+'_tracked.pkl')

    bubbles = backlight.make_list_of_bubbles(tracked,min_length=min_length)
    bubbles = [backlight.interp_df_by_frame(b,dt) for b in bubbles]
    
    ax = axs[i]
    ax_md = axs_md[i]
    ax_md_x = axs_md_x[i]
    
    #[ax.plot(b['x'],b['y']) for b in bubbles]

    f_list = []
    freq_list = []
    
    ffts_by_bubble = []
    #xy = []
    v_vals_dict = {dyb:[] for dyb in bins}
    u_vals_dict = {dyb:[] for dyb in bins}
    
    mean_displacement_list = []
    mean_displacement_list_x = []
    lags_list = []

    for b in bubbles:     
        
        if (b['y'].max() - b['y'].min() > 0.05) and (b['radius'].median()>0.0004) and (b['y'].iloc[0] < b['y'].iloc[-1]) and (b['x'].min()>0.04) and (b['x'].max()<0.14):
            
            b = backlight.compute_transient_quantities(b,roll_vel=3)
            
            print(bi)
            
            #ax.plot(b['x'],b['y'])
            #xy.append([b['x'],b['y']])
            
            #b = b.iloc[0:600]
        
            all_meta.loc[bi,'med_radius'] = b['radius'].median()
            all_meta.loc[bi,'mean_radius'] = b['radius'].mean()
            all_meta.loc[bi,'std_radius'] = b['radius'].std()
            all_meta.loc[bi,'med_u'] = b['u'].abs().median()
            all_meta.loc[bi,'med_v'] = b['v'].median()
            all_meta.loc[bi,'mean_u'] = b['u'].abs().mean()
            all_meta.loc[bi,'mean_v'] = b['v'].mean()
            all_meta.loc[bi,'std_v'] = b['v'].std()
            all_meta.loc[bi,'std_u'] = b['u'].std()
            all_meta.loc[bi,'std_x'] = b['x'].std()
            all_meta.loc[bi,'std_orientation'] = b['orientation'].std()
            all_meta.loc[bi,'std_direction'] = b['direction'].std()
            all_meta.loc[bi,'len'] = len(b)        
            all_meta.loc[bi,'pumps'] = pumps[i]
            all_meta.loc[bi,'needle'] = needle[i]
            all_meta.loc[bi,'expmeanlog_ecc'] = np.exp(np.mean(np.log(b['eccentricity'])))
            all_meta.loc[bi,'mean_ecc'] = np.mean(b['eccentricity'])
            all_meta.loc[bi,'range_y'] = b['y'].max() - b['y'].min()
            
            #ax.plot(b.index,b['orientation'])
            #ax.plot(b.index,b['direction'])
            ax.plot(b['x'],b['y'],alpha=0.4)
            
            '''
            Bin the normalized vertical velocity values by vertical position,
            and add to the dict for the entire set of bubbles
            '''
            b['v_norm'] = b['v'] - b['v'].mean()
            b['u_norm'] = b['u']# / b['v'].mean()
            b['y_bin'] = [min(bins[bins>=y]) for y in b['y']]
            for ix in b.index:
                v_vals_dict[b.loc[ix,'y_bin']].append(b.loc[ix,'v_norm'])
                u_vals_dict[b.loc[ix,'y_bin']].append(b.loc[ix,'u_norm'])
            
            '''
            FFTs
            '''
            fft_dict = {q:[] for q in fft_quantities}
            for q in fft_quantities:            
                f,freq = spectra.abs_fft(b[q].values-b[q].mean(),b.index)
                # get rid of lowest frequency, which is zero since we removed the mean before taking the fft
                f = f[1:]; freq = freq[1:]
                f_list.append(f)
                freq_list.append(freq)
                fft_dict[q] = (f,freq)
            ffts_by_bubble.append(fft_dict)
            
            
            '''
            KDE plots
            '''            
            v_norm = b['v'] / b['v'].mean()
            v_norm.plot.kde(ax=axs_dist_v[i],alpha=0.2)
            
            u_norm =  b['u'] / b['u'].mean()
            u_norm.plot.kde(ax=axs_dist_u[i],alpha=0.2)
            
            r_norm = b['radius'] / b['radius'].mean()
            r_norm.plot.kde(ax=axs_dist_r[i],alpha=0.2)
            
            '''
            MSD
            '''
            lags = np.arange(0,min(len(b),1000)-2,1)
            mean_displacement = np.zeros(np.shape(lags))
            mean_displacement_x = np.zeros(np.shape(lags))
            
            for li,lag in enumerate(lags):
                mean_displacement[li] = np.sqrt(np.nanmean((b['x'].values[0:len(b)-lag]-b['x'].values[lag:len(b)])**2 + (b['y'].values[0:len(b)-lag]-b['y'].values[lag:len(b)])**2))
                mean_displacement_x[li] = np.sqrt(np.nanmean((b['x'].values[0:len(b)-lag]-b['x'].values[lag:len(b)])**2))
                
            lags_list.append(lags)
            mean_displacement_list.append(mean_displacement)
            mean_displacement_list_x.append(mean_displacement_x)
            
            bi = bi+1
            
    # create a list of median v values at each y value.
    mean_v_by_y = np.array([np.nanmean(np.array(v_vals_dict[y_bin])) for y_bin in sorted(list(v_vals_dict.keys()))])
    std_v_by_y = np.array([np.nanstd(np.array(v_vals_dict[y_bin])) for y_bin in sorted(list(v_vals_dict.keys()))])
    axs_spatial_v[i].plot(mean_v_by_y,bins,color=[0,0,1])
    axs_spatial_v[i].plot(mean_v_by_y+std_v_by_y,bins,'--',color=[0,0,1,0.4])
    axs_spatial_v[i].plot(mean_v_by_y-std_v_by_y,bins,'--',color=[0,0,1,0.4])
    
    mean_u_by_y = np.array([np.nanmean(np.array(u_vals_dict[y_bin])) for y_bin in sorted(list(u_vals_dict.keys()))])
    std_u_by_y = np.array([np.nanstd(np.array(u_vals_dict[y_bin])) for y_bin in sorted(list(u_vals_dict.keys()))])
    axs_spatial_v[i].plot(mean_u_by_y,bins,color=[1,0,0])
    axs_spatial_v[i].plot(mean_u_by_y+std_u_by_y,bins,'--',color=[1,0,0,0.4])
    axs_spatial_v[i].plot(mean_u_by_y-std_u_by_y,bins,'--',color=[1,0,0,0.4])
    
    plt.show()
    plt.pause(1)
            
    #xy = random.shuffle(xy)
    #groups = []
    #for n in range(n_groups):
    #   groups.append([xy[xyi] for xyi in range(n*len(xy)/n_groups,(n+1)*len(xy)/n_groups)])
    
    fft_res.append(ffts_by_bubble)
    
    #ax.set_aspect('equal')
    

        
        
    [ax_md.loglog(l*dt,md**2/urms**2,alpha=0.2) for l,md in zip(lags_list,mean_displacement_list)]
    [ax_md_x.loglog(l*dt,md**2/urms**2,alpha=0.2) for l,md in zip(lags_list,mean_displacement_list_x)]
    
    '''
    Autocorrelation
    '''
    
[a.axvline(0,color='gray',alpha=0.4) for a in axs_spatial_v]
[a.set_xlim([-0.2,0.2]) for a in axs_spatial_v]

axs_spatial_v[0].plot(np.nan,np.nan,color='r',label='horizontal')
axs_spatial_v[0].plot(np.nan,np.nan,color='b',label='vertical (diff. from mean)')
axs_spatial_v[0].legend()

axs_spatial_v[0].set_title('no pumps')
axs_spatial_v[1].set_title('pumps')

axs_spatial_v[0].set_ylabel('tiny needle \n vertical position [m]')
axs_spatial_v[2].set_ylabel('med needle \n vertical position [m]')
axs_spatial_v[4].set_ylabel('large needle \n vertical position [m]')

axs_spatial_v[4].set_xlabel('velocity [m/s]')
axs_spatial_v[5].set_xlabel('velocity [m/s]')

fig_spatial_v.savefig(figfolder+'velocity_maps.pdf')




#[a.axvline(1,color='gray',alpha=0.4) for a in axs_spatial_v]
fig_spatial_v.tight_layout()

    
axs_md[0].plot([.01,.1],[.002,.2],'--',color='k',alpha=0.5)
    
axs_md[0].set_ylabel('tiny needle \n MSD / u_ms [s^2]')
axs_md[2].set_ylabel('med needle \n MSD / u_ms [s^2]')
axs_md[4].set_ylabel('large needle \n MSD / u_ms [s^2]')


axs_md[4].set_xlabel('lag [s]')
axs_md[5].set_xlabel('lag [s]')

axs_md[0].set_title('quiescent')
axs_md[1].set_title('turbulence')


axs_md_x[0].plot([.01,.1],[.002,.2],'--',color='k',alpha=0.5)
axs_md_x[0].plot([.1,1],[.2,2],'--',color='k',alpha=0.5)

axs_md_x[0].set_ylabel('tiny needle \n MSD_x / u_ms [s^2]')
axs_md_x[2].set_ylabel('med needle \n MSD_x / u_ms [s^2]')
axs_md_x[4].set_ylabel('large needle \n MSD_x / u_ms [s^2]')


axs_md_x[4].set_xlabel('lag [s]')
axs_md_x[5].set_xlabel('lag [s]')

axs_md_x[0].set_title('quiescent')
axs_md_x[1].set_title('turbulence')

[a.set_ylim([10**-7,10]) for a in axs_md]
[a.set_ylim([10**-7,10]) for a in axs_md_x]
[a.set_xlim([10**-3,1]) for a in axs_md]
[a.set_xlim([10**-3,1]) for a in axs_md_x]

fig_md.tight_layout()
fig_md_x.tight_layout()

avgd = [ np.mean(np.array([x[i] for x in mean_displacement_list_x if (len(x)-500)>i ])) for i in range(1000)]
plt.figure(); plt.loglog(avgd)

mdict = {0:'^',.25:'s',.5:'o',1:'+'}
cdict = {True:'r',False:'b'}

def rolling_by_x(df,by,window=20):    
    df_r = df.sort_values(by=by,inplace=False).rolling(window=window,center=True,min_periods=0).mean()
    return df_r
    
def scatter_plot(df,x,y):
    fig = plt.figure(figsize=(9,5))
    ax = fig.add_subplot(111)
    ax.plot(np.nan,np.nan,'-',color='r',label='pumps')
    ax.plot(np.nan,np.nan,'-',color='b',label='no pumps')
    ax.plot(np.nan,np.nan,'^',color='k',label='tiny needle')
    ax.plot(np.nan,np.nan,'s',color='k',label='small needle')
    ax.plot(np.nan,np.nan,'o',color='k',label='med needle')
    ax.plot(np.nan,np.nan,'+',color='k',label='large needle')
    ax.legend()
    #for i in all_meta.index:
    #    xs = np.array([all_meta.loc[i,'med_radius']-all_meta.loc[i,'std_radius'],all_meta.loc[i,'med_radius']+all_meta.loc[i,'std_radius']])*1000
    #    ys = np.array([all_meta.loc[i,'med_v']-all_meta.loc[i,'std_v'],all_meta.loc[i,'med_v']+all_meta.loc[i,'std_v']])*100
    #    ax.plot(xs,np.array([all_meta.loc[i,'med_v']]*2)*100,color='k',alpha=0.2)
    #    ax.plot(np.array([all_meta.loc[i,'med_radius']]*2)*1000,ys,color='k',alpha=0.2)
    for n in all_meta['needle'].unique():
        this_meta = all_meta[all_meta['needle']==n]
        ax.scatter(this_meta[x],this_meta[y],c=[cdict[p] for p in this_meta['pumps']],marker=mdict[n],alpha=0.5)
    for p in all_meta['pumps'].unique():
        this_meta = all_meta[all_meta['pumps']==p]
        this_meta[y].iloc[:] = this_meta[y].values* this_meta['len'].values
        rolling = rolling_by_x(this_meta,x)
        rolling[y].iloc[:] = rolling[y].div(rolling_by_x(this_meta,x)['len'],axis=0)
        ax.plot(rolling[x],rolling[y],c=[0,1,0])    

    return fig,ax

fig,ax = scatter_plot(all_meta,'mean_radius','mean_v')
ax.set_ylabel('mean rise velocity [m/s]')
ax.set_xlabel('mean radius [m]')
fig.tight_layout()
fig.savefig(figfolder+'v_vs_r.pdf')

fig,ax = scatter_plot(all_meta,'med_radius','std_v')
ax.set_ylabel('standard deviation of v [m/s]')
ax.set_xlabel('median radius [m]')
fig.tight_layout()
fig.savefig(figfolder+'med_r_vs_std_v.pdf')

fig,ax = scatter_plot(all_meta,'med_radius','std_x')
ax.set_ylabel('standard deviation of x [m]')
ax.set_xlabel('median radius [m]')
fig.tight_layout()

fig,ax = scatter_plot(all_meta,'std_radius','std_u')
ax.set_ylabel('standard deviation of u [m/s]')
ax.set_xlabel('standard deviation of r [m]')
fig.tight_layout()

fig,ax = scatter_plot(all_meta,'med_radius','expmeanlog_ecc')
ax.set_ylabel('exp mean log eccentricity [-]')
ax.set_xlabel('median radius [m]')
fig.tight_layout()

fig,ax = scatter_plot(all_meta,'med_radius','mean_ecc')
ax.set_ylabel('mean eccentricity [-]')
ax.set_xlabel('median radius [m]')
fig.tight_layout()


fig,ax = scatter_plot(all_meta,'med_radius','mean_radius')
ax.set_ylabel('mean radius [m]')
ax.set_xlabel('median radius [m]')
fig.tight_layout()

fig,ax = scatter_plot(all_meta,'med_v','mean_v')
ax.set_ylabel('mean vertical velocity [m/s]')
ax.set_xlabel('median vertical velocity [m/s]')
fig.tight_layout()
    
fft_plot = ['u','v','orientation','direction']
fig,axs = plt.subplots(len(fft_plot),1,sharex=True,figsize=(7,9))
for i in range(len(names)):
    
    for qi,q in enumerate(fft_plot):
        
        ax = axs[qi]
            
        f_list = [x[q][0] for x in fft_res[i]]
        freq_list = [x[q][1] for x in fft_res[i]]
        
        all_freq = []
        [all_freq.append(freq.tolist()) for freq in freq_list]
        all_freq = [item for sublist in all_freq for item in sublist]
        
        all_f = []
        [all_f.append(f.tolist()) for f in f_list]
        all_f = [item for sublist in all_f for item in sublist]
        
        all_s = pd.Series(index=all_freq,data=all_f).sort_index()
        ax.loglog(all_s.index,all_s.rolling(window=len(all_s)/1000,center=True,min_periods=0).median(),label=descriptions[i],alpha=0.5)
        
        
        interp_freqs = np.logspace(np.log(min(all_f)),np.log(max(all_f)),1001)
        
        #interpr = scipy.interpolate.interp1d(all_s.index,all_s.values,bounds_error=False)
        #interpd = interpr(interp_freqs)
        
        #interpr = scipy.interpolate.UnivariateSpline(all_s.index,all_s.values,k=1)
        #interpd = interpr(interp_freqs)
        
        #
        #ax.loglog(interp_freqs,interpd,label=descriptions[i])
        
        ax.set_ylabel(q)
        
axs[1].loglog()

axs[0].legend()
axs[qi].set_xlabel('frequency [Hz]')
fig.tight_layout()
fig.savefig(figfolder+'bubbles_ffts.pdf')

#fig,axs = plt.subplots(2,2,figsize=(9,9)); axs = axs.flatten()
#for n in range(n_groups):
#    for c