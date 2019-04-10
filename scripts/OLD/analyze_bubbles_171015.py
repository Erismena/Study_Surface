# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 13:12:39 2017

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

dt = 1/500.
dx = 0.000151377290739

parent_folder = r'E:\Experiments_Stephane\Grid column\Backlight_bubbles\20171015\\'

filenames = [r'Backlight_bubble_fps500_sv_Needle_grid3tx2x20_noCap_Cam_20861_Cine'+str(i) for i in range(1,61)]
#filenames = [r'backlight_bubbles_sv_grid3tx2x20_Cam_20861_Cine'+str(i) for i in range(1,61)]

metadata = pd.DataFrame(index=np.arange(1,61))
metadata['filename'] = filenames
metadata['A'] = [0]*10+[2]*5+[4]*5+[6]*5+[8]*5+[10]*5+[1]*5+[2]*5+[4]*5+[6]*5+[8]*5
metadata['freq'] = [0]*10+[5]*25+[10]*25
#metadata['A'] = [0]*20 + [2]*20 + [3]*20
#metadata['freq'] = [10]*20 + [10]*20 + [10]*20


shape_dict = {0:'o',5:'+',10:'x'}
color_dict = {0:'k',5:'b',10:'r'}
ls_dict = {0:'-',5:'--',10:'-.'}

all_bubbles = pd.DataFrame(columns=list(metadata.columns)+['med_u','med_v','med_radius'])


def multi_color_plot(ax,x,y,c,*kwargs):
    x = np.array(x)
    y = np.array(y)
    for i in range(0,len(x)-1):
        ax.plot([x[i],x[i+1]],[y[i],y[i+1]],color=c[i])
        
def area_to_radius(area_pix):
    '''
    A = pi*r^2
    r = sqrt(A/pi)
    
    A = A_pix * dx^2
    '''    
    return np.sqrt(area_pix*dx**2/np.pi) * 1000
    


#fig,ax = plt.subplots(1,3,sharex=True,sharey=True)
#ax = ax.flatten()
#ax_dict = {0:ax[0],5:ax[1],10:ax[2]}

fig = plt.figure(figsize=(9,8))
ax = fig.add_subplot(111)

bottom = 0.125
top = 0.225

bi = -1

speed_min = 0.20
speed_max = 0.25

scaling = 13
norm = matplotlib.colors.Normalize(vmin=speed_min,vmax=speed_max)
norm_area = matplotlib.colors.Normalize(vmin=1,vmax=2.5)

df_dict = {}

for fi in metadata.index:
    
    f = metadata.loc[fi,'filename']
    
    dfs = pd.read_pickle(parent_folder+f+r'_trackedParticles.pkl')
    
    #a = ax_dict[metadata.loc[fi,'freq']]
    
    
    if fi in [1,11,16,21,26,31,36,41,46,51,56]:
        di=-1
        
    for df in list(dfs.values()):
        
        time = df.index * dt
        
        ref_indx = df.index[df['y']>=bottom]
        ref_indx2 = df.index[df['y']>=top]
        
        if (ref_indx.empty==False) & (ref_indx2.empty==False):      
            
            df = df[(df.index>=ref_indx[0])&(df.index<=ref_indx2[0])]
            
            if len(df)>50:
                print(len(df))
                
                #df['filled_area'] = area_to_radius(df['filled_area'])              
                df_dict[bi] = df.copy()
                
                di = di+1
                bi = bi+1
                    
                all_bubbles.loc[bi,'med_u'] = (df['x'].diff()/dt).median()
                all_bubbles.loc[bi,'med_v'] = (df['y'].diff()/dt).median()
                #all_bubbles.loc[bi,'med_area'] = area_to_radius(df['filled_area'].median())
                all_bubbles.loc[bi,'med_radius'] = df['radius'].median()
                
                [all_bubbles.set_value(bi,col,metadata.loc[fi,col]) for col in list(metadata.columns)]
                
                x_shift = metadata.loc[fi,'freq'] + (float(di)*.2)*((-1)**di)
                #print(x_shift)
                y_shift = metadata.loc[fi,'A'] - scaling*(top-bottom)/2 - scaling*bottom
                
                lw = norm_area(all_bubbles.loc[bi,'med_radius'],clip=True)*3+2
                #color = matplotlib.cm.get_cmap('PuRd')(norm_area(all_bubbles.loc[bi,'med_area'],clip=True))
                color = matplotlib.cm.get_cmap('viridis')(norm(all_bubbles.loc[bi,'med_v'],clip=True))
                #print(lw)
                
                speed = df['y'].diff().rolling(window=5,center=True,min_periods=0).mean()/dt
                #ax.plot(df['x']*scaling+x_shift,(df['y']-0.125)*scaling+y_shift,color=color,linewidth=lw,alpha=0.2) #,linewidth=lw
                ax.plot(df['x'].iloc[0]*scaling+x_shift,df['y'].iloc[0]*scaling+y_shift,'o',color=color,markersize=2+norm_area(all_bubbles.loc[bi,'med_radius'],clip=True)*5,markeredgecolor='r',markeredgewidth=0.4)
                cax = ax.scatter(df['x']*scaling+x_shift,df['y']*scaling+y_shift,color=matplotlib.cm.get_cmap('viridis')(norm(speed,clip=True)),s=0.3,alpha=0.5) #,linewidth=lw
                #cax = ax.plot(df['x']*scaling+x_shift,df['y']*scaling+y_shift,color=matplotlib.cm.get_cmap('viridis')(norm(all_bubbles.loc[bi,'med_v'],clip=True)),alpha=0.5)
                #multi_color_plot(ax,df['x']*scaling+x_shift,df['y']*scaling+y_shift,matplotlib.cm.get_cmap('viridis')(norm(speed,clip=True)))
                
ax.plot([3,3],[0-(top-bottom)/2*scaling,0+(top-bottom)/2*scaling],color='k')
ax.text(3.15,0,'10 cm',horizontalalignment='left',verticalalignment='center',color='k')

cax = fig.add_axes([0.52,0.19,0.20,0.015])
cax.set_visible(True)
cb = matplotlib.colorbar.ColorbarBase(cax, cmap=matplotlib.cm.get_cmap('viridis'),
                                norm=norm,
                                orientation='horizontal')
cb.set_label('vertical velocity [m/s]')
cb.ax.xaxis.set_label_position('top')

sizeax = fig.add_axes([0.50,0.12,0.17,0.03])
sizeax.plot([0],[0],'o',color='gray',markeredgecolor='r',markeredgewidth=0.4,markersize=2+norm_area(1,clip=True)*5); sizeax.text(0.1,0,'1 mm',verticalalignment='center')
sizeax.plot([1],[0],'o',color='gray',markeredgecolor='r',markeredgewidth=0.4,markersize=2+norm_area(2.5,clip=True)*5); sizeax.text(1.1,0,'2.5 mm',verticalalignment='center')
sizeax.set_xlim([-0.5,1.5])
sizeax.set_ylim([-1,1])
sizeax.set_axis_off()

ax.set_aspect('equal')    
ax.set_ylabel('frequency [Hz]')
ax.set_xlabel('amplitude [mm]')

ax.set_yticks(list(all_bubbles['A'].sort_values().unique()))
ax.set_xticks(list(all_bubbles['freq'].sort_values().unique()))
ax.set_title('Bubble Trajectories')

plt.tight_layout()

'''
Rise velocity vs frequency
'''
            
fig = plt.figure()
ax = fig.add_subplot(111)

grouped = all_bubbles.groupby(['A','freq'])
mean_dict = {f:[] for f in list(all_bubbles['freq'].unique())}
A_dict = {f:[] for f in list(all_bubbles['freq'].unique())}
for group in list(grouped):
    A,freq = group[0]
    data = group[1]
    mean_dict[freq].append(data['med_v'].mean())
    A_dict[freq].append(A)
    
for freq in list(mean_dict.keys()):
    
    ax.plot(A_dict[freq],mean_dict[freq],'-^',color=color_dict[freq],markersize=10,markeredgecolor=color_dict[freq],markerfacecolor='gray')
    
for bi in all_bubbles.index:
    ax.plot(all_bubbles['A'][bi],all_bubbles['med_v'][bi],color=color_dict[all_bubbles['freq'][bi]],marker='o',markersize=2+norm_area(all_bubbles.loc[bi,'med_radius'],clip=True)*5,alpha=0.8) # [shape_dict[freq] for freq in all_bubbles['freq']]
    
ax.set_xlabel('amplitude [mm]')
ax.set_ylabel('median vertical velocity [m/s]')

ax.plot(np.nan,np.nan,'o',color='r',label='10 Hz')
ax.plot(np.nan,np.nan,'o',color='b',label='5 Hz')
ax.legend()
    
'''
Size vs frequency
'''

fig = plt.figure()
ax = fig.add_subplot(111)

grouped = all_bubbles.groupby(['A','freq'])
mean_dict = {f:[] for f in list(all_bubbles['freq'].unique())}
A_dict = {f:[] for f in list(all_bubbles['freq'].unique())}
for group in list(grouped):
    A,freq = group[0]
    data = group[1]
    mean_dict[freq].append(data['med_radius'].mean())
    A_dict[freq].append(A)
    
for freq in list(mean_dict.keys()):
    
    ax.plot(A_dict[freq],mean_dict[freq],'-^',color=color_dict[freq],markersize=10,markeredgecolor=color_dict[freq],markerfacecolor='gray')
    
for bi in all_bubbles.index:
    ax.plot(all_bubbles['A'][bi],all_bubbles['med_radius'][bi],color=color_dict[all_bubbles['freq'][bi]],marker='o',markersize=2+norm_area(all_bubbles.loc[bi,'med_radius'],clip=True)*5,alpha=0.8) # [shape_dict[freq] for freq in all_bubbles['freq']]

ax.set_xlabel('amplitude [mm]')
ax.set_ylabel('bubble radius [mm]')

ax.plot(np.nan,np.nan,'o',color='r',label='10 Hz')
ax.plot(np.nan,np.nan,'o',color='b',label='5 Hz')
ax.legend()