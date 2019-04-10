# -*- coding: utf-8 -*-
"""
Created on Wed Feb 07 13:29:27 2018

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import pims
from copy import deepcopy
import scipy.ndimage
import fluids2d.backlight
from scipy.ndimage.morphology import binary_fill_holes
import skimage.draw

folder = r'\\Mae-deike-lab3\d\data_comp3_D\180210\\'
cine_name = r'backlight_bubbles_4pumps_stackedParallel__sunbathing_meanon100_meanoff400_L450mm_h080mm_fps1000'

c = pims.open(folder+cine_name+'.cine')
dt = 1./1000
dx= 0.000211798121201

def get_filled(im,thresh):
    im_filt = scipy.ndimage.filters.median_filter(im,size=2)
    im_filt = fluids2d.backlight.binarize(im_filt,thresh,large_true=False)
    filled = binary_fill_holes(im_filt)
    return filled

def mask(im):
    im = im.astype(float)
    #im = np.rot90(im,k=3)
    #im[690:,:] = 1000
    #im[500:800,:] = 1000
    #im[:,0:260] = 1000
    #im[:,]
    #im = im[0:500,:]
    #im = im[:,250:1100]    
    return im

'''
Get the image
'''
#bg = mask(c[0])
#f = 500
#im = mask(c[f]) #-bg

#im_filled = get_filled(im,700).astype(int)

'''
Crop to a small region containing overlapping bubbles
'''
#im_small = im_filled[275:310,45:80]
#im_small_orig = im[275:310,45:80]
im_small_orig = im_small_orig.copy()
im_small = im_small.copy()




#im_small_bool = props[1].image.copy()
#im_small = np.zeros(np.shape(im_small_bool))
#im_small[im_small_bool==False]=1

ny,nx = np.shape(im_small)

n_ellipse = 2

lims = {'x':[0,nx],
        'y':[0,ny],
        'major_axis_length':[1,50],
        'minor_axis_length':[1,50],
        'orientation':[-np.pi,np.pi]}

class Ellipse:
    def __init__(self,params):
        self.params = params
        
        if self.params is None:
            params = {}
            params['x'] = np.random.uniform(low=0,high=nx)
            params['y'] = np.random.uniform(low=0,high=ny)
            params['major_axis_length'] = np.random.normal(10,5)
            params['minor_axis_length'] = np.random.normal(10,5)
            params['orientation'] = np.random.uniform(-np.pi,np.pi)
            self.params = params
            self.enforce_lims(lims)
        
    def draw(self,ax,color='r'):
        e = matplotlib.patches.Ellipse([self.params['x'],self.params['y']],width=self.params['major_axis_length']*2,height=self.params['minor_axis_length']*2,angle=-1*self.params['orientation']/np.pi*180)
        ax.add_artist(e)
        e.set_facecolor('None')
        e.set_edgecolor(color)
        
    def mutate(self,n=2):
        to_mutate = list(np.random.choice(list(self.params.keys()),size=n))
        for key in to_mutate:
            self.params[key] = self.params[key]*np.random.normal(1,0.1)
    
    def enforce_lims(self,lims):
        for key in list(lims.keys()):
            self.params[key] = max( min( self.params[key],lims[key][1]),lims[key][0])
        
class Member:
    '''
    A collection of multiple ellipses approximating the multiple bubbles
    '''
    def __init__(self,ellipses):
        self.ellipses = ellipses
        
    def copy(self):
        return deepcopy(self)
    
    def covered_area(self,im):        
        res = np.zeros(np.shape(im))        
        for e in self.ellipses:
            in_ellipse_points = skimage.draw.ellipse(e.params['y'],e.params['x'],e.params['minor_axis_length'],e.params['major_axis_length'],shape=np.shape(im),rotation=e.params['orientation'])
            res[in_ellipse_points[0],in_ellipse_points[1]] = 1
        return res
    
    def convolve(self,im):
        res = self.covered_area(im)
        res[res==0] = -1        
        im[im==0] = -1
        conv = res*im
        return conv
    
    def score(self,im):
        conv = self.convolve(im)
        self.score_num = np.sum(conv)
        return self.score_num
    
    def mutate(self,n=3):
        [e.mutate(n=n) for e in self.ellipses]
        [e.enforce_lims(lims) for e in self.ellipses]
        
    def rand_init(self):
        self.ellipses = [Ellipse(None) for _ in range(2)]
        
class Population:
    '''
    The state of the algorithm at a given iteration. Contains the members of 
    the population and methods to evaluate and modify them for the next 
    iteration.
    '''
    def __init__(self,members):
        self.members = members
        
    def score(self,im):
        scores = []
        [scores.append(m.score(im)) for m in self.members]
        order = np.argsort(scores)
        members_new = [self.members[i] for i in order]
        self.members = members_new
        self.scores = [scores[i] for i in order]
        
    def mutate(self,portion=.3,n_params=3):
        #members_to_mutate = range(int(portion*len(self.members)),int(len(self.members)))
        members_to_mutate = np.random.choice(range(int(.25*len(self.members)),int(.8*len(self.members))),int(len(self.members)*portion))
        for mi in members_to_mutate:
            self.members[mi].mutate(n=n_params)
            
    def crossover(self,n_params=2,portion_to=0.3):
        crossover_from = range(int(.3*len(self.members)),int(1*len(self.members)))
        crossover_to = np.random.choice(range(int(.6*len(self.members)),int(.95*len(self.members))),int(len(self.members)*portion_to))
        
        for mi in crossover_to:
            
            from_mi = np.random.choice(crossover_from,1)[0]
            ei = np.random.choice(range(len(self.members[from_mi].ellipses)),1)[0]
            
            #print('Crossing over to member '+str(mi)+' from member '+str(from_mi)+', ellipse number'+str(ei))
            
            to_mutate = list(np.random.choice(list(self.members[from_mi].ellipses[ei].params.keys()),size=n_params))
            for key in to_mutate:
                #print('   gene '+key)
                self.members[mi].ellipses[ei].params[key] = self.members[from_mi].ellipses[ei].params[key]
            
    def reset_some(self,portion=.4):
        members_to_reset = range(0,int(portion*len(self.members)))
        for mi in members_to_reset:
            self.members[mi] = Member([Ellipse(None) for _ in range(len(self.members[mi].ellipses))])
            
    def duplicate_some(self,portion=.2):
        dup_to = np.random.choice(range(int(.4*len(self.members)),int(.5*len(self.members))),int(len(self.members)*.1))
        dup_from_possibilities = range(int(.7*len(self.members)),len(self.members))
        for mi in dup_to:
            print(mi)
            self.members[mi] = deepcopy(self.members[np.random.choice(dup_from_possibilities,1)[0]])
            
def fit_subimage_with_ga(im_small,im_small_orig,n_bubbles=2):
    
    pad = 10
    im_small_orig = skimage.util.pad(im_small_orig,pad,'constant',constant_values=0)
    im_small = skimage.util.pad(im_small,pad,'constant',constant_values=0)
    
    
    '''
    Initialize the population with a list of members
    '''
    scores=[]
    members = []
    for _ in range(200):            
        members.append(Member([Ellipse(None) for _ in range(n_bubbles)]))    
    pop = Population(members)
    
    '''
    Run the algorithm for a given number of iterations, occasionally showing the 
    best configuration
    '''
    
    mean_scores = []
    max_scores = []
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(200):
    
        print(i)
        print('-------------')
        
        '''
        Modify the population from the last time
        '''
        pop.duplicate_some()
        pop.mutate()
        pop.crossover()
        pop.reset_some()    
        
        '''
        Evaluate and re-sort the population's members
        '''
        pop.score(im_small)
        
        max_scores.append(np.max(pop.scores).copy())
        mean_scores.append(np.mean(pop.scores).copy())
        
        '''
        Occasionally show the state of the best member
        '''    
        best_member = pop.members[-1]    
        if i%10==0:
        
            ax.clear()
            ax.imshow(im_small_orig)
            for e in best_member.ellipses:
                e.draw(ax)
                print(e.params)
            
            ax.set_title('iteration '+str(i))
            plt.show()
            plt.pause(.01)
            
    best_member = pop.members[-1]
    
    ellipse_params = [e.params for e in best_member.ellipses]
    
    for ei,ep in enumerate(ellipse_params):
        ellipse_params[ei]['x'] = ellipse_params[ei]['x']-pad
        ellipse_params[ei]['y'] = ellipse_params[ei]['y']-pad
    
    '''
    Show the performance of the algorithm over iterations
    '''
    plt.figure()
    plt.plot(mean_scores,label='population mean')
    plt.plot(max_scores,label='best member')
    
    
    return ellipse_params

ellipse_params = fit_subimage_with_ga(im_small,im_small_orig,n_bubbles=2)

#
#
#res = best_member.covered_area(im_small)
#
#
#
#'''
#Show the best member over three binarized versions of the image/approx
#'''
#
#fig,axs = plt.subplots(1,3); axs=axs.flatten()
#
#axs[0].imshow(im_small)
#axs[1].imshow(res)
#axs[2].imshow(best_member.convolve(im_small))
#
#for e in best_member.ellipses:
#    for ax in axs:
#        e.draw(ax)
#
#'''
#Show it over the original (cropped) image
#'''
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.imshow(im_small_orig,origin='upper')
#for e in best_member.ellipses:
#    e.draw(ax)
#    print(e.params)