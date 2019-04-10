 # -*- coding: utf-8 -*-
"""
Created on Tue Sep 05 15:08:08 2017

@author: danjr
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import cv2

def perspective(im,pts0):
    
    #pts=[[,],[,],[,],[,]]
    #pts=[[topleft,],[topright,],[bottomright,],[bottomleft,]]
    'PLOT ORIGINAL IMAGE avec carre'
    pts=np.array(pts0)
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.imshow(im,cmap='gray')
#    SquareSide=[[(pts[0][0],pts[0][1]),(pts[1][0],pts[1][1])],[(pts[1][0],pts[1][1]),(pts[2][0],pts[2][1])],[(pts[2][0],pts[2][1]),(pts[3][0],pts[3][1])],[(pts[3][0],pts[3][1]),(pts[0][0],pts[0][1])]]
#    square = mc.LineCollection(SquareSide, linewidths=1)
#    ax.add_collection(square)
    'HOMOGRAPHY'
    d=pts[2][0]-pts[3][0]
    pts_dst = np.array([[pts[3][0],pts[3][1]-d],[pts[3][0]+d,pts[3][1]-d],[pts[3][0]+d,pts[3][1]],[pts[3][0],pts[3][1]]])
    h, status = cv2.findHomography(pts, pts_dst)
    im_out = cv2.warpPerspective(im, h, (im.shape[1],im.shape[0]))
    'PLOT IMAGE REDRESEE avec carre'
#    fig1 = plt.figure()
#    ax = fig1.add_subplot(111)
#    ax.imshow(im_out,cmap='gray')
#    SquareSide=[[(pts[3][0],pts[3][1]-d),(pts[3][0]+d,pts[3][1]-d)],[(pts[3][0]+d,pts[3][1]-d),(pts[3][0]+d,pts[3][1])],[(pts[3][0]+d,pts[3][1]),(pts[3][0],pts[3][1])],[(pts[3][0],pts[3][1]),(pts[3][0],pts[3][1]-d)]]
#    square = mc.LineCollection(SquareSide, linewidths=1)
#    ax.add_collection(square)
    return im_out


class GeometryScaler:
    '''
    Handles pixels -> lab position for a dataset, given the spatial resolution
    and the location of the image.
    '''
    
    def __init__(self,dx=1,im_shape=(0,0),origin_pos=(0,0),origin_units='m'):
        '''
        Initialize the scaler with the dist/pix scaling, shape of the image,
        and (y,x) of the image origin (the top left corner!)
        '''
        
        self.dx=dx # pixel width
        self.im_shape = np.asarray(im_shape,dtype=int)
        self.origin_pos=origin_pos # location of (0,0) relative to the lab origin
        
        self.x = self(np.arange(self.im_shape[1]))
#        self.y = np.flipud(self(np.arange(self.im_shape[0])))
        self.y = self(np.arange(self.im_shape[0]))
        
        '''
        Handle the origin offset
        '''
        
        self.origin_pos = np.asarray(origin_pos)
        if origin_units == 'pix':
            '''
            If the origin position is given in pixels, convert that to meters.
            '''
            self.origin_pos = self(self.origin_pos)        
        
        self.x = self.x + self.origin_pos[1]
        self.y = self.y + self.origin_pos[0]
        
        self.X,self.Y = np.meshgrid(self.x,self.y)
        
        self.im_extent = [self.x[0]-self.dx/2,self.x[-1]+self.dx/2,self.y[-1]-self.dx/2,self.y[0]+self.dx/2]
        
    def __call__(self,coords,absolute=True):        
        '''
        Scale pixels to meters  
        '''        
        coords = np.array(coords)
        return coords * self.dx
        
    def get_coords(self,loc):
        '''
        Return the [row,column] corresponding to some [y,x] in loc.
        '''
        f_y = interp1d(self.y,np.arange(0,len(self.y)),kind='nearest',fill_value='extrapolate')  
        f_x = interp1d(self.x,np.arange(0,len(self.x)),kind='nearest',fill_value='extrapolate')        

        return np.asarray([f_y(loc[0]),f_x(loc[1])],dtype=int)
    
    def get_loc(self,coords):
        '''
        Return the [y,x] corresponding to some [row,column] in loc.
        '''
        f_y = interp1d(np.arange(0,len(self.y)),self.y,kind='linear',fill_value='extrapolate')  
        f_x = interp1d(np.arange(0,len(self.x)),self.x,kind='linear',fill_value='extrapolate') 
        
        coords = np.atleast_2d(coords)
        
        #return np.asarray([f_y(coords[0]),f_x(coords[1])],dtype=float)
        return np.asarray([f_y(coords[:,0]),f_x(coords[:,1])],dtype=float)
    
    def set_axes_limits(self,ax,which='both'):
        if which=='both':
            ax.set_ylim(self.im_extent[2],self.im_extent[3])
            ax.set_xlim(self.im_extent[0],self.im_extent[1])
        elif which=='x':
            ax.set_xlim(self.im_extent[0],self.im_extent[1])
        elif which=='y':
            ax.set_ylim(self.im_extent[2],self.im_extent[3])
            
def create_piv_scaler(p,g_cine):
    
    '''
    Define the piv scaler's origin offset in relation 
    '''    
    
    if p.crop_lims is not None:
        piv_origin_offset_pix = [(p.crop_lims[0])+(p.window_size-p.overlap),
                                 (p.crop_lims[2])+(p.window_size-p.overlap)]
    else:
        piv_origin_offset_pix = [(p.window_size-p.overlap),
                                 (p.window_size-p.overlap)]
    
    piv_origin_offset_m = p.dx*np.asarray(piv_origin_offset_pix) + g_cine.origin_pos
    
    g_piv = GeometryScaler(dx=p.dx*(p.window_size-p.overlap),im_shape=np.shape(p.data.ff[0,:,:,0]),origin_pos=piv_origin_offset_m,origin_units='m')
    
    return g_piv
    
def highlight_region_on_im(im,region,ax=None,gscaler=None,facecolor='y'):
    if ax is None:
        fig=plt.figure()
        ax=fig.add_subplot(111)
    
    from matplotlib import patches
    
    bottom_left = region[0,2]
    width = region[1]-region[0]
    height = region[3]-region[2]
    
    ax.imshow(im)
    r=patches.Rectangle(bottom_left,width,height,facecolor=facecolor,alpha=0.3)
    ax.add_patch(r)
    return ax
    
        
if __name__ == '__main__':
    s = GeometryScaler(dx=0.01,origin_pos=[0.2,0.05])
    
    print(s([[1,0],[-2,0]]))