# -*- coding: utf-8 -*-
"""
Created on Tue May 22 10:09:42 2018

@author: Luc Deike
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import fluids2d.geometry
import fluids2d.backlight as backlight
import pims
import scipy.ndimage
import pandas as pd
from scipy.ndimage.morphology import binary_fill_holes
import skimage.measure
import skimage.filters
import trackpy as tp
import skimage
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from fluids2d.backlight import labeled_props
from fluids2d.backlight import filled2regionpropsdf


C = pims.open(r'C:\Users\Luc Deike\Nicolas\SurfaceTension\SurfaceTension.cine')

sss


