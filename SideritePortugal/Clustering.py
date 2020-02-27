import numpy as np
import pandas as pd 
#import matplotlib.pyplot as plt
from skimage.io import imread
import seaborn as sns
from ClassMask import *
import hyperspy.api as hs
import holoviews as hv
plt.rcParams['image.cmap'] = 'cividis' 

t = Mask('thin_section_1_','.bmp','Mask.csv')
t.datacube_creation()
t.workingcube_creation()

image=hv.Image(t.datacube_[:,:,3],['x','y'])
t.datacube_[np.isnan(t.datacube_)] = 0
cube = hs.signals.Signal1D(t.datacube_)
cube.set_signal_type("EDS_SEM")
cube.axes_manager[-1].name = 'Elements'
#cube.axes_manager['E'].units = 
