import numpy as np
import os

height = np.linspace(2,26,num=7)
Wtip = np.linspace(4000,5800,num=10)
VT = np.linspace(-0.6,0.6,num=241)
VB = np.linspace(-75,75,num=75)

np.save('raw-data/height.npy',height)
np.save('raw-data/Wtip.npy',Wtip)
np.save('raw-data/VT.npy',VT)
np.save('raw-data/VB.npy',VB)