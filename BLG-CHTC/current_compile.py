import numpy as np
import os


numpyfiles = [f for f in os.listdir('raw-data') if 'Current_' in f]

heights = np.load('raw-data/height.npy')
Wtips = np.load('raw-data/Wtip.npy')
VT = np.load('raw-data/VT.npy')
VB = np.load('raw-data/VB.npy')

Current = np.zeros((len(heights),len(Wtips),len(VT),len(VB)))

for f in numpyfiles:
	first = f.find('_') + 1
	second = first + f[first:].find('_') + 1
	d1 = int(f[first:second-1])
	Wtip = int(f[second:second+4])

	d1_index = np.argwhere(heights==d1)[0][0]
	Wtip_index = np.argwhere(Wtips==Wtip)[0][0]

	Current[d1_index,Wtip_index,:,:] = np.load(os.path.join('raw-data',f))

np.save('raw-data/Current.npy',Current)