import sys
import os
import numpy as np
from STM_package import STM

if len(sys.argv) != 3:
	print('Usage: %s d1 Wtip' % (os.path.basename(sys.argv[0])))

filename = 'Current_%d_%d.npy' % (int(sys.argv[1]), int(sys.argv[2]))

d1, d2 = float(sys.argv[1])/10, 285
e1, e2 = 1, 4
T = 0
Wtip= float(sys.argv[2])/1000

experiment = STM.BLGinSTM(d1,d2,e1,e2,T,Wtip)

VT = np.linspace(-0.6,0.6,num=10)
VB = np.linspace(-75,75,num=10)

experiment.generate_tunnelcurrent(VT,VB)

# File format, I_tipheight_workfunction.npy
# where tipheight is in angstroms and workfunction is in meV
np.save(filename,experiment.I)