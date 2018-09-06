from TunnelingExperiment import STM
import numpy as np
d1, d2 = 1, 285
e1, e2 = 1, 4
T = 0
Wtip = 4

experiment = STM.BLGinSTM(d1,d2,e1,e2,T,Wtip)

VT = np.linspace(-0.6,0.6, num=50)
VB = np.linspace(-75,75,num=50)

experiment.plot_dIdV_waterfall(VT,VB)
#experiment.generate_tunnelcurrent(VT,VB)
#experiment.plot_dIdV()