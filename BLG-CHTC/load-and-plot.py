import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from STM_package import STM


def plot_dIdV(I,VT,VB,show=True,save=False,norm=False):
    """
    Plots a color map of dI/dV spectrum (VT vs VB).

    Parameters
    ----------
    show:       Boolean; Select "True" to show the image. Select "False" (default)
                to simply generate it. Useful if you would just like to save the image.

    save:       Boolean; Select "True" to save to (file location)

    norm:       Boolean; Select "True" to normalize and plot (dI/dV)/(I/VT) spectrum.
    """

    dIdV = np.gradient(I,axis=0) # dI/dV

    if norm == True:
        IV = I / VT[:,np.newaxis] # I/V
        dIdV = dIdV / IV

    fig, ax = plt.subplots(figsize=(7,6))

    dIdV_plot = plt.imshow(dIdV,cmap=cm.RdYlGn,origin='lower',
                            aspect='auto',extent=(VB[0],VB[-1],1e3*VT[0],1e3*VT[-1]))
    fig.suptitle('$dI/dV$, Tip Height ={} nm'.format(d1))
    cbar = fig.colorbar(dIdV_plot,label='$dI/dV$ (S)')
    ax.set_xlabel('Gate Voltage (V)')
    ax.set_ylabel('Tip Voltage (mV)')
    if show == True:
        plt.show()

    if save == True:
        import os
        save_dir = os.path.join( os.path.dirname(__file__),
                                'dIdV_Plots')
        fig.savefig(os.path.join(save_dir,'tip_height_{}ang.png'.format(round(d1))))

# def plot_dIdV_waterfall(I,VT,VB):
    """
    Plots a waterfall plot of dIdV spectrum

    Parameters
    ----------
    
    VT:     array-like, array of tip voltages to plot

    VB:     array-like, array of gate voltages to plot
    """

    # fig, ax = plt.subplots(figsize=(6,10))

    # fig.suptitle('dI/dV')

    # current = self.generate_tunnelcurrent(VT,VB,return_current=True)

    # dIdV = np.gradient(current, axis=0)

    # num_points = np.shape(dIdV)[0]
    # num_plots = np.shape(dIdV)[1]

    # offsets = np.linspace(0,num_plots-1,num=num_plots).reshape(1,num_plots)*3*10**16

    # ax.plot(VT,dIdV+offsets,color='b')

    # ax.text(VT[-int(num_points/4)],(dIdV+offsets)[0,-1],"$V_B={}$ V".format(VB[-1]))
    # ax.text(VT[-int(num_points/4)],(dIdV+offsets)[0,0],"$V_B={}$ V".format(VB[0]))


    # plt.show()

if len(sys.argv) != 2:
	print('Usage: %s NPY_FILE' % (os.path.basename(sys.argv[0])))

filename = sys.argv[1]

first = filename.find('_') + 1
second = first + filename[first:].find('_') + 1


d1 = float(filename[first:second-1])/10
Wtip = float(filename[second:second+4])/1000
d2 = 285
e1, e2 = 1, 4
T = 0

VT = np.linspace(-0.6,0.6,num=10)
VB = np.linspace(-75,75,num=10)

current = np.load(filename)


plot_dIdV(current,VT,VB)