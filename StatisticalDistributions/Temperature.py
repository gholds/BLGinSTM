from . import *

def FermiDirac(en, T):
	"""
	The Fermi-Dirac distribution.

	Parameters
	----------
	en: 		array-like
				Energy of state in units J

	T:			scalar
				Temperature in units K

	Returns
	----------
	FD:			array-like
				Fermi-Dirac probability of state at energy en

	"""

	kB = 1.38064852*10**-23 # J/K 

	# Using logaddexp reduces chance of underflow error
	FD = np.exp( -np.logaddexp(en/(kB*(T+0.000000000001)),0) )

	return FD