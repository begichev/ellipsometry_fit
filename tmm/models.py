import numpy as np

def DrudeLorentz(einf,wp,w0,g,w):
	# only positive wp**2 are presented
	# returns complex refr index in DL model on measured omega grid in eV
    return np.sqrt(einf+wp**2/(w0**2-w**2-1.j*w*g))

def Cauchy(A,B,lambd):
	return A+B/lambd**2

