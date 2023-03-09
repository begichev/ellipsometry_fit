import numpy as np

def DrudeLorentz(wp,w0,g,w):
	# only positive wp**2 are presented
	# returns complex diel function in DL model on measured omega grid in eV
	# to consider drude response w0=0
    return wp**2/(w0**2-w**2-1.j*w*g)

def Cauchy(A,B,lambd):
	return A+B/lambd**2

def TaucLorentz(A,E0,C,Eg,E):
    # from weidtn/tmm

    t1 = (Eg **2 - E0**2) * E**2
    t2 = (Eg**2 * C**2)
    t3 = E0**2 * (E0**2 + 3* Eg**2)
    a_ln = t1 + t2 - t3 

    t1 = (E**2 - E0**2) * (E0**2 + Eg**2)
    t2 = Eg**2 * C**2
    a_atan = t1 + t2

    alpha = np.sqrt(4* E0**2 - C**2)
    gamma = np.sqrt(E0**2 - (C**2/2))
    t1 = (E**2 - gamma**2)**2
    t2 = (0.25 * alpha**2 * C**2)
    zeta4 = t1 + t2 

    t1 = 0
    t2 = (A * C * a_ln)/(np.pi * zeta4 * 2 * alpha * E0) * np.log((E0**2 + Eg**2 + alpha * Eg)/(E0**2 + Eg**2 - alpha * Eg))
    t3 = -1 * (A * a_atan)/(np.pi * zeta4 * E0) * ( np.pi - np.arctan( (2*Eg+alpha)/C) + np.arctan((-2*Eg+alpha)/C))
    t4 = 2 * (A * E0 * Eg)/(np.pi * zeta4 * alpha) * (E**2 - gamma**2) * (np.pi + 2* np.arctan(2*(gamma**2-Eg**2)/(alpha*C)))
    t5 = -1 * (A * E0 * C)/(np.pi * zeta4) * (E**2 + Eg**2)/(E) * np.log(np.abs(E-Eg)/(E+Eg))
    t6 = 2 * (A * E0 * C)/(np.pi * zeta4) * Eg * np.log((np.abs(E-Eg)*(E+Eg))/(np.sqrt((E0**2 - Eg**2)**2 + Eg**2 * C**2)))
    epsR = (t1+t2+t3+t4+t5+t6)

    a = 1/E * (A*E0*C*(E-Eg)**2)/((E**2-E0**2)**2+(C**2)*(E**2))
    epsI = np.where(E > Eg, a,0)

    return epsR+1.j*epsI

