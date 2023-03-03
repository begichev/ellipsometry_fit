import numpy as nmp # to avoid name conflict with refrative index np
from numba import njit

PI = 3.14159265359

@njit
def transfermatrixs_phi(thick, refr, phi, lamb):
	
	theta = phi*PI/180
	k = 2*PI/lamb
	refr = nmp.sqrt(refr**2-nmp.sin(theta)**2)
	tmatrix = nmp.zeros((2, 2), dtype=nmp.complex128)
	tmatrix[0][0] = nmp.cos(k*refr*thick)
	tmatrix[0][1] = -1j*nmp.sin(k*refr*thick)/refr
	tmatrix[1][0] = -1j*nmp.sin(k*refr*thick)*refr
	tmatrix[1][1] = nmp.cos(k*refr*thick)
	
	return tmatrix

@njit
def transfermatrixp_phi(thick, refr, phi, lamb):
	
	theta = phi*PI/180
	k = 2*PI/lamb
	# next two lines don't commute; refr here is original refractive index, not corrected
	thick = thick*(refr**2-nmp.sin(theta)**2)/refr**2
	refr = refr**2/nmp.sqrt(refr**2-nmp.sin(theta)**2)
	tmatrix = nmp.zeros((2, 2), dtype=nmp.complex128)
	tmatrix[0][0] = nmp.cos(k*refr*thick)
	tmatrix[0][1] = -1j*nmp.sin(k*refr*thick)*refr
	tmatrix[1][0] = -1j*nmp.sin(k*refr*thick)/refr
	tmatrix[1][1] = nmp.cos(k*refr*thick)
	
	return tmatrix

#@nb.njit(fastmath=True)
@njit
def matmul(matrix1, matrix2):
	ij = nmp.shape(matrix1)
	jk = nmp.shape(matrix2)
	if(ij[1]==jk[0]):
		res = nmp.zeros((ij[0],jk[1]),dtype=nmp.complex128)
		for i in range(ij[0]):
			for k in range(jk[1]):
				res[i][k] = 0.
				for j in range(ij[1]):
					res[i][k] += matrix1[i][j]*matrix2[j][k]
		return res
	else:
		print('incompatible matrices')
		return nmp.complex128([[1.,0.],[0.,1.]])

@njit
def build_tmatrix(n, d, phi, lambd, pol):
	# input: n is array with shape [1,2,3,4,...,n] where 1 is medium where light is reflected, 
	# n is the depth of the last substrate. 
	# to build tmatrix we take the transfer matrix of second last medium. then multiply from the left with transfer 
	# matrices of upper layers: n-3,n-4,...,2 -> if len(n)=N, then N-4 times. 
	
	Nsize = n.shape[0]

	if pol=='s':
		tms = nmp.zeros((2,2), dtype=nmp.complex128)
		tmatrixs = nmp.eye(2, dtype=nmp.complex128)
		for i in range(Nsize-2):
			ind = Nsize-2-i
			tms = transfermatrixs_phi(d[ind],n[ind],phi,lambd)
			tmatrixs = matmul(tms,tmatrixs)
		return tmatrixs

	elif pol=='p':
		tmp = nmp.zeros((2,2), dtype=nmp.complex128)
		tmatrixp = nmp.eye(2, dtype=nmp.complex128)
		for i in range(Nsize-2):
			ind = Nsize-2-i
			tmp = transfermatrixp_phi(d[ind],n[ind],phi,lambd)
			tmatrixp = matmul(tmp,tmatrixp)
		return tmatrixp
	else:
		print("correct polarization parameters: 's', 'p' ")
		return nmp.zeros((2,2), dtype=nmp.complex128)

@njit
def solve_rt(n,d,phi,lambd):
	# works for single-valued lambda, not arrays (can be used for testing on given lambda)
	# solves for u0,v0 at the upper interface given that u,v at the lower interface are known
	# and then converts it to reflection and transmission amplitudes 
	# for s-pol reflection ampl for electric field, for p-pol for magnetic
	# n[-1] is n of substrate
	
	theta = phi*PI/180

	transfermats = build_tmatrix(n,d,phi,lambd,'s')
	transfermatp = build_tmatrix(n,d,phi,lambd,'p')

	uvs = nmp.array([[1,nmp.sqrt(n[-1]**2-n[0]**2*nmp.sin(theta)**2)]], dtype=nmp.complex128).transpose()
	uvp = nmp.array([[1,nmp.sqrt(n[-1]**2-nmp.sin(theta)**2)/n[-1]**2]], dtype=nmp.complex128).transpose()
	
	u0v0s = matmul(transfermats,uvs)
	u0v0p = matmul(transfermatp,uvp)
	
	rs = (u0v0s[0]*n[0]*nmp.cos(theta)-u0v0s[1])/(u0v0s[0]*n[0]*nmp.cos(theta)+u0v0s[1])
	ts = 2*n[0]*nmp.cos(theta)/(u0v0s[0]*n[0]*nmp.cos(theta)+u0v0s[1])
	rp = (u0v0p[0]*nmp.cos(theta)-n[0]*u0v0p[1])/(u0v0p[0]*nmp.cos(theta)+n[0]*u0v0p[1])
	tp = 2*nmp.cos(theta)/(u0v0p[0]*nmp.cos(theta)+n[0]*u0v0p[1])
	
	# factors, factorp account for different mediums for reflected and transmitted ray
	Rs = nmp.abs(rs)**2
	factors = nmp.real(nmp.sqrt(n[-1]**2-n[0]**2*nmp.sin(theta)**2)/(n[0]*nmp.cos(theta)))
	Ts = factors*nmp.abs(ts)**2
	Rp = nmp.abs(rp)**2
	factorp = 1/(n[-1])**2*nmp.real(nmp.sqrt(n[-1]**2-n[0]**2*nmp.sin(theta)**2))/(nmp.cos(theta)/n[0])
	Tp = factorp*nmp.abs(tp)**2
	
	# line below doesnt work for jit because jit functions cant return array
	#return nmp.array([[Rs,Ts],[Rp,Tp]])
	return Tp 

@njit
def solve_psi(n,d,phi,lambd):
	# works for n: arrays on measured lambd grid
	# solves for u0,v0 at the upper interface given that u,v at the lower interface are known
	# and then converts it to reflection and transmission amplitudes 
	# for s-pol reflection ampl for electric field, for p-pol for magnetic
	# n[-1] is n of substrate
	
	theta = phi*PI/180

	transfermats = build_tmatrix(n,d,phi,lambd,'s')
	transfermatp = build_tmatrix(n,d,phi,lambd,'p')

	uvs = nmp.array([[1,nmp.sqrt(n[-1]**2-n[0]**2*nmp.sin(theta)**2)]], dtype=nmp.complex128).transpose()
	uvp = nmp.array([[1,nmp.sqrt(n[-1]**2-nmp.sin(theta)**2)/n[-1]**2]], dtype=nmp.complex128).transpose()
	
	u0v0s = matmul(transfermats,uvs)
	u0v0p = matmul(transfermatp,uvp)
	
	rs = (u0v0s[0]*n[0]*nmp.cos(theta)-u0v0s[1])/(u0v0s[0]*n[0]*nmp.cos(theta)+u0v0s[1])
	rp = (u0v0p[0]*nmp.cos(theta)-n[0]*u0v0p[1])/(u0v0p[0]*nmp.cos(theta)+n[0]*u0v0p[1])

	psi = nmp.arctan(nmp.abs(rp/rs))*180/PI
	
	return psi

@njit
def solve_delta(n,d,phi,lambd):
	# works for n: arrays on measured lambd grid
	# solves for u0,v0 at the upper interface given that u,v at the lower interface are known
	# and then converts it to reflection and transmission amplitudes 
	# for s-pol reflection ampl for electric field, for p-pol for magnetic
	# n[-1] is n of substrate
	
	theta = phi*PI/180

	transfermats = build_tmatrix(n,d,phi,lambd,'s')
	transfermatp = build_tmatrix(n,d,phi,lambd,'p')

	uvs = nmp.array([[1,nmp.sqrt(n[-1]**2-n[0]**2*nmp.sin(theta)**2)]], dtype=nmp.complex128).transpose()
	uvp = nmp.array([[1,nmp.sqrt(n[-1]**2-nmp.sin(theta)**2)/n[-1]**2]], dtype=nmp.complex128).transpose()
	
	u0v0s = matmul(transfermats,uvs)
	u0v0p = matmul(transfermatp,uvp)
	
	rs = (u0v0s[0]*n[0]*nmp.cos(theta)-u0v0s[1])/(u0v0s[0]*n[0]*nmp.cos(theta)+u0v0s[1])
	rp = (u0v0p[0]*nmp.cos(theta)-n[0]*u0v0p[1])/(u0v0p[0]*nmp.cos(theta)+n[0]*u0v0p[1])

	if nmp.angle(rp/rs)>0:
		delta = nmp.angle(rp/rs)*180/PI
	else:
		delta = nmp.angle(rp/rs)*180/PI + 360 # compensation of 2pi jumps
	#delta = nmp.angle(rp/rs, deg=True)

	return 360-delta
