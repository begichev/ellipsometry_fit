import numpy as nmp # to avoid name conflict with refrative index np

def build_tmatrix(n, d, phi, lambd):
    # input: n is array with shape [1,2,3,4,...,n] where 1 is medium where light is reflected, 
    # n is the depth of the last substrate. 
    # to build tmatrix we take the transfer matrix of second last medium. then multiply from the left with transfer 
    # matrices of upper layers: n-3,n-4,...,2 -> if len(n)=N, then N-4 times. 
    
    tmatrixs = nmp.eye(2, dtype=float)
    tmatrixp = nmp.eye(2, dtype=float)
    Nsize = nmp.shape(n)[0]
    
    for i in range(Nsize-2):
        ind = Nsize-2-i
        tmatrixs = nmp.matmul(transfermatrixs_phi(d[ind],n[ind],phi,lambd),tmatrixs)
        tmatrixp = nmp.matmul(transfermatrixp_phi(d[ind],n[ind],phi,lambd),tmatrixp)

    return [tmatrixs, tmatrixp]

def solve_rt(n,d,phi,lambd):
    # solves for u0,v0 at the upper interface given that u,v at the lower interface are known
    # and then converts it to reflection and transmission amplitudes 
    # for s-pol reflection ampl for electric field, for p-pol for magnetic
    # n[-1] is n of substrate
    
    theta = phi*nmp.pi/180
    uvs = nmp.array([1,nmp.sqrt(n[-1]**2-n[0]**2*nmp.sin(theta)**2)])
    uvp = nmp.array([1,nmp.sqrt(n[-1]**2-nmp.sin(theta)**2)/n[-1]**2])
    
    [transfermats, transfermatp] = build_tmatrix(n,d,phi,lambd)
    u0v0s = nmp.matmul(transfermats,uvs)
    u0v0p = nmp.matmul(transfermatp,uvp)
    
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
    
    return nmp.array([[Rs,Ts],[Rp,Tp]])

def solve_psidelta(n,d,phi,lambd):
    # solves for u0,v0 at the upper interface given that u,v at the lower interface are known
    # and then converts it to reflection and transmission amplitudes 
    # for s-pol reflection ampl for electric field, for p-pol for magnetic
    # n[-1] is n of substrate
    
    theta = phi*nmp.pi/180
    uvs = nmp.array([1,nmp.sqrt(n[-1]**2-n[0]**2*nmp.sin(theta)**2)])
    uvp = nmp.array([1,nmp.sqrt(n[-1]**2-nmp.sin(theta)**2)/n[-1]**2])
    
    [transfermats, transfermatp] = build_tmatrix(n,d,phi,lambd)
    u0v0s = nmp.matmul(transfermats,uvs)
    u0v0p = nmp.matmul(transfermatp,uvp)
    
    rs = (u0v0s[0]*n[0]*nmp.cos(theta)-u0v0s[1])/(u0v0s[0]*n[0]*nmp.cos(theta)+u0v0s[1])
    rp = (u0v0p[0]*nmp.cos(theta)-n[0]*u0v0p[1])/(u0v0p[0]*nmp.cos(theta)+n[0]*u0v0p[1])
    psi = nmp.arctan(nmp.abs(rp/rs))*180/nmp.pi
    delta = nmp.angle(rp/rs)*180/nmp.pi
    
    return nmp.array([psi,delta])
    