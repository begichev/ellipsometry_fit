import numpy as np

def grid_adapt(omegas,tab_grid,tab_n,tab_k):
### adapt tabulated model params to measured energy grid
    si_ad = np.zeros(omegas.shape[0],dtype=complex)
    for i in np.arange(omegas.shape[0]):
        en = omegas[i]
        # find energy value in a table closest to measured value
        ind = (np.abs(tab_grid - en)).argmin()
        # fill this value in grid with complex refractive index
        si_ad[i] = tab_n[ind]+1.j*tab_k[ind]
    
    return si_ad

def corr_delta(delt,omegas):
    # eliminates 2pi steps in delta
    if np.size(np.where(np.diff(delt)>100)[0])>0:
        crr_delta = np.zeros(omegas.shape[0],dtype=np.float64)
        for i in range(delt.shape[0]):
            crr_delta[i] = delt[i]
        for i in np.where(np.diff(delt)>100)[0]:
            for j in range(int(i)+1,omegas.shape[0],1):
                crr_delta[j] = delt[j]-360*(2*int(np.diff(delt)[i]>0)-1)
        return corr_delta(crr_delta,omegas)
    else:        
        return delt