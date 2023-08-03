# modules
import numpy as np

import calc_columndensity
from calc_tb import LTEAnalysis


clight = 29979245800.0 # cm s^-1

# main
if __name__ == '__main__':
    # Assumptions
    Tk        = 20.  # K
    freq_cont = clight/(0.850*1e-1)*1e-9 # Hz
    kappa     = 1.37 # g cm^-2 at 1.0 mm, Ossenkopf & Henning (1994)
    bmaj      = 14.6 # arcsec
    bmin      = bmaj
    mu        = 2.8
    Icont    = 0.12 # Jy/beam
    number   = True
    Xc18o    = 1.0e-7
    delv     = 0.5*1e5 # km/s
    eta      = 0.66 # main beam efficiency

    Tk        = 20.  # K
    freq_cont = clight/(0.870*1e-1)*1e-9 # Hz
    kappa     = 3.5 # g cm^-2 at 1.0 mm, Ossenkopf & Henning (1994)
    bmaj      = 0.12 # arcsec
    bmin      = bmaj
    mu        = 2.8
    Icont    = 2.812e-3 # Jy/beam
    number   = True
    Xc18o    = 1.0e-8
    delv     = 5.*1e5 # km/s
    #eta      = 0.66 # main beam efficiency

    N_H2 = calc_columndensity.calc_column_cont(Icont, freq_cont, bmaj, 
        bmin, Tk, kappa, number=number, err_Iv=Icont*0.1, mu=mu)

    ltemodel = LTEAnalysis()
    Tb_c18o32 = ltemodel.get_intensity('c18o', 2, Tk, N_H2, delv, lineprof='rect', 
        mode='lte', Xconv=Xc18o, Tbg=2.73, Tb=True, return_tau=False)

    print ('N_H2: %.2e cm^-2'%N_H2)
    print ('Tb: %.2e K'%Tb_c18o32)



    # --> extraporate to edge
    print (Tb_c18o32) # Mote+01 for p