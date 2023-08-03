### 2020.8.25 Tue.
### calculating Sigma_H2 column density
### assuming LTE condition, optically thin

### Line: from integrated intensity map (int Iv dv)
### Continuum: from continuum image (Iv)


### import modules
import os
import sys
import math
import numpy as np
import pandas as pd



### constants
kb     = 1.38064852e-16  # Boltzmann constant [erg K^-1]
hp     = 6.626070040e-27 # Planck constant [erg s]
clight = 2.99792458e10   # light speed [cm s^-1]
mH     = 1.672621898e-24 # proton mass [g]
Msun   = 1.9884e33       # solar mass [g]

# distance
# AU --> km 1.50e8
# AU --> cm 1.50e13
auTOkm = 1.495978707e8  # AU --> km
auTOcm = 1.495978707e13 # AU --> cm
auTOpc = 4.85e-6        # au --> pc
pcTOau = 2.06e5         # pc --> au
pcTOcm = 3.09e18        # pc --> cm


# path
path_to_here = os.path.dirname(__file__)
#print (path_to_here)


# partition function
def Pfunc(EJ, gJ, J, Tex):
    '''
    Calculate the partition function.

    Args:
        EJ: energy at energy level J
        gJ: statistical weight
        J: energy level
        Tk: kinetic energy

    Return:
        Z: partition function
    '''
    Zarray = np.array([gJ[j]*np.exp(-EJ[j]/Tex) for j in J])
    Z      = np.sum(Zarray)
    return Z


# planck function
def Bv(T,v):
    '''
    Planck function

    Args:
        T: temprature [K]
        v: frequency [GHz]
    '''
    v     = v * 1.e9 # GHz --> Hz
    exp   = np.exp((hp*v)/(kb*T))-1.0
    fterm = (2.0*hp*v*v*v)/(clight*clight)
    Bv    = fterm/exp
    return Bv



# dust opacity kappa in Beckwith+90
def kappa_BW90(freq,beta):
    '''
    Dust opacity kappa drived in Beckwith et al. (1990).
    Note that this dust opacity is for deriving total mass,
    i.e. for gas assuming gas to dust mass ratio 100.
    For dust, it must be multiplied by 100.

    Args:
        freq: observational frequancy [Hz]
        beta: dust opacity index beta
        kappa: dust opacity [cm2 g-1]
    '''
    kappa = 0.1*(freq*1.e-12)**beta
    print ('Caution kappa_BW90:\tKappa derived here is for gas mass. Muptiple 100 with kappa if you want to derive dust mass.')
    return kappa



# dust column density
def calc_column_cont(Iv, freq, bmaj, bmin, T, kappa, dist=140.,
 Rg_to_d=100., mu=2.8, number=False, err_Iv=0.):
    '''
    Calculate dust column density assuming optically thin and temperature.

    Args:
        Iv: intensity [Jy/beam]
        freq: frequency [GHz]
        bmaj, bmin: [arcsec]
        T: temperature [K]
        kappa: dust opacity [cm2 g-1]. Note that this opacity is
               assumed to be an opacity for dust mass.
        dist: distance to the object [pc]
        Rd_to_g: gas-to-dust mass ratio
        mu: mean molecular weight. Default mu=2.8 derives
         correct column number density of H2 gas (Kauffman+08).
    '''
    print ('')
    print ('Input values')
    print ('Beam size: %.2f x %.2f arcsec'%(bmaj,bmin))
    print ('Frequency: %.6e GHz'%freq)
    print ('temperature: %.2f K'%T)

    # convert units
    bmaj = bmaj*np.pi/(180.*60.*60.) # arcsec --> radian
    bmin = bmin*np.pi/(180.*60.*60.) # arcsec --> radian

    # Jy/beam -> Jy/str
    # Omg_beam (str) = (pi/4ln(2))*beam (rad^2)
    # I [Jy/beam] / Omg_beam = I [Jy/str]
    # beam area = Omega_beam x d^2
    C2     = np.pi/(4.*np.log(2.)) # beam(rad) -> beam (sr)
    bTOstr = bmaj*bmin*C2          # beam --> str

    Istr = Iv/bTOstr       # Jy/beam --> Jy/str
    Istr = Istr*1.0e-26    # Jy --> MKS (Jy = 10^-26 Wm-2Hz-1)
    Istr = Istr*1.e7*1.e-4 # MKS --> cgs (erg s^-1 cm^-2 Hz^-1 str^-1)
    err_Istr = err_Iv/bTOstr
    err_Istr = err_Istr*1.0e-26*1.e7*1.e-4

    # column density (of gas or dust depending on kappa)
    Sigma     = Istr/(Bv(T,freq)*kappa) # g cm^-2
    Sigma_gas = Sigma*Rg_to_d           # g cm^-2
    N_H2      = Sigma_gas/(mu*mH)

    # Error propagation
    dSig_dIv   = Sigma/Istr
    err_Sig    = np.sqrt( (dSig_dIv*err_Istr)**2. )
    err_Siggas = np.sqrt( (dSig_dIv*err_Istr)**2. )*Rg_to_d
    err_NH2    = err_Siggas/(mu*mH)

    print ('')
    print ('Output')
    if number:
        print ('N_H2: %.4e cm^-2'%N_H2)
        print ('Uncertainty: %.4e cm^-2'%err_NH2)
        return N_H2
    else:
        print ('Sigma_dust: %.4e g cm^-2'%Sigma)
        print ('Uncertainty: %.4e g cm^-2'%err_Sig)
        print ('Sigma_gas: %.4e g cm^-2'%Sigma_gas)
        return Sigma_gas


# read molecular data
def read_lamda_moldata(infile):
    '''
    Read a molecular data file from LAMDA (Leiden Atomic and Molecular Database)
    '''
    # read
    data = pd.read_csv(infile, comment='!', delimiter='\n', header=None)

    # get
    # line name, weight, nlevels
    line, weight, nlevels = data[0:3][0].values
    weight  = float(weight)
    nlevels = int(nlevels)

    # energy on each excitation level
    elevels = data[3:3+nlevels].values
    elevels = np.array([ elevels[i][0].split() for i in range(nlevels)])
    lev, EJ, gJ, J = elevels.T
    lev = np.array([ int(lev[i]) for i in range(nlevels)])
    EJ  = np.array([ float(EJ[i]) for i in range(nlevels)])
    gJ  = np.array([ float(gJ[i]) for i in range(nlevels)])
    J   = np.array([ int(J[i]) for i in range(nlevels)])

    # number of transition
    ntrans = data[0][3+nlevels].strip()
    ntrans = int(ntrans)

    # Einstein A coefficient
    vtrans = data[3+nlevels+1:3+nlevels+1+ntrans].values
    vtrans = np.array([vtrans[i][0].split() for i in range(ntrans)])

    itrans, Jup, Jlow, Acoeff, freq, delE = vtrans.T
    itrans = np.array([ int(itrans[i]) for i in range(ntrans)])
    Jup    = np.array([ int(Jup[i]) for i in range(ntrans)])
    Jlow   = np.array([ int(Jlow[i]) for i in range(ntrans)])
    Acoeff = np.array([ float(Acoeff[i]) for i in range(ntrans)])
    freq   = np.array([ float(freq[i]) for i in range(ntrans)])
    delE   = np.array([ float(delE[i]) for i in range(ntrans)])

    return line, weight, nlevels, EJ, gJ, J, ntrans, Jup, Acoeff, freq, delE




# gas column density from line
def calc_column_line(mol, Ivint, bmaj, bmin, Tex, Ju, Xconv,
 dist=140., tau=False, delv=None, sigma=False, Tbg=2.73, number=True,
 err_Ivint=0., err_Tex=0., err_delv=0.):
    '''
    Calculate gas column density from line

    Args:
      Iving: integrated intensity [Jy/beam km/s]
      Tk: kinetic temperature [K]
      bmaj, bmin: beam size along beam major and minor axis [arcsec]
      Ju(int): upper excitation state
      Z: partision function
      Xconv: conversion factor
      dist: distance to the object [pc]
      Tbg: background temperature [K]

      (parameters of molecules)
      Acoeff: Einstein A coefficient [s^-1]
      freq: frequency [GHz]
      delE: transitional energy [K]
      EJ: energy at J [cm^-1]
      gJ: statistical weight
      J: energy level
      mu: molecular weight
      Brot: rotational constant [s^-1]

    Return:
      Sigma_H2O [g cm^-2]
    '''

    if mol == 'c18o':
        infile = mol + '.dat'
    elif mol == 'n2hp':
        infile = mol + '.dat'
    elif mol == 'co':
        infile = mol + '.dat'
    else:
        print ('ERROR: mol must be c18o, co, and n2hp for now.')
        return

    line, mu, nlevels, EJ, gJ, J, ntrans, Jup, Acoeff, freq, delE = read_lamda_moldata(path_to_here + '/' + infile)

    # units
    J  = J.astype(np.int) # float --> integer
    EJ = EJ*hp*clight/kb  # cm^-1 --> K


    ### start calculation
    # excitation
    Ju = int(Ju)

    # emission of C18O Ju --> Jl
    freq_ul = freq[Ju-1]   # GHz
    Aul     = Acoeff[Ju-1]
    gu      = gJ[Ju]
    gl      = gJ[Ju-1]
    Eu      = EJ[Ju]

    # partition function
    Qrot = Pfunc(EJ, gJ, J, Tex)
    #Z = kb*Tex/(hp*Brot) # under the condision in which hBrot << kT

    # check
    print ('')
    print ('Line: %s J=%i--%i'%(line, Ju, Ju-1))
    print ('Frequency: %.6e GHz'%freq_ul)
    print ('Assuming Tex=%.2f'%Tex)
    print ('Abandance: %2e'%Xconv)
    #print 'weight: ', gu ,'to', gl

    # convert units
    bmaj = bmaj*np.pi/(180.*60.*60.) # arcsec --> radian
    bmin = bmin*np.pi/(180.*60.*60.) # arcsec --> radian


    if tau:
        # derive column density from optical depth tau
        if delv:
            pass
        else:
            print ('ERROR: delv is necessary if tau=True')
            return

        delv = delv*1e5 # km/s --> cm/s

        if sigma:
            term_delv = np.sqrt(2.*np.pi)*delv
        else:
            term_delv = np.sqrt(np.pi)*delv/(2.*np.sqrt(np.log(2.)))

        tau_tot   = Ivint
        c1        = 8.*np.pi*(freq_ul*1.e9)**3.*Qrot
        c2        = clight*clight*clight*gu*Aul
        exp       = np.exp(Eu/Tex)
        exp2      = np.exp(hp*freq_ul*1.e9/(kb*Tex)) - 1.
        N_mol     = tau_tot*term_delv*c1/c2*exp/exp2
        Sigma_mol = N_mol*mu*mH

        # Calculate the error propagation.
        dN_dtau  = N_mol/tau_tot
        dN_ddelv = N_mol/delv
        dN_dTex  = c1/c2*tau_tot*term_delv*Tex**(-2.)*exp*exp2**(-2.)\
         *((-Eu + hp*freq_ul*1.e9/kb)*np.exp(hp*freq_ul*1.e9/(kb*Tex)) + Eu)

        err_Nmol   = np.sqrt((dN_dtau*err_Ivint)**2. + (dN_ddelv*err_delv)**2. + (dN_dTex*err_Tex)**2. )
        err_Sigmol = err_Nmol*mu*mH
    else:
        # derive column density from the integrated intensity

        # Jy/beam -> Jy/str
        # Omg_beam (str) = (pi/4ln(2))*beam (rad^2)
        # I [Jy/beam] / Omg_beam = I [Jy/str]
        # beam area = Omega_beam*d^2
        C2     = np.pi/(4.*np.log(2.)) # beam(rad) -> beam (sr)
        bTOstr = bmaj*bmin*C2          # beam --> str

        Istr = Ivint/bTOstr          # Jy/beam km/s --> Jy/str km/s
        Istr = Istr*1.0e-26          # Jy --> MKS (Jy = 10^-26 Wm-2Hz-1)
        Istr = Istr*1.e7*1.e-4*1.e5  # MKS --> cgs (erg s^-1 cm^-2 Hz^-1 str^-1 cm/s)
        err_Istr = err_Ivint/bTOstr
        err_Istr = err_Istr*1.0e-26*1.e7*1.e-4*1.e5

        # coefficients
        c1  = 8.*np.pi*(freq_ul*1.e9)**3.*Qrot
        c2  = clight*clight*clight*gu*Aul
        exp = np.exp(Eu/Tex)
        jterm = Bv(Tex, freq_ul) - Bv(Tbg, freq_ul)
        exp2  = np.exp(hp*freq_ul*1.e9/(kb*Tex)) - 1.

        N_mol     = Istr*c1/c2*exp/exp2/jterm # cm^-2
        Sigma_mol = N_mol*mu*mH               # g cm^-2

        # Calculate the error propagation.
        dN_dIvint  = N_mol/Istr
        err_Nmol   = np.sqrt((dN_dIvint*err_Istr)**2. )
        err_Sigmol = err_Nmol*mu*mH

    # column density of H2O gas
    N_H2      = N_mol/Xconv
    Sigma_H2  = Sigma_mol/Xconv

    # print results
    print ('')
    print ('Output')

    if number:
        print ('N_%s: %4e cm^-2'%(mol, N_mol))
        print ('Uncertainty: %4e cm^-2'%err_Nmol)
        #print ('N_H2: %4e cm^-2'%N_H2)
        return N_mol #, N_H2
    else:
        print ('Sigma_%s: %4e g cm^-2'%(mol, Sigma_mol))
        print ('Uncertainty: %4e g cm^-2'%err_Sigmol)
        #print ('Sigma_H2: %4e g cm^-2'%Sigma_H2)
        return Sigma_mol #, Sigma_H2


def LTEmass(mol, Fv, Tex, Ju, Xconv, dist=140., mu=2.8,
     S_TA=None, bmaj=None, bmin=None):
    '''
    Calculate LTE mass from line flux density.

    Input values
        mol: molecule
        Fv: observed flux density [Jy km/s or K km/s]
        dist: distance from the solar system [pc]
        Tex: excited temperature [K]
        (Tk: kinetic temperature [K])
        Xconv: relative abandunce of the molecule to H2
        Ju: upper excitation level. Ju-Jl
        mu: mean molecular weight (Kauffuman+08)
        S_TA: S/TA which is a conversion factor from antenna temperature (TA) K to flux density/beam Jy/beam.
         If it's given, Fv is treated as K km/s and converted to in Jy.

    For C18O
    ## emission: C18O J=2-1, 219.56035 GHz
    # Acoeff: Einstein A coefficient [s^-1]
    # freq: frequency [GHz]
    # delE: transitional energy [K]
    # EJ: energy at J [cm^-1]
    # gJ: statistical weight
    # J: energy level
    # Brot: rotational constant [s^-1]
    '''

    if mol == 'c18o':
        infile = mol + '.dat'
    elif mol == 'n2hp':
        infile = mol + '.dat'
    elif mol == 'co':
        infile = mol + '.dat'
    else:
        print ('ERROR: mol must be c18o, co, and n2hp for now.')
        return

    line, m_mol, nlevels, EJ, gJ, J, ntrans, Jup, Acoeff, freq, delE = read_lamda_moldata(path_to_here + '/' + infile)

    # units
    J  = J.astype(int)   # float --> integer
    EJ = EJ*hp*clight/kb # cm^-1 --> K


    ### start calculation
    # excitation
    Ju = int(Ju)

    # emission of C18O Ju --> Jl
    freq_ul = freq[Ju-1]   # GHz
    Aul     = Acoeff[Ju-1]
    gu      = gJ[Ju]
    gl      = gJ[Ju-1]
    Eu      = EJ[Ju]

    # partition function
    Qrot = Pfunc(EJ, gJ, J, Tex)
    #Z = kb*Tex/(hp*Brot) # under the condision in which hBrot << kT

    # check
    print ('')
    print ('Line: %s J=%i--%i'%(line, Ju, Ju-1))
    print ('Frequency: %.6e GHz'%freq_ul)
    print ('Assuming Tex=%.2f'%Tex)
    print ('Abandance: %2e'%Xconv)
    #print 'weight: ', gu ,'to', gl



    # !!! start !!!
    dist = dist*pcTOcm # pc --> cm

    C1 = Qrot*(4.*np.pi*mu*mH)/(hp*clight*gu*Aul)

    # observed flux
    if S_TA:
        if bmaj == None:
            print ('ERROR\tLTEmass: bmaj and bmin must be given\
             for conversion from K arcsec2 km/s --> Jy km/s.')
            return
        elif bmin == None:
            print ('ERROR\tLTEmass: bmaj and bmin must be given\
             for conversion from K arcsec2 km/s --> Jy km/s.')
            return

        # convert units
        bmaj = bmaj*np.pi/(180.*60.*60.) # arcsec --> radian
        bmin = bmin*np.pi/(180.*60.*60.) # arcsec --> radian
        Fv       = Fv*S_TA                         # K arcsec2 km/s --> Jy/beam arcesc2 km/s
        beamsize = np.pi/(4.*np.log(2.))*bmaj*bmin # beam --> arcsec2
        Fv       = Fv/beamsize                     # --> Jy km/s


    # the unit is (Jy km/s)
    Fv = Fv*1.0e-26         # Jy km/s --> MKS (1 Jy = 10^-26 Wm-2Hz-1)
    Fv = Fv*1.e7*1.e5*1.e-4 # MKS --> CGS (erg cm^-2 cm/s)

    # Mgas (assuming optically thin)
    Mgas = C1*np.exp(Eu/Tex)*dist*dist*Fv/Xconv #[g]
    Mgas = Mgas/Msun # [g] --> [Msun]
    #print ('Mgas: %.4f Msun'%Mgas)

    return Mgas


# calculate Virial mass
def Mvir(Tex, R, mu=2.37, dist=140.):
    '''
    Calculate Virial mass.

    Tex: excitation temperature (K)
    R: radius (arcsec)
    mu: mean molecular weight
    dist: distance to the object (pc)
    '''

    delv2 = 8.*np.log(2)*kb*Tex/(mu*mH) # line width ((cm/s)^2)
    delv2 = delv2*1e-10                 # (cm/s)^2 --> (km/s)^2
    #print 'del v: ', np.sqrt(delv2), ' km/s'
    R     = R*dist                      # arcsec --> au
    R     = R*auTOcm/pcTOcm             # au --> pc

    Mvir = 210*R*delv2 # Msun
    return Mvir