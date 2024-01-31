# -*- coding: utf-8 -*-


### Calculating Tb from Sigma_H2 & Tex
###  assuming LTE condition and optically thin


# import modules
import os
import sys
import math
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from astropy import constants, units


# constants (in cgs)

Ggrav  = constants.G.cgs.value        # Gravitational constant
ms     = constants.M_sun.cgs.value    # Solar mass (g)
ls     = constants.L_sun.cgs.value    # Solar luminosity (erg s^-1)
rs     = constants.R_sun.cgs.value    # Solar radius (cm)
au     = units.au.to('cm')            # 1 au (cm)
pc     = units.pc.to('cm')            # 1 pc (cm)
clight = constants.c.cgs.value        # light speed (cm s^-1)
kb     = constants.k_B.cgs.value      # Boltzman coefficient
hp     = constants.h.cgs.value        # Planck constant
sigsb  = constants.sigma_sb.cgs.value # Stefan-Boltzmann constant (erg s^-1 cm^-2 K^-4)
mp     = constants.m_p.cgs.value      # Proton mass (g)


# path to here
path_to_here = os.path.dirname(__file__)
path_to_library = path_to_here.strip('lteanalysis')
#print (path_to_here)
#print (path_to_library)


class LTEAnalysis():

    def __init__(self):
        self.moldata = {}

    # read molecular data
    def read_lamda_moldata(self, line):
        '''
        Read a molecular data file from LAMDA (Leiden Atomic and Molecular Database)
        '''
        # find path
        line = line.lower()
        if '+' in line: line = line.replace('+','p')
        infile = glob.glob(path_to_library+'moldata/'+line+'.dat')
        if len(infile) == 0:
            print('ERROR\tread_lamda_moldata: Cannot find LAMDA file.')
            print('ERROR\tread_lamda_moldata: Only C18O, CO, and N2H+ are \
                supported for now')
            return
        else:
            data = pd.read_csv(infile[0], comment='!', 
                delimiter='\r\n', header=None, engine='python')

        # get
        # line name, weight, nlevels
        _, weight, nlevels = data[0:3][0].values
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

        self.moldata[line] = {
        'weight':weight,
        'nlevels': nlevels,
        'EJ': EJ,
        'gJ': gJ,
        'J': J,
        'ntrans': ntrans,
        'Jup': Jup,
        'Jlow': Jlow,
        'Acoeff': Acoeff,
        'freq': freq,
        'delE': delE,
        }

        #return line, weight, nlevels, EJ, gJ, J, ntrans, Jup, Acoeff, freq, delE

    def get_intensity(self, line, Ju, Tex, Ncol, delv, lineprof='rect', 
        mode='lte', Xconv=None, Tbg=2.73, Tb=True, return_tau=False):
        '''
        Calculate the intensity or brightness temperature of a molecular line transition.
        Currently only LTE assumption is supported. Non-LTE calculation using RADEX will be 
        implemented in future.

        Parameter
        ---------
                line (str): Name of line.
                Ju (int): Upper excitation level.
                Tex (float): Excitation temperature (K).
                Ncol (float): Number column density of H2 or the molecule for which the intensity
                 is calculated (cm^-2).
                delv (float): FWHM of the line profile (cm s^-1).
                mode (str): LTE or non-LTE. Currently only LTE assumption is supported.
                Xconv (float): Conversion factor from the H2 column density to the column density of the
                 target molecule. If this is not given, then the input column density is regarded as the 
                 column density of the target molecule.
                Tbg (float): Temperature of the background emission (K).
                Tb: (bool): If True, the output will be in the brightness temperature. Otherwise, the output
                 will be in intensity in a unit of cgs.
                return_tau (bool): If True, the optical depth tau will be returned instead of the intensity.
        '''
        # line
        if line in self.moldata.keys():
            pass
        else:
            self.read_lamda_moldata(line)

        # line Ju --> Jl
        freq_ul = self.moldata[line]['freq'][Ju-1]*1e9 # Hz
        Aul     = self.moldata[line]['Acoeff'][Ju-1]
        gu      = self.moldata[line]['gJ'][Ju]
        gl      = self.moldata[line]['gJ'][Ju-1]
        EJu     = self.moldata[line]['EJ'][Ju]

        # partition function
        Qrot = Pfunc(self.moldata[line]['EJ'], self.moldata[line]['gJ'], 
            self.moldata[line]['J'], Tex)

        # N_H2 --> N_mol
        if Xconv: Ncol *= Xconv

        # tau_v
        tau_v = (clight*clight)/(8.*np.pi*freq_ul*freq_ul)*(gu/Qrot)\
        *np.exp(-EJu/Tex)*Ncol*Aul*(np.exp(hp*freq_ul/(kb*Tex)) - 1.)/delv

        if return_tau:
            return tau_v
        Iv = Bv(Tbg,freq_ul)*np.exp(-tau_v) + Bv(Tex, freq_ul)*(1. - np.exp(-tau_v))

        if Tb:
            # equivalent brightness temperature
            return (clight*clight/(2.*freq_ul*freq_ul*kb))*Iv
        else:
            return Iv


    def get_tbratio(self, line, Ju_u, Ju_l, Tex, Ncol, delv, lineprof='rect', 
        mode='lte', Xconv=None, Tbg=2.73):
        '''
        Get Tb ratio between two transitions.
        '''

        tb_u = self.get_intensity(line, Ju_u, Tex, Ncol, delv, lineprof=lineprof, 
            mode=mode, Xconv=Xconv, Tbg=Tbg, return_tau=False, Tb=True)
        tb_l = self.get_intensity(line, Ju_l, Tex, Ncol, delv, lineprof=lineprof, 
            mode=mode, Xconv=Xconv, Tbg=Tbg, return_tau=False, Tb=True)

        return tb_u/tb_l


    def makegrid(self, lines, J1, J2, Texes, Ncols, delv, lineprof='rect', 
        mode='lte', Xconv=[], Tbg=2.73, Tb=True, fig=None, ax=None, lw=1., aspect=1.):
        '''
        Produce a grid for the line intensity ratio for two transitions.

        Parameters
        ----------
            lines (str or list): Name of lines. A single line can be given as the str object.
             Two different lines can be given as a list of two line names.
            J1 (int): Upper excitation level of the first transition.
            J2 (int): Upper excitation level of the second transition.
            Texes (array): Array of excitation temperatures (K), with which the grid will be calculated.
            Ncols (array): Array of H2 or molecule's number column densities, with which 
             the grid will be calculated (cm^-2).
            delv (float): FWHM of the line profile (cm s^-1).
            mode (str): LTE or non-LTE. Currently only LTE assumption is supported.
            Xconv (float or list): Conversion factors from the H2 column density to the column density of the
             target molecule. Can be given as float for a single molecule, or as a list for two different kinds.
             If this is not given, then the input column density is regarded as the 
             column density of the target molecule.
            Tbg (float): Temperature of the background emission (K).
            Tb: (bool): If True, the output will be in the brightness temperature. Otherwise, the output
             will be in intensity in a unit of cgs.
        '''
        # Parameters
        if type(lines) is str:
            lines = [lines]*2
        elif type(lines) is list:
            if len(lines) !=2:
                print("ERROR\tget_grid: More than two elements are given for 'lines'.")
                print("ERROR\tget_grid: lines must be str or list with two elements.")
                return 0
        else:
            print("ERROR\tget_grid: Type of 'lines' is wrong.")
            print("ERROR\tget_grid: lines must be str or list with two elements.")
            return 0

        if type(Xconv) is float:
            Xconv = [Xconv]*2
        elif type(Xconv) is list:
            if len(Xconv) == 0:
                Xconv = [None]*2
            elif len(Xconv) != 2:
                print("ERROR\tget_grid: More than two elements are given for 'Xconv'.")
                print("ERROR\tget_grid: Xconv must be float or list with two elements.")
                return 0
        else:
            print("ERROR\tget_grid: Type of 'Xconv' is wrong.")
            print("ERROR\tget_grid: Xconv must be float or list with two elements.")
            return 0

        # figure
        if (fig is None) and (ax is None):
            fig = plt.figure()#figsize=(11.69, 8.27))
            ax  = fig.add_subplot(111)
        elif (fig is not None) and (ax is None):
            ax  = fig.add_subplot(111)


        st_idx = 127
        # iso-temperature curves
        for i, Tex_i in enumerate(Texes):
            tb1 = []
            tb2 = []
            
            for j, Ncol_i in enumerate(np.logspace(np.log10(Ncols[0]), np.log10(Ncols[-1]),128)):
                #print ('N, T: %.2e %.f'%(Ncol_i, Tex_i))
                tb1.append(self.get_intensity(lines[0], J1, Tex_i, Ncol_i, delv, 
                    lineprof=lineprof, mode=mode, Xconv=Xconv[0], Tbg=Tbg, return_tau=False, Tb=Tb))
                tb2.append(self.get_intensity(lines[1], J2, Tex_i, Ncol_i, delv, 
                    lineprof=lineprof, mode=mode, Xconv=Xconv[1], Tbg=Tbg, return_tau=False, Tb=Tb))

            ax.plot(tb1, tb2, c=cm.Dark2(float(i+1)/len(Texes)), lw=lw)
            ax.text(x = tb1[st_idx],y=tb2[st_idx], c =cm.Dark2(float(i+1)/len(Texes)), s = str(Tex_i)+"K",fontsize = 15)
            st_idx = int(st_idx/(i+1))



        # iso-density curves
        st_idx = 127
        for i, Ncol_i in enumerate(Ncols):
            tb1 = []
            tb2 = []
            for j, Tex_i in enumerate(np.linspace(Texes[0], Texes[-1],128)):
                #print ('N, T: %.2e %.f'%(Ncol_i, Tex_i))
                tb1.append(self.get_intensity(lines[0], J1, Tex_i, Ncol_i, delv, 
                    lineprof=lineprof, mode=mode, Xconv=Xconv[0], Tbg=Tbg, return_tau=False, Tb=Tb))
                tb2.append(self.get_intensity(lines[1], J2, Tex_i, Ncol_i, delv, 
                    lineprof=lineprof, mode=mode, Xconv=Xconv[1], Tbg=Tbg, return_tau=False, Tb=Tb))

            ax.plot(tb1, tb2, c='k', lw=lw)#)cm.BrBG(float(i+0.5)/len(Ncols)))
#            ax.text(x = tb1[st_idx],y=tb2[st_idx], c ='k', s = f"{Ncol_i:.0e}cm\u00b2",fontsize = 15)
            st_idx = int(st_idx/(i+1))


        # aspect ratio
        ax.set_xlabel(r'$T_\mathrm{b}$(%i-%i)'%(J1, J1-1))
        ax.set_ylabel(r'$T_\mathrm{b}$(%i-%i)'%(J2, J2-1))
        change_aspect_ratio(ax, aspect)

        return fig, ax





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
    Zarray = np.array([gJ[j]*np.exp(-EJ[j]/Tex) for j in range(len(J))])
    Z      = np.sum(Zarray)
    return Z


# planck function
def Bv(T,v):
    '''
    Planck function

    Args:
        T: temprature [K]
        v: frequency [Hz]
    '''
    exp   = np.exp((hp*v)/(kb*T))-1.0
    fterm = (2.0*hp*v*v*v)/(clight*clight)
    Bv    = fterm/exp
    return Bv



def change_aspect_ratio(ax, ratio, plottype='linear'):
    '''
    This function change aspect ratio of figure.
    Parameters:
        ax: ax (matplotlit.pyplot.subplots())
            Axes object
        ratio: float or int
            relative x axis width compared to y axis width.
    '''
    if plottype == 'linear':
        aspect = (1/ratio) *(ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    elif plottype == 'loglog':
        aspect = (1/ratio) *(np.log10(ax.get_xlim()[1]) - np.log10(ax.get_xlim()[0])) / (np.log10(ax.get_ylim()[1]) - np.log10(ax.get_ylim()[0]))
    elif plottype == 'linearlog':
        aspect = (1/ratio) *(ax.get_xlim()[1] - ax.get_xlim()[0]) / np.log10(ax.get_ylim()[1]/ax.get_ylim()[0])
    elif plottype == 'loglinear':
        aspect = (1/ratio) *(np.log10(ax.get_xlim()[1]) - np.log10(ax.get_xlim()[0])) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    else:
        print('ERROR\tchange_aspect_ratio: plottype must be choosen from the types below.')
        print('   plottype can be linear or loglog.')
        print('   plottype=loglinear and linearlog is being developed.')
        return

    aspect = np.abs(aspect)
    aspect = float(aspect)
    ax.set_aspect(aspect)