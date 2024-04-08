import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import bettermoments as bm
import pandas as pd
import matplotlib.pyplot as plt
from astropy import constants, units
import numpy as np

current_path = os.getcwd()
# Check if it is office computer or laptop and set path of imfits accordingly
if current_path.split(sep=':')[0] == 'D':                           # Office computer
    sys.path.append("D:\L1489_IRS_ssp\imfits")
else:                                                               # Laptop N
    sys.path.append("E:/Mihir_new/ASIAA-SSP/imfits/")

from imfits import Imfits
from imfits.drawmaps import AstroCanvas


def curve_function(v_rot, v_100):
        
        r_au = 100.*np.power(x1 = (v_100/v_rot), x2 = 2)

        return r_au

class pv_analyze():

    G_grav  = constants.G.value        # Gravitational constant
    M_sun   = constants.M_sun.value    # Solar mass (g)


    obj_name = 'L1489_irs'
    inclination = 73.   # Inclination in degree 
    PA = 69.            # PA im degree
    M_star = 1.6        # Mass of star in units of M_sun
    distanc_pc = 140.   # Distance of object in parsec
    v_sys = 7.3         # Systemic velocity in kmps

    def __init__(self,
                pv_path = None,
                is_Tb = True,
                line_name = 'J_2_1'          
                ):

                self.pv_path = pv_path
                self.is_Tb = is_Tb
                self.pv = None
                self.rms = None
                self.v_obs = None
                self.v_rot = None
                self.v_rot_redshifted = None
                self.v_rot_blueshifted = None
                self.x_axis = None





    def read_PV(self, print_details = True):

        """
        Reads PV data and sets up attributes of velocities.
        parameters:
        -----------
        print_details:  `bool`, If True, prints name of imfits PV object and rms in units of Jy/beam and units of Kelvin if `is_Tb` = True
        """

        # Read data and convert intensity to Tb units
        pv = Imfits(self.pv_path, pv=True) # Add the pv option
        self.rms = pv.estimate_noise()
     
        if print_details:
            print(f" rms = {pv.estimate_noise() :.2e} Jy beam$^-1$")

        if self.is_Tb:
            pv.convert_units(conversion='IvtoTb')
            self.rms = pv.estimate_noise()
            print(fr" rms = {self.rms :.2f} K")

        self.pv = pv        # PV imfits object

        self.v_obs = pv.vaxis
        self.v_rot = np.array(self.v_obs - self.v_sys) 
        self.v_rot_redshifted = self.v_rot[self.v_rot > 0]
        self.v_rot_blueshifted = self.v_rot[self.v_rot <= 0]
        self.x_axis = pv.xaxis
    
        return
    
    def _get_pixel_ridx_on_curve(self, x_data = None, x_curve = None):       
        # Find x indices of pixels in which the curve points fall
        _idx = lambda x_data, x_curve: [np.abs(x_data - b).argmin() for b in x_curve]
        return _idx

    def _get_pixel_vidx_on_curve(self, v_obs=None, v_rot=None, v_sys=None, v_tol=None):
        # Find velocity indices of rs and bs (we try to find v_obs which are equal to v_rot_bs + v_sys with a tolerance of 0.05 kmps as velocity channels are seperated by 0.2 kmps) )
        _idx = np.where(np.abs(v_obs[:, None] - (v_rot + v_sys)) <= v_tol)[0]
        return _idx


    def get_Tb_on_curve(self, curve_function = None, get_surrounding_pix = False, num_pix = None, cf_kwargs = None):
         

        # Get radial corrdinates of point on given curve in units of AU 
        if curve_function is not None:
            # using a customized function to find r_au corresponding to observed velocity channels
            r_au_rs = curve_function(self.v_rot_redshifted, **cf_kwargs)
            r_au_bs = curve_function(self.v_rot_blueshifted, **cf_kwargs)
            #r_au_rs = 100*pow(base = v_100/v_rot_redshifted, exp=2)
            #r_au_bs = -100*pow(base = v_100/v_rot_blueshifted, exp=2)

        else:          
             # use 2D keplerian velocity profile 
             r_au_rs = (self.G_grav*self.M_star*self.M_sun/(self.v_rot_redshifted*1.e3/np.sin(self.inclination*np.pi/180.)))/1.496e11
             r_au_bs = (self.G_grav*self.M_star*self.M_sun/(self.v_rot_blueshifted*1.e3/np.sin(self.inclination*np.pi/180.)))/1.496e11         

        # Convert raddial coordinates to arcsec
        self.r_as_rs = r_au_rs/self.distanc_pc        # (for points in redshifted side) radial distance from star in arcsec 
        self.r_as_bs = r_au_bs/self.distanc_pc        # (for points in blueshifted side) radial distance from star in arcsec 

        # dropping r larger then rmax of pv image and corresponding Vs as well
        self.v_rot_redshifted, self.r_as_rs = self.v_rot_redshifted[self.r_as_rs <= self.x_axis.max()], self.r_as_rs[self.r_as_rs <= self.x_axis.max()]
        self.v_rot_blueshifted, self.r_as_bs = self.v_rot_blueshifted[self.r_as_bs >= self.x_axis.min()], self.r_as_bs[self.r_as_bs >= self.x_axis.min()]

        # Find x indices of pixels in which the curve points fall
        self.r_as_rs_idx = self._get_pixel_ridx_on_curve(x_data = self.x_axis, x_curve = self.r_as_rs)
        self.r_as_bs_idx = self._get_pixel_ridx_on_curve( x_data = self.x_axis, x_curve = self.r_as_bs)

        self.vidx_rs = self._get_pixel_vidx_on_curve(v_obs = self.v_obs, v_rot = self.v_rot_redshifted, v_sys = self.v_sys, v_tol = 0.005)
        self.vidx_bs = self._get_pixel_vidx_on_curve(v_obs = self.v_obs, v_rot = self.v_rot_blueshifted, v_sys = self.v_sys, v_tol = 0.005)

        # Get temperature values of pixels falling on keplerian points
        #pv_b6_data = pv_b6.data[0]
        #pv_b7_data = pv_b7.data[0]

        # Get the Tb values on keplerian points
        #t_kep_b6_rs = pv_b6_data[vidx_rs, r_rs_real_idx]
        # t_kep_b6_bs = pv_b6_data[vidx_bs, r_bs_real_idx]


        self.pv_data = self.pv.data[0] # self.pv.data has a redundant z axis of length one since we are considering only one stokes axis
        
        if get_surrounding_pix:
             return
             
             
             

        
        else:
            Tb_on_point_rs = self.pv_data[self.vidx_rs, self.r_as_rs_idx]
            Tb_on_point_bs = self.pv_data[self.vidx_bs, self.r_as_bs_idx]
            data_cube = {'Tb_on_point_rs':Tb_on_point_rs, 'Tb_on_point_bs':Tb_on_point_bs}

        return data_cube
         

         





