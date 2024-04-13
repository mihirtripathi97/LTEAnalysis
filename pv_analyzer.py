import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import constants


current_path = os.getcwd()
# Check if it is office computer or laptop and set path of imfits accordingly
# Office computer
if current_path.split(sep=":")[0].lower() == "d":
    sys.path.append("D:\L1489_IRS_ssp\imfits")
    from imfits.drawmaps import AstroCanvas
    from imfits import Imfits
else:  # Laptop N
    sys.path.append("E:/Mihir_new/ASIAA-SSP/imfits/")
    from imfits.drawmaps import AstroCanvas
    from imfits import Imfits


class pv_analyze:
    """
    Analyzes a PV diagramm
    """

    G_grav = constants.G.value  # Gravitational constant (MKS)
    M_sun = constants.M_sun.value  # Solar mass (Kg)

    obj_name = "L1489_irs"
    inclination = 73.0  # Inclination in degree
    PA = 69.0  # PA im degree
    M_star = 1.6  # Mass of star in units of M_sun
    distanc_pc = 140.0  # Distance of object in parsec
    v_sys = 7.22  # Systemic velocity in kmps

    def __init__(self, pv_path=None, is_Tb=True, line_name="J_2_1"):

        self.pv_path = pv_path
        self.is_Tb = is_Tb
        self.pv = None  # Imfits PV object
        self.rms = None  # Root mean squared error of PV
        self.pv_data = None  # 2D PV image data extracted from self.pv
        self.v_obs = None  # Observed velocities
        # Observed velocities converted to rotational velocities (v_rot = V_obs - Vsys)
        self.v_rot = None
        # Redshifted (positive) rotational velocities for points on curve, cropped till the extent of PV plot
        self.v_rot_redshifted = None
        # Blueshifted (positive) rotational velocities for points on curve, cropped till the extent of PV plot
        self.v_rot_blueshifted = None
        self.x_axis = None  # Observed x values (in arcec)
        # x coordinates of Keplerian or other Curve points in redshifted side (in arcsec)
        self.r_as_rs = None
        # x coordinates of Keplerian or other Curve points in blue shifted side (in arcsec)
        self.r_as_bs = None

    def read_pv(self, print_details=True):
        """
        Reads PV data and sets up attributes of velocities.
        parameters:
        -----------
        print_details:  `bool`, If True, prints name of imfits PV object and rms in units of Jy/beam and units of Kelvin if `is_Tb` = True
        """

        # Read data and convert intensity to Tb units
        pv = Imfits(self.pv_path, pv=True)
        self.rms = pv.estimate_noise()

        if print_details:
            print(f" rms = {pv.estimate_noise() :.2e} Jy beam$^-1$")

        if self.is_Tb:
            pv.convert_units(conversion="IvtoTb")
            self.rms = pv.estimate_noise()
            print(rf" rms = {self.rms :.2f} K")

        self.pv = pv  # PV imfits object

        self.v_obs = pv.vaxis
        self.v_rot = np.array(self.v_obs - self.v_sys)
        self.v_rot_redshifted = self.v_rot[self.v_rot > 0]
        self.v_rot_blueshifted = self.v_rot[self.v_rot <= 0]
        self.x_axis = pv.xaxis

        return

    def _get_pixel_ridx_on_curve(self, x_data=None, x_curve=None):
        # Find x indices of pixels in which the curve points fall
        _idx = [np.abs(x_data - b).argmin() for b in x_curve]
        return _idx

    def _get_pixel_vidx_on_curve(self, v_obs=None, v_rot=None, v_sys=None, v_tol=None):
        # Find velocity indices of rs and bs (we try to find v_obs which are equal to 
        # v_rot_bs + v_sys with a tolerance of 0.05 kmps as velocity channels are seperated by 0.2 kmps) )
        v_sum = v_rot + v_sys
        _idx = [np.abs(v_obs - v).argmin() for v in v_sum]
        # _idx = np.where(np.abs(v_obs[:, None] - (v_rot + v_sys)) <= v_tol)[0]
        return _idx

    def get_tb_on_curve(
        self,
        mode:str = {'func', 'vals'},
        curve_function:list = None,
        get_surrounding_pix:bool = False,
        get_pix_along:str = {'r','v'},
        num_pix:int = 4,
        return_avg:bool = True,
        cf_kwargs = None,
    ) -> dict[np.ndarray, np.ndarray]:
        """
        Returns a pandas dataframe where each column is a temperature at pixels on/surrounding the curve points in each velocity channel

        Parameters:
        -----------
        mode                :   `str`, {'func', 'vals'}, whether the curve function is a funcrion or list of rs and vs
        curve_function      :   `callable` or list, optional, default: None,
                                A function f(v_rotational) -> r, typically an inverse of kepler velocity function.
                                It provides angular distance `r` at which one should expect the emission comming from material rotating at
                                keplerian velocities. More generally, a function that returs r coordinates in units of AU
                                wrt center of disk for given velocity channels.
                                Default - a inverse kepler function GM/(v_rot**2) --> r

                                If mode = 'vals', then a list of lists, specifying r and v values in following order
                                [r_rs (au)] , [v_rs (kmps)], [r_bs (au)], [v_bs (kmps)]

        get_surrounding_pix :   `bool`, optional, If True then returns dataframe with Tb values in surrounding `num_pix` pixels
        get_pix_along       :   `str`, `r` | `v`, optional, whether to get pixels alnong fix r or fix v
        num_pix             :   `int` or `None`, optional, default: None, number of pixels around the pixel on which curve points fall for which Tb values are to be returned
        cf_kwargs           :    dict, optional,
                                 dictionary of other arguments needed to evaluet the `cost_function()`

        -------
        Returns
        -------

        Tb_cube             :   `pandas.DataFrame`,

                                With columns "Tb_on_pt_rs" and "Tb_on_pt_bs" if `get_surrounding_pix` is `False` or with columns "Tb_sur_pt_rs" and "Tb_sur_pt_bs", if `get_surrounding_pix` is `True`

                                - column1: numpy.ndarray
                                - column2: numpy.ndarray

        """

        # Get radial corrdinates of point on given curve in units of AU
        if mode.lower() == 'func':
        
            if curve_function is not None:
                # using a customized function to find r_au corresponding to observed velocity channels
                r_au_rs = curve_function(self.v_rot_redshifted, **cf_kwargs)
                r_au_bs = -1.0 * curve_function(self.v_rot_blueshifted, **cf_kwargs)

            else:
                # use 2D keplerian velocity profile
                r_au_rs = (self.G_grav*self.M_star*self.M_sun/ (self.v_rot_redshifted* 1.0e3/ np.sin(self.inclination * np.pi / 180.0))** 2) / 1.496e11
                r_au_bs = (self.G_grav* self.M_star* self.M_sun/ (self.v_rot_blueshifted* 1.0e3/ np.sin(self.inclination * np.pi / 180.0))** 2) / 1.496e11
        
        elif mode.lower() == 'vals':

            r_au_rs, self.v_rot_redshifted  = curve_function[0], curve_function[1]
            r_au_bs, self.v_rot_blueshifted = curve_function[2], curve_function[3]
        
        else:
            print("Please specify currect mode.")
            return 0
        
        # Convert raddial coordinates to arcsec
        # (for points in redshifted side) radial distance from star in arcsec
        self.r_as_rs = r_au_rs / self.distanc_pc
        # (for points in blueshifted side) radial distance from star in arcsec
        self.r_as_bs = r_au_bs / self.distanc_pc

        # dropping r larger then rmax of pv image and corresponding Vs as well
        self.v_rot_redshifted, self.r_as_rs = (
            self.v_rot_redshifted[self.r_as_rs <= self.x_axis.max()],
            self.r_as_rs[self.r_as_rs <= self.x_axis.max()],
        )
        self.v_rot_blueshifted, self.r_as_bs = (
            self.v_rot_blueshifted[self.r_as_bs >= self.x_axis.min()],
            self.r_as_bs[self.r_as_bs >= self.x_axis.min()],
        )

        # Find x indices of pixels in which the curve points fall
        self.r_as_rs_idx = self._get_pixel_ridx_on_curve(
            x_data=self.x_axis, x_curve=self.r_as_rs
        )
        self.r_as_bs_idx = self._get_pixel_ridx_on_curve(
            x_data=self.x_axis, x_curve=self.r_as_bs
        )

        self.vidx_rs = self._get_pixel_vidx_on_curve(
            v_obs=self.v_obs, v_rot=self.v_rot_redshifted, v_sys=self.v_sys, v_tol=0.005
        )


        self.vidx_bs = self._get_pixel_vidx_on_curve(
            v_obs=self.v_obs,
            v_rot=self.v_rot_blueshifted,
            v_sys=self.v_sys,
            v_tol=0.005,
        )

        # self.pv.data has a redundant z axis of length one. 
        # Since we are considering that image have only one stokes axis. Let's get rid of that
        self.pv_data = self.pv.data[0]

        if get_surrounding_pix:

            # NOTE: not yet tested properly 
            if num_pix is None or not isinstance(num_pix, int):
                raise TypeError(
                    "`num_pix` has to be an integer to get the Tb from surrounding pixels"
                )

            tb_sur_pt_rs = []
            tb_sur_pt_bs = []

            # Geather Tb values along redshifted side
            for pt_v_idx, pt_r_idx in zip(self.vidx_bs, self.r_as_bs_idx):

                # else:
                tb_surroundings = []

                for i in range(-num_pix, num_pix):

                    if get_pix_along == 'r':
                        tb_surroundings.append(float(self.pv_data[pt_v_idx, pt_r_idx + i]))
                    elif get_pix_along == 'v':
                        tb_surroundings.append(float(self.pv_data[pt_v_idx+i, pt_r_idx]))
                
                if return_avg:
                    tb_sur_pt_bs.append(np.mean(tb_surroundings))
                else:
                    tb_sur_pt_bs.append(tb_surroundings)
            
            # Geather Tb values along redshifted side
            for pt_v_idx, pt_r_idx in zip(self.vidx_rs, self.r_as_rs_idx):

                tb_surroundings = []

                for i in range(-num_pix, num_pix):

                    if get_pix_along == 'r':
                        tb_surroundings.append(float(self.pv_data[pt_v_idx, pt_r_idx + i]))
                    elif get_pix_along == 'v':
                        tb_surroundings.append(float(self.pv_data[pt_v_idx+i, pt_r_idx]))
                
                if return_avg:
                    tb_sur_pt_rs.append(np.mean(tb_surroundings))
                else:
                    tb_sur_pt_rs.append(tb_surroundings)

            data_cube = {"Tb_sur_pt_rs": tb_sur_pt_rs, "Tb_sur_pt_bs": tb_sur_pt_bs}

        else:

            tb_on_point_rs = []
            tb_on_point_bs = []

            for pt_v_idx, pt_r_idx in zip(self.vidx_rs, self.r_as_rs_idx):
                tb_on_point_rs.append(float(self.pv_data[pt_v_idx, pt_r_idx]))

            for pt_v_idx, pt_r_idx in zip(self.vidx_bs, self.r_as_bs_idx):
                tb_on_point_bs.append(float(self.pv_data[pt_v_idx, pt_r_idx]))

            data_cube = {
                "Tb_on_point_rs": tb_on_point_rs,
                "Tb_on_point_bs": tb_on_point_bs,
            }

        return data_cube

    def plot_pv(self, plot_curve: bool = False, **kwargs):
        """
        Plots PV diagram and overplots curve points if `plot_curve` is True. Returns the fig, axes object.
        """
        # Work in progress

        canvas = AstroCanvas((1, 1))
        pv_plot = canvas.pvdiagram(
            self.pv,
            vrel=True,
            color=True,
            cmap="inferno",
            vmin=-2.0,
            vmax=14.0,
            contour=True,
            clip=0.0000000,
            ylim=[-8.5, 6.5],
            clevels=np.array([3, 7, 10, 15, 25, 35, 45]) * self.rms,
            # If true, offset (radial distance from star) will be the x axis
            x_offset=True,
            vsys=self.v_sys,  # systemic velocity
            # plot vertical center (systemic velocity)
            ln_var=True,
            # plot horizontal center (zero offset)
            ln_hor=True,
            cbaroptions=("right", "3%", "3%"),
            cbarlabel=r"(Jy beam$^{-1})$",
            colorbar=True,
        )

        if plot_curve:
            pv_plot[0].scatter(self.r_as_rs,self.v_rot_redshifted, c='red')
            pv_plot[0].scatter(self.r_as_bs,self.v_rot_blueshifted,  c ='blue')
        
        
        # mark extracted pixels
            pv_plot[0].scatter(self.x_axis[self.r_as_rs_idx],self.v_obs[self.vidx_rs] - self.v_sys,  marker = 'x', color = 'k')
            pv_plot[0].scatter(self.x_axis[self.r_as_bs_idx],self.v_obs[self.vidx_bs] - self.v_sys,  marker = 'x', color = 'white')

        return pv_plot