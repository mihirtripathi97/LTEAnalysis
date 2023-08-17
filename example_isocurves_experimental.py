import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 12

from lteanalysis import LTEAnalysis


# ----- input -----
# for LTE calculation
line  = 'c18o'
Xconv = 1e-7
delv  = 0.5 # km/s
ilines = [3,2] # Ju
Ncols = np.array([1.e16, 1.e17, 1.e18, 1.e19,]) # cm^-2
Texes = np.array([5., 13.5, 16.5, 22., 30., 40.]) # K



# Following arrays are in this form [[Tb_band7, Tb_band_6, Rotational_velocity,offset]]

blue_shifted_pairs_outside_gap = np.array([ 
                                            [11.57,   7.29, -1.51, -1.14],      
#                                            [5.79637718,  6.10198784, -1.92938777, -1.98],
                                            [10.62,   6.67, -1.72103403, -2.52]
                                            ])

blue_shifted_pairs_in_gap = np.array([
                                      #[10.23020935,  6.42316008, -2.34609525, -1.38],
                                      [6.79325724,  5.21653748, -2.13774151, -1.62],
                                      #[5.79637718,  6.10198784, -1.92938777, -1.98],
                                    ])

red_shifted_pairs_outside_gap = np.array([ 
                                            [9.24, 9.12,  3.071, 1.44],
                                            [12.42, 10.18,  1.818, 1.44]
                                            #[6.71466255, 7.6401782,  2.23768703, 1.44],
                                            #[5.88139868, 5.22014475, 2.65439451, 1.02]
                                        ])

red_shifted_pairs_in_gap = np.array([
                                    [5.84474993, 4.46442366, 2.44604077, 1.2]
                                     ])



#data = [6., 6.,] # Tb3-2, Tb2-1
#e_data = [1., 0.5] # Error of data
# ------------------



# -------- start --------
model = LTEAnalysis()
model.read_lamda_moldata(line)
fig, ax = model.makegrid(line, ilines[0], ilines[1], Texes, Ncols, delv, Xconv=Xconv, lw=1.)
ax.set_xlim(0., 15)
ax.set_ylim(0., 15)

ax.errorbar(blue_shifted_pairs_outside_gap[:,0], blue_shifted_pairs_outside_gap[:,1], xerr=0.1, yerr=0.1,
    color='blue', marker='o', ls='none')
ax.errorbar(blue_shifted_pairs_in_gap[:,0], blue_shifted_pairs_in_gap[:,1], xerr=0.1, yerr=0.1,
    color='blue', marker='x', ls='none')

ax.errorbar(red_shifted_pairs_outside_gap[:,0], red_shifted_pairs_outside_gap[:,1], xerr=0.1, yerr=0.1,
    color='red',  marker='o', ls='none')
ax.errorbar(red_shifted_pairs_in_gap[:,0], red_shifted_pairs_in_gap[:,1], xerr=0.1, yerr=0.1,
    color='red', marker='x', ls='none')

plt.show()
# -----------------------