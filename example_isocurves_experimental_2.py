import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 18

from lteanalysis import LTEAnalysis


# ----- input -----
# for LTE calculation
line  = 'c18o'
Xconv = 1e-7
delv  = 0.2 # km/s
ilines = [3,2] # Ju
Ncols = np.array([5.e15, 1.e17, 1.e19,]) # cm^-2   # Note the distinction in column density, we have gotten gown a whole order of magnitude
Texes = np.array([5., 14., 16.5, 22., 30., 40.]) # K



# Following arrays are in this form [[Tb_band7, Tb_band_6, Rotational_velocity,offset]]

blue_shifted_pairs_outside_gap_outer_edge = np.array([ 
                                            [11.57,   7.29, -1.51, -3.2],      
                                            [10.62,   6.67, -1.72103403, -2.5],  
                                            ])

blue_shifted_pairs_outside_gap_inner_edge = np.array([   
                                            # [10.70,  6.88, -2.34609525, -1.38],
                                            [9.67,9.84,-2.554, -1.1]      # extra added on early 18/8
                                            ])

blue_shifted_pairs_in_gap = np.array([ 
                                      # [6.79325724,  5.21653748, -2.13774151, -1.62],
                                      [5.79637718,  6.10198784, -1.92938777, -2.0],   # extra point
                                    ])

red_shifted_pairs_outside_gap_inner_edge = np.array([ 
                                            [8.64, 7.38, 2.858, 0.9],
                                            [9.24, 9.12,  3.071, 0.8]])

red_shifted_pairs_outside_gap_outer_edge = np.array([
                                                [12.42, 9.79,  1.818, 1.8]
                                                ])

red_shifted_pairs_in_gap = np.array([
                                    [6.71466255, 7.6401782,  2.23768703, 1.4],
                                    [5.84474993, 4.46442366, 2.44604077, 1.2],
                                    [5.88139868, 5.22014475, 2.65439451, 1.0]
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

ax.errorbar(blue_shifted_pairs_outside_gap_outer_edge[:,0], blue_shifted_pairs_outside_gap_outer_edge[:,1], xerr=1.01, yerr=0.43,
    color='blue', marker='s', ls='none', label = 'points outside depletion region')
for row_idx in range(blue_shifted_pairs_outside_gap_outer_edge.shape[0]):
    point_coord = blue_shifted_pairs_outside_gap_outer_edge[row_idx,:]
    ax.annotate(text = f"{point_coord[3]}", xy = (point_coord[0],point_coord[1]), xytext = (15,-15), textcoords='offset points',
                        ha='center', va='bottom')

ax.errorbar(blue_shifted_pairs_outside_gap_inner_edge[:,0], blue_shifted_pairs_outside_gap_inner_edge[:,1], xerr=1.01, yerr=0.43,
    color='blue', marker='o', ls='none')
for row_idx in range(blue_shifted_pairs_outside_gap_inner_edge.shape[0]):
    point_coord = blue_shifted_pairs_outside_gap_inner_edge[row_idx,:]
    ax.annotate(text = f"{point_coord[3]}", xy = (point_coord[0],point_coord[1]), xytext = (15,-15), textcoords='offset points',
                        ha='center', va='bottom')
    
ax.errorbar(blue_shifted_pairs_in_gap[:,0], blue_shifted_pairs_in_gap[:,1], xerr=1.01, yerr=0.43,
    color='blue', marker='x', ls='none')
for row_idx in range(blue_shifted_pairs_in_gap.shape[0]):
    point_coord = blue_shifted_pairs_in_gap[row_idx,:]
    ax.annotate(text = f"{point_coord[3]}", xy = (point_coord[0],point_coord[1]), xytext = (15,-15), textcoords='offset points',
                        ha='center', va='bottom')


ax.errorbar(red_shifted_pairs_outside_gap_outer_edge[:,0], red_shifted_pairs_outside_gap_outer_edge[:,1], xerr=1.01, yerr=0.43,
    color='red',  marker='s', ls='none')
for row_idx in range(red_shifted_pairs_outside_gap_outer_edge.shape[0]):
    point_coord = red_shifted_pairs_outside_gap_outer_edge[row_idx,:]
    ax.annotate(text = f"{point_coord[3]}", xy = (point_coord[0],point_coord[1]), xytext = (15,-15), textcoords='offset points',
                        ha='center', va='bottom')
    
ax.errorbar(red_shifted_pairs_outside_gap_inner_edge[:,0], red_shifted_pairs_outside_gap_inner_edge[:,1], xerr=1.01, yerr=0.43,
    color='red',  marker='o', ls='none')
for row_idx in range(red_shifted_pairs_outside_gap_inner_edge.shape[0]):
    point_coord = red_shifted_pairs_outside_gap_inner_edge[row_idx,:]
    ax.annotate(text = f"{point_coord[3]}", xy = (point_coord[0],point_coord[1]), xytext = (15,-15), textcoords='offset points',
                        ha='center', va='bottom')
    
ax.errorbar(red_shifted_pairs_in_gap[:,0], red_shifted_pairs_in_gap[:,1], xerr=1.01, yerr=0.43,
    color='red', marker='x', ls='none')
for row_idx in range(red_shifted_pairs_in_gap.shape[0]):
    point_coord = red_shifted_pairs_in_gap[row_idx,:]
    ax.annotate(text = f"{point_coord[3]}", xy = (point_coord[0],point_coord[1]), xytext = (15,-15), textcoords='offset points',
                        ha='center', va='bottom')

plt.legend()
plt.show()
# -----------------------