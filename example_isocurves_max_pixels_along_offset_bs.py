import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 18

from lteanalysis import LTEAnalysis
import pandas as pd


# ----- input -----
# for LTE calculation
line  = 'c18o'
Xconv = 1e-7
delv  = 0.2 # km/s
ilines = [3,2] # Ju
Ncols = np.array([5.e15, 9e15, 1.8e16, 3.e16, 1e19]) # cm^-2  
Texes = np.array([5, 17.5,  20,  22., 30., 40.]) # K


# Tb_dict_bs = pd.read_csv(filepath_or_buffer='Tb_dict_blueshifted_side.csv', sep=',',header='infer',index_col=False)
#Tb_dict_rs = pd.read_csv(filepath_or_buffer='Tb_dict_redshifted_side.csv', sep=',',header='infer',index_col=False)

#Tb_dict_bs = Tb_dict_bs['Tb_max_pix_bs_b7', 'Tb_max_pix_bs_b6',  'V', 'R_arcsec']

#Tb_dict_rs = Tb_dict_rs['Tb_max_pix_rs_b7', 'Tb_max_pix_rs_b6', 'V', 'R_arcsec']

blue_shifted_pairs_outside_gap_outer_edge = np.array([ 
                                            [10.4,   7.71, -1.51, -3.138],      
                                            [12.54,   9.05, -1.72103403, -2.469],  
                                            ])

blue_shifted_pairs_outside_gap_inner_edge = np.array([   
                                            [11.62, 9.8, -2.554, -1.122], 
                                            [11.60,  10.31, -2.767, 0.958],           
                                            [12.49, 10.25, -2.970,-0.824]   
                                            ])

blue_shifted_pairs_in_gap = np.array([ 
                                      [11.0,  8.1, -2.13774151, -1.602],
                                      [9.74,  8.8, -1.92938777, -1.959], 
                                    ])


model = LTEAnalysis()
model.read_lamda_moldata(line)
fig, ax = model.makegrid(line, ilines[0], ilines[1], Texes, Ncols, delv, Xconv=Xconv, lw=1.)
ax.set_xlim(0., 15)
ax.set_ylim(0., 15)


plot_Tb_points = True

if plot_Tb_points:

    ax.errorbar(blue_shifted_pairs_outside_gap_outer_edge[:,0], blue_shifted_pairs_outside_gap_outer_edge[:,1], xerr=1.01, yerr=0.43,
        color='blue', marker='s', ls='none', label = r'r > r$_{dep}$')
    for row_idx in range(blue_shifted_pairs_outside_gap_outer_edge.shape[0]):
        point_coord = blue_shifted_pairs_outside_gap_outer_edge[row_idx,:]
        ax.annotate(text = f"{int(point_coord[3]*140)}AU", xy = (point_coord[0],point_coord[1]), xytext = (25,-25), textcoords='offset points',
                            ha='center', va='bottom')

    ax.errorbar(blue_shifted_pairs_outside_gap_inner_edge[:,0], blue_shifted_pairs_outside_gap_inner_edge[:,1], xerr=1.01, yerr=0.43,
        color='blue', marker='o', ls='none', label = r'r < r$_{dep}$')
    for row_idx in range(blue_shifted_pairs_outside_gap_inner_edge.shape[0]):
        point_coord = blue_shifted_pairs_outside_gap_inner_edge[row_idx,:]
        ax.annotate(text = f"{int(point_coord[3]*140)}AU", xy = (point_coord[0],point_coord[1]), xytext = (35,6), textcoords='offset points',
                            ha='center', va='bottom')
        
    ax.errorbar(blue_shifted_pairs_in_gap[:,0], blue_shifted_pairs_in_gap[:,1], xerr=1.01, yerr=0.43,
        color='blue', marker='x', ls='none', label = r'r $\approx$ r$_{dep}$')
    for row_idx in range(blue_shifted_pairs_in_gap.shape[0]):
        point_coord = blue_shifted_pairs_in_gap[row_idx,:]
        ax.annotate(text = f"{int(point_coord[3]*140)}AU", xy = (point_coord[0],point_coord[1]), xytext = (28,6), textcoords='offset points',
                            ha='center', va='bottom')

'''
    ax.errorbar(red_shifted_pairs_outside_gap_outer_edge[:,0], red_shifted_pairs_outside_gap_outer_edge[:,1], xerr=1.01, yerr=0.43,
        color='red',  marker='s', ls='none')
    for row_idx in range(red_shifted_pairs_outside_gap_outer_edge.shape[0]):
        point_coord = red_shifted_pairs_outside_gap_outer_edge[row_idx,:]
        ax.annotate(text = f"{int(point_coord[3]*140)} AU", xy = (point_coord[0],point_coord[1]), xytext = (15,-15), textcoords='offset points',
                            ha='center', va='bottom')
        
    ax.errorbar(red_shifted_pairs_outside_gap_inner_edge[:,0], red_shifted_pairs_outside_gap_inner_edge[:,1], xerr=1.01, yerr=0.43,
        color='red',  marker='o', ls='none')
    for row_idx in range(red_shifted_pairs_outside_gap_inner_edge.shape[0]):
        point_coord = red_shifted_pairs_outside_gap_inner_edge[row_idx,:]
        ax.annotate(text = f"{int(point_coord[3]*140)} AU", xy = (point_coord[0],point_coord[1]), xytext = (17,-19), textcoords='offset points',
                            ha='center', va='bottom')
        
    ax.errorbar(red_shifted_pairs_in_gap[:,0], red_shifted_pairs_in_gap[:,1], xerr=1.01, yerr=0.43,
        color='red', marker='x', ls='none')
    for row_idx in range(red_shifted_pairs_in_gap.shape[0]):
        point_coord = red_shifted_pairs_in_gap[row_idx,:]
        ax.annotate(text = f"{int(point_coord[3]*140)} AU", xy = (point_coord[0],point_coord[1]), xytext = (15,-15), textcoords='offset points',
                            ha='center', va='bottom')
'''

plt.legend()
plt.show()
# -----------------------