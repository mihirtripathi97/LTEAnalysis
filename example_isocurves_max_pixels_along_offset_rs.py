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
Ncols = np.array([5.e15, 9e15, 1.8e16, 3.e16, 1e19]) # cm^-2   # Note the distinction in column density, we have gotten gown a whole order of magnitude
Texes = np.array([5, 16,  19,  22., 30., 40.]) # K


# Tb_dict_bs = pd.read_csv(filepath_or_buffer='Tb_dict_blueshifted_side.csv', sep=',',header='infer',index_col=False)
#Tb_dict_rs = pd.read_csv(filepath_or_buffer='Tb_dict_redshifted_side.csv', sep=',',header='infer',index_col=False)

#Tb_dict_bs = Tb_dict_bs['Tb_max_pix_bs_b7', 'Tb_max_pix_bs_b6',  'V', 'R_arcsec']

#Tb_dict_rs = Tb_dict_rs['Tb_max_pix_rs_b7', 'Tb_max_pix_rs_b6', 'V', 'R_arcsec']

blue_shifted_pairs_outside_gap_outer_edge = np.array([ 
                                            [12.61,   9.12, -1.51, -3.138],      
                                            [10.26,   6.67, -1.72103403, -2.469],  
                                            ])

blue_shifted_pairs_outside_gap_inner_edge = np.array([   
                                            [11.46,  9.28, -2.767, 0.958],
                                            [11.63, 10.34, -2.554, -1.122], 
                                            [9.75, 8.43, -2.970,-0.824]   
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


plot_Tb_points = True

if plot_Tb_points:






#    ax.errorbar(inner_region_blue[:,0], inner_region_blue[:,1], xerr=1.01, yerr=0.43,
#       color='k', marker='x', ls='none')
#   for row_idx in range(inner_region_blue.shape[0]):
#       point_coord = inner_region_blue[row_idx,:]
#       ax.annotate(text = f"{point_coord[3]}", xy = (point_coord[0],point_coord[1]), xytext = (15,-15), textcoords='offset points',
#                           ha='center', va='bottom')
    ax.errorbar(red_shifted_pairs_outside_gap_outer_edge[:,0], red_shifted_pairs_outside_gap_outer_edge[:,1], xerr=1.01, yerr=0.43,
        color='red',  marker='s', ls='none', label = r'r > r$_{dep}$')
    for row_idx in range(red_shifted_pairs_outside_gap_outer_edge.shape[0]):
        point_coord = red_shifted_pairs_outside_gap_outer_edge[row_idx,:]
        ax.annotate(text = f"{int(point_coord[3]*140)} AU", xy = (point_coord[0],point_coord[1]), xytext = (15,-15), textcoords='offset points',
                            ha='center', va='bottom')
        
    ax.errorbar(red_shifted_pairs_outside_gap_inner_edge[:,0], red_shifted_pairs_outside_gap_inner_edge[:,1], xerr=1.01, yerr=0.43,
        color='red',  marker='o', ls='none', label = r'r < r$_{dep}$')
    for row_idx in range(red_shifted_pairs_outside_gap_inner_edge.shape[0]):
        point_coord = red_shifted_pairs_outside_gap_inner_edge[row_idx,:]
        ax.annotate(text = f"{int(point_coord[3]*140)} AU", xy = (point_coord[0],point_coord[1]), xytext = (17,-19), textcoords='offset points',
                            ha='center', va='bottom')
        
    ax.errorbar(red_shifted_pairs_in_gap[:,0], red_shifted_pairs_in_gap[:,1], xerr=1.01, yerr=0.43,
        color='red', marker='x', ls='none', label = r'r $\approx$ r$_{dep}$')
    for row_idx in range(red_shifted_pairs_in_gap.shape[0]):
        point_coord = red_shifted_pairs_in_gap[row_idx,:]
        ax.annotate(text = f"{int(point_coord[3]*140)} AU", xy = (point_coord[0],point_coord[1]), xytext = (15,-15), textcoords='offset points',
                            ha='center', va='bottom')



plt.legend()
plt.show()
# -----------------------