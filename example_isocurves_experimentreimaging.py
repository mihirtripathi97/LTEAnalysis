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
Ncols = np.array([5.e15, 5e16, 5.e17, 5.e18]) # cm^-2   # Note the distinction in column density, we have gotten gown a whole order of magnitude
Texes = np.array([5., 10., 14., 15, 16.5, 18,  22., 30., 40.]) # K



# Following arrays are in this form [[Tb_band7, Tb_band_6, Rotational_velocity,offset]]

blue_shifted_pairs_outside_gap_outer_edge = np.array([ 
                                            [10.42,   7.35, -1.51, -3.138],      
                                            # [10.62,   6.67, -1.72103403, -2.469],  
                                            ])

blue_shifted_pairs_outside_gap_inner_edge = np.array([   
                                            # [10.70,  6.88, -2.34609525, -1.333],
                                            [8.82, 10.17, -2.554, -1.122]    

                                            ])

blue_shifted_pairs_in_gap = np.array([ 
                                      [6.54,  5.20, -2.13774151, -1.602],
                                      [6.51,  6.12, -1.92938777, -1.959], 
                                    ])

red_shifted_pairs_outside_gap_inner_edge = np.array([ 
                                            # [8.64, 7.38, 2.858, 0.9],
                                            [9.09, 9.07,  3.071, 0.775]])

red_shifted_pairs_outside_gap_outer_edge = np.array([
                                                [9.06, 9.79,  2.023, 1.775]
                                                ])

red_shifted_pairs_in_gap = np.array([
                                     # [6.71466255, 7.6401782,  2.23768703, 1.4],
                                     # [5.84474993, 4.46442366, 2.44604077, 1.212],
                                     [5.72, 4.63, 2.65439451, 1.04]
                                     ])


inner_region_red = np.array([
                            [8.70,8.23,3.27,0.685, 16.5],
                            [10.07,7.57,3488,0.606, 20.],
                            [10.28,7.13,3.699,0.539,21.],
                            [8.25,6.13,3.98,0.478,18.0 ],
                            [7.24,6.94,4.1,0.435,15.0],
                            [6.01,4.43,4.3,0.393,16.5],
                            [4.01,4.93,4.73,0.326,10.]      
                        ])
outer_region_red = np.array([
                            [12.42,10.18,1.81, 2.208,21.5],
                            [8.34,8.50,1.6,2.82, 15.0],
                            [4.65,6.62,1.4,3.7, 10.0],
                            [1.42,4.58,1.20,5.11,5.0],                          
                            ])

outer_region_blue = np.array([
                            [5.64,2.44, -1.3,-4.30,30.0]
                            ])

inner_region_blue = np.array([
                    [10.60,8.58,-2.77,-0.958, 20.],
                    [9.30,7.84,-2.97,-0.823, 18.],
                    [9.55,7.38,-3.17,-0.724, 20.],
                    [9.21,6.43,-3.39,-0.632, 21.0],
                    [7.65,5.35,-3.6,-0.576, 18.],
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

    ax.errorbar(blue_shifted_pairs_outside_gap_outer_edge[:,0], blue_shifted_pairs_outside_gap_outer_edge[:,1], xerr=1.01, yerr=0.43,
        color='blue', marker='s', ls='none', label = 'points outside depletion region')
    for row_idx in range(blue_shifted_pairs_outside_gap_outer_edge.shape[0]):
        point_coord = blue_shifted_pairs_outside_gap_outer_edge[row_idx,:]
        ax.annotate(text = f"{int(point_coord[3]*140)}AU", xy = (point_coord[0],point_coord[1]), xytext = (25,-25), textcoords='offset points',
                            ha='center', va='bottom')

    ax.errorbar(blue_shifted_pairs_outside_gap_inner_edge[:,0], blue_shifted_pairs_outside_gap_inner_edge[:,1], xerr=1.01, yerr=0.43,
        color='blue', marker='o', ls='none')
    for row_idx in range(blue_shifted_pairs_outside_gap_inner_edge.shape[0]):
        point_coord = blue_shifted_pairs_outside_gap_inner_edge[row_idx,:]
        ax.annotate(text = f"{int(point_coord[3]*140)}AU", xy = (point_coord[0],point_coord[1]), xytext = (35,6), textcoords='offset points',
                            ha='center', va='bottom')
        
    ax.errorbar(blue_shifted_pairs_in_gap[:,0], blue_shifted_pairs_in_gap[:,1], xerr=1.01, yerr=0.43,
        color='blue', marker='x', ls='none')
    for row_idx in range(blue_shifted_pairs_in_gap.shape[0]):
        point_coord = blue_shifted_pairs_in_gap[row_idx,:]
        ax.annotate(text = f"{int(point_coord[3]*140)}AU", xy = (point_coord[0],point_coord[1]), xytext = (28,6), textcoords='offset points',
                            ha='center', va='bottom')


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


#    ax.errorbar(inner_region_blue[:,0], inner_region_blue[:,1], xerr=1.01, yerr=0.43,
#       color='k', marker='x', ls='none')
#   for row_idx in range(inner_region_blue.shape[0]):
#       point_coord = inner_region_blue[row_idx,:]
#       ax.annotate(text = f"{point_coord[3]}", xy = (point_coord[0],point_coord[1]), xytext = (15,-15), textcoords='offset points',
#                           ha='center', va='bottom')

    # plt.legend()
plt.show()
# -----------------------