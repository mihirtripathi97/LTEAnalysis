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
Ncols = np.array([5.e16, 5.e17, 5.e18, 5.e19]) # cm^-2  
Texes = np.array([5, 30]) # K



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



def cost_function(params, X, Y, model):

    # print("printing params", params)
    N, T = params[0], params[1]
    X_predicted = model.get_intensity(line = line, Ju = ilines[0], Ncol = N, Tex = T, delv = 0.5, Xconv = Xconv),
    Y_predicted = model.get_intensity(line = line, Ju = ilines[1], Ncol = N, Tex = T, delv = 0.5, Xconv = Xconv)
    error = np.sum((X_predicted - X)**2 + (Y_predicted - Y)**2)
    return error

def gradient(params, X, Y, model, h=[1e16, 1e-5] ):
    grad = np.zeros_like(params, dtype=float)  # Ensure the gradient array has the same data type as params
    for i in range(len(params)):
        params_plus_h = params.copy()
        params_plus_h[i] = params_plus_h[i] + h[i]
        cost_plus_h = cost_function(params_plus_h, X, Y, model)
        params_minus_h = params.copy()
        params_minus_h[i] = params_minus_h[i] - h[i]
        cost_minus_h = cost_function(params_minus_h, X, Y, model)
        grad[i] = (cost_plus_h - cost_minus_h) / (2 * h[i])
    return grad

def gradient_descent(X, Y, initial_params, learning_rate, tolerance, max_iter, model):
    params = initial_params
    for i in range(max_iter):
        grad = gradient(params, X, Y, model)
        params -= learning_rate * grad
        if np.linalg.norm(grad) < tolerance:
            print(f"Converged after {i+1} iterations")
            break
    return params



lte_model = LTEAnalysis()
lte_model.read_lamda_moldata(line)

# Let's read Tb values of all points from blue shifted points outside gap
df_bs_os_gap = pd.DataFrame(blue_shifted_pairs_outside_gap_inner_edge, columns = ["Tb_b7", "Tb_b6", "V", "R_as"])

# Call gradient decent to estimate N and T for each pair of Tbs in the dataframe
N_c = []
T_e = []

for i in range(len(df_bs_os_gap)):

    print(f"Finding best fit for {i}th pair")
    parameters = gradient_descent(X = df_bs_os_gap["Tb_b7"][i], Y = df_bs_os_gap["Tb_b6"][i],
                                    initial_params = np.array([1e17,30]), learning_rate=0.01, tolerance=1e-2, 
                                    max_iter=100000, model = lte_model)

    N_c.append(parameters[0])
    T_e.append(parameters[-1])


Texes = np.append(Texes, T_e)
Ncols = np.append(Ncols, N_c)


Tb_7_pred = lte_model.get_intensity(line = line, Ju = ilines[0], Ncol = N_c[0]*1.1, Tex = T_e[0], delv = 0.5, Xconv=Xconv)
Tb_6_pred = lte_model.get_intensity(line = line, Ju = ilines[1], Ncol = N_c[1]*1.1, Tex = T_e[0], delv = 0.5, Xconv=Xconv)

print(Ncols)
fig, ax = lte_model.makegrid(line, ilines[0], ilines[1], Texes, Ncols, delv, Xconv=Xconv, lw=1.)
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

    ax.scatter(Tb_7_pred, Tb_6_pred, color = 'green')


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