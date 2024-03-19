import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 18
from scipy.optimize import minimize
from lteanalysis import LTEAnalysis
import pandas as pd



# ----- input -----
# for LTE calculation
line  = 'c18o'
Xconv = 1e-7
delv  = 0.2 # km/s
ilines = [3,2] # Ju
Ncols = np.array([5.e16, 5.9e16, 5.e17]) # cm^-2  
Texes = np.array([5, 18, 22, 30]) # K



blue_shifted_pairs_outside_gap_outer_edge = np.array([ 
                                            [10.4,   7.71, -1.51, -3.138],      
                                            [12.54,   9.05, -1.72103403, -2.469],  
                                            ])

blue_shifted_pairs_outside_gap_inner_edge = np.array([   
                                            [11.62, 9.8, -2.554, -1.122], 
                                            [11.60,  10.31, -2.767, -0.958],           
                                            [12.49, 10.25, -2.970,-0.824]   
                                            ])

blue_shifted_pairs_in_gap = np.array([ 
                                      [11.0,  8.1, -2.13774151, -1.602],
                                      [9.74,  8.8, -1.92938777, -1.959], 
                                    ])

plot_Tb_points = True
plot_normal_gd = True
plot_sp_gd = True

def cost_function(params, X, Y, model):

    # print("printing params", params)
    N, T = params[0]*1.e19, params[1]*50
    X_predicted = model.get_intensity(line = line, Ju = ilines[0], Ncol = N, Tex = T, delv = 0.5, Xconv = Xconv),
    Y_predicted = model.get_intensity(line = line, Ju = ilines[1], Ncol = N, Tex = T, delv = 0.5, Xconv = Xconv)
    error = (X_predicted - X)**2 + (Y_predicted - Y)**2

    # print(error)
    return error

def gradient(params, X, Y, model, h=0.01):
    grad = np.zeros_like(params, dtype=float)  # Ensure the gradient array has the same data type as params
    for i in range(len(params)):
        params_plus_h = params.copy()
        params_plus_h[i] = params_plus_h[i] + h
        cost_plus_h = cost_function(params_plus_h, X, Y, model)
        params_minus_h = params.copy()
        params_minus_h[i] = params_minus_h[i] - h
        cost_minus_h = cost_function(params_minus_h, X, Y, model)
        grad[i] = (cost_plus_h - cost_minus_h) / (2 * h)
    return grad

def gradient_descent(X, Y, initial_params, learning_rate, tolerance, max_iter, model):
    params = initial_params
    for i in range(max_iter):
        grad = gradient(params, X, Y, model)
        params -= np.dot(learning_rate, grad)
        if np.linalg.norm(grad) < tolerance:
            print(f"Converged after {i+1} iterations")
            break
    return params





lte_model = LTEAnalysis()
lte_model.read_lamda_moldata(line)

# Let's read Tb values of all points from blue shifted points outside gap
df_bs_os_gap = pd.DataFrame(blue_shifted_pairs_outside_gap_inner_edge, columns = ["Tb_b7", "Tb_b6", "V", "R_as"])

# Call gradient decent to estimate N and T for each pair of Tbs in the dataframe
N_c_normal_gd = []
T_e_normal_gd = []

print("Using normal grad decent")

for i in range(len(df_bs_os_gap)):

    print(f"Finding best fit for {i}th pair")
    parameters = gradient_descent(X = df_bs_os_gap["Tb_b7"][i], Y = df_bs_os_gap["Tb_b6"][i],
                                    initial_params = np.array([1.e16/1.e19,30./50.]), learning_rate = 0.000001, tolerance = 1e-8, 
                                    max_iter = 100000, model = lte_model)

    N_c_normal_gd.append(parameters[0]*1.e19)
    T_e_normal_gd.append(parameters[-1]*50.)

print(N_c_normal_gd)
print(T_e_normal_gd)

# Let's do another method

print("Using scipy grad decent")
initial_params = np.array([1.e15/1.e19, 30./50.])

N_c_pred_gd_sp = []
T_e_pred_gd_sp = []

for i in range(len(df_bs_os_gap)):
    
    result = minimize(cost_function, initial_params, args=(df_bs_os_gap["Tb_b7"][i], df_bs_os_gap["Tb_b6"][i], lte_model), method='Nelder-Mead', tol=1e-8)

    print('Optimization successful: ', result.success)
    print("Cause - ", result.message)
    N_c_pred_gd_sp.append(result.x[0]*1.e19)
    T_e_pred_gd_sp.append(result.x[1]*50.)


print(N_c_pred_gd_sp)
print(T_e_pred_gd_sp)



fig, ax = lte_model.makegrid(line, ilines[0], ilines[1], Texes, Ncols, delv, Xconv=Xconv, lw=1.)
ax.set_xlim(0., 25)
ax.set_ylim(0., 25)


if plot_normal_gd:

    Tb_7_pred_gd_normal = []
    Tb_6_pred_gd_normal = []

    for i in range(len(N_c_normal_gd)):

        Tb_7_pred_gd_normal.append(lte_model.get_intensity(line = line, Ju = ilines[0], Ncol = N_c_normal_gd[i], Tex = T_e_normal_gd[i], delv = 0.5, Xconv=Xconv))
        Tb_6_pred_gd_normal.append(lte_model.get_intensity(line = line, Ju = ilines[1], Ncol = N_c_normal_gd[i], Tex = T_e_normal_gd[i], delv = 0.5, Xconv=Xconv))

    ax.scatter(Tb_7_pred_gd_normal, Tb_6_pred_gd_normal, color = 'green', marker = '^')

if plot_sp_gd:

    Tb_7_pred_gd_sp = []
    Tb_6_pred_gd_sp = []

    for i in range(len(N_c_pred_gd_sp)):

        Tb_7_pred_gd_sp.append(lte_model.get_intensity(line = line, Ju = ilines[0], Ncol = N_c_pred_gd_sp[i], Tex = T_e_pred_gd_sp[i], delv = 0.5, Xconv=Xconv))
        Tb_6_pred_gd_sp.append(lte_model.get_intensity(line = line, Ju = ilines[1], Ncol = N_c_pred_gd_sp[i], Tex = T_e_pred_gd_sp[i], delv = 0.5, Xconv=Xconv))

    ax.scatter(Tb_7_pred_gd_sp, Tb_6_pred_gd_sp, marker = '^', facecolors='none', edgecolors='k')

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