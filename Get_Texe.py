from pv_analyzer import pv_analyze
import numpy as np
import Tb_estimator as tb_est
from lteanalysis import LTEAnalysis
import corner
import matplotlib.pyplot as plt
import os


def emp_kep(v_rot, v_100):        
        r_au = 100.*np.power(v_100/v_rot, 2)
        return r_au

# Read b7 data
pv_b7 = pv_analyze(pv_path='uid___A002_Xb5ee5a_X288a.ms.split.cal.l1489irs.spw3.cvel_chavg2.cube.clean_2_6_8_05.image.pbcor.regridded.smoothened.PV_69_w1.fits',
                    is_Tb=True)
pv_b7.read_pv(print_details = True)


# Read band 6 data
pv_b6 = pv_analyze(pv_path='uid___A002_b_6.cal.l1489_irs.spw_1_7.line.cube.clean.c_baseline_0.image.pbcor.Regridded.Smoothened.PV_69_w1.fits',
                    is_Tb=True)
pv_b6.read_pv(print_details = True)


Tb_df_b7 = pv_b7.get_tb_on_curve(curve_function = emp_kep, get_surrounding_pix = False,
                              num_pix = 3, cf_kwargs = {'v_100':3.2})
Tb_df_b6 = pv_b6.get_tb_on_curve(curve_function = emp_kep, get_surrounding_pix = False,
                              num_pix = 3, cf_kwargs = {'v_100':3.2})


# Initiate LTE model
lte_model = LTEAnalysis()
lte_model.read_lamda_moldata('c18o')


Texe_rs_empkep = []
Texe_rs_sgm_empkep = []

Nexe_rs_empkep = []
Nexe_rs_sgm_empkep = []

list_samples = []

plot_corner = True

plot_uif = True



np.random.seed(42)

lg_n_bounds = [10.,20.]
T_bounds = [5.,100.]

lg_n_init = np.random.uniform(lg_n_bounds[0], lg_n_bounds[-1])
T_init = np.random.uniform(T_bounds[0], T_bounds[-1])

for Tb7, Tb6, r, v in zip(Tb_df_b7["Tb_on_point_rs"][1:4], Tb_df_b6["Tb_on_point_rs"][1:4], 
                          pv_b6.r_as_rs[1:4], pv_b6.v_rot_redshifted[1:4]):

    print(f"estimating for r = {r:.3e} arcsec, v = {v: .2f} Kmps , Tb7 = {Tb7:.1f}, Tb6 = {Tb6:.1f}")

    # + 1.08**2   + 0.49**2

    flat_samples, autocorr = tb_est.estimate_params(t1 = Tb7, t2=Tb6, s1=np.sqrt((0.1*Tb7)**2 + 1.08**2), s2=np.sqrt((0.1*Tb6)**2+ 0.49**2), 
                                     estimator='mcmc', initial_params = [lg_n_init, T_init], 
                                     bounds=(lg_n_bounds[0], lg_n_bounds[-1], T_bounds[0], T_bounds[-1]), 
                                     initial_scatter = 10, args= None,
                                     nwalkers = 20, n_steps = 10000, burn_in = 1000, thin_by = 20, return_flat= True,
                                     intensity_model = lte_model, plot_chain = True, 
                                     r_v_info = [str(round(r,3)), str(round(v,2))], 
                                     chain_plot_path = os.path.join(os.path.abspath(os.getcwd()),"chains","redshifted_points"),
                                     show_chains = False)
    
    flat_samples_N = 10**(flat_samples[:, 0])
    flat_samples_T = flat_samples[:, 1]
    
    Texe_rs_empkep.append(np.median(flat_samples_T))
    Texe_rs_sgm_empkep.append(np.std(flat_samples_T))
    Nexe_rs_empkep.append(np.median(flat_samples_N))
    Nexe_rs_sgm_empkep.append(np.std(flat_samples_N))


    if plot_corner:


        fig = corner.corner(flat_samples, labels= ['lg_N', 'T (K)'], truths=[np.log10(Nexe_rs_empkep[-1]), Texe_rs_empkep[-1]], 
                            truth_color = 'green', quantiles=[0.16,0.5,0.84], show_titles=True,  #range=(lg_n_bounds, [1.5,65.])
                            )
        print("hi")



        fig.suptitle('corner_r_'+str(round(r,2))+'_v_'+str(round(v,2)), fontsize=16)
        fig.subplots_adjust(top=0.86)
        figname = 'corner_r_'+str(round(r,2))+'_v_'+str(round(v,2))+'.jpg'
        figpath = os.path.join(os.path.abspath(os.getcwd()),"corner_plots","redshifted_points", figname)
        print(figpath)
        plt.show()
        fig.savefig(fname = figpath, dpi=300, format='jpeg')
        plt.close()
