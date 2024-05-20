import numpy as np
import emcee as em
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize
from scipy.optimize import Bounds
from typing import List

def log_likelihood(params:List[float], Y1:float, Y2:float, s1:float, s2:float, 
                   model, like_func:str = 'chisq')->float:

    """
    Computes log likelihood of brightness temperatures of C18O J = 3-2 and J = 2-1 line emission.

    Parameters:
    -----------
    params      : `Iterable`, List of two floats, parameter values (log10Ncol, Texe) 
    Y1          : `float`, Tb from J = 3-2 emission
    Y2          : `float`, Tb from J = 2-1 emission
    s1          : `float`, Sigma (measurment error) for Tb (J = 3-2)
    s2          : `float`, Sigma (measurment error) for Tb (J = 2-1)
    model       : `class: LTEanalysis` object. Model for LTE analysis
    like_funct  : `str`, default: 'chisq', Log Likelihood function for use, 
                    `chisq` - log10(Chi-square) 
                    `lg_cosh' - log10(cosh((y_pred - y_truth)/sigma_pred))

    """
    
    lg_N, T = params
    N = 10**lg_N     # We convert lg_N back to N

    # Y1 --> Tb(3-2), Y2 --> Tb(2-1)
    Y1_predicted = model.get_intensity(line = 'c18o', Ju = 3, Ncol = N, Tex = T, 
                                       delv = 7417.8, Xconv = 1.e-7)
    Y2_predicted = model.get_intensity(line = 'c18o', Ju = 2, Ncol = N, Tex = T, 
                                       delv = 7417.8, Xconv = 1.e-7)

    if like_func.lower() == "chisq":
        # Compute the log likelihood using normal distributions
        log_likelihood_Y1 = -0.5 * (np.log(2 * np.pi * s1**2) + (Y1 - Y1_predicted)**2 / s1**2)
        log_likelihood_Y2 = -0.5 * (np.log(2 * np.pi * s2**2) + (Y2 - Y2_predicted)**2 / s2**2)
    
    elif like_func.lower() == "lg_cosh":
        log_likelihood_Y1 = -0.5* np.log10(np.cosh((Y1 - Y1_predicted)/np.abs(s1))) 
        log_likelihood_Y2 = -0.5* np.log10(np.cosh((Y2 - Y2_predicted)/np.abs(s2)))
    
    else:
        raise ValueError("Please specify correct log likelihood function")
    
    lg_l = log_likelihood_Y1 + log_likelihood_Y2
    return lg_l

# Define the log prior function with variable bounds
def log_prior(params, bounds):
    # Uniform prior within the specified bounds for N and T
    lg_N, T = params
    lg_N_min, lg_N_max, T_min, T_max = bounds
    if lg_N_min < lg_N < lg_N_max and T_min < T < T_max:
        return 0.0
    return -np.inf

# Define the log posterior function
def log_posterior(params, Y1, Y2, s1, s2, bounds, model, like_func):
    log_prior_value = log_prior(params, bounds)
    if np.isinf(log_prior_value):
        return log_prior_value
    return log_prior_value + log_likelihood(params, Y1, Y2, s1, s2, model, like_func)



def estimate_params(t1:float, t2:float, s1:float, s2:float, estimator:str={'mcmc', 'scipy', 'both'},
                    initial_params:list = None, bounds:list = None, args:dict = None, initial_scatter:float = 0.1,
                    nwalkers:int = 100, n_steps:int = 1000, burn_in:int = 100, thin_by:int = 15, 
                    return_flat:bool = False, intensity_model = None, plot_chain:bool=True, 
                    r_v_info:list=[], chain_plot_path:str='', show_chains:bool = False) -> dict :
    """
    mcmc estimator

    Parameters:
    -----------

    t1          :   `float`, Tb of J = 3-2 line.
    t2          :   `float`, Tb of J = 2-1 line.
    s1          :   `float`, sigma for t1
    s2          :   `float`, sigma for t2
    estimator   :   `str`, Type of estimator 
    """
    if estimator == 'scipy':

        result = minimize(cost_function, initial_params, args=(df_blue["Tb_on_pix_b7"][i], df_blue["Tb_on_pix_b6"][i], lte_model),
                        method='Nelder-Mead', bounds = bounds, options={'xatol': 1e-12, 'maxiter' : 10000})
        # NOTE - Work in progress
        



    if estimator == 'mcmc':

        # Check if both temperature values are float
        if isinstance(t1, float) and isinstance(t2, float):
            
            ndim = len(initial_params)
            initial_scatter = 0.1

            p0 = np.array(initial_params, dtype=float) + initial_scatter * np.random.randn(nwalkers, ndim)

            args = (t1, t2, s1, s2, bounds, intensity_model)

            sampler = em.EnsembleSampler(nwalkers, ndim, log_posterior, args=args)
            
            sampler.run_mcmc(p0, n_steps, progress=True)

            # Extract samples

            samples = sampler.get_chain()

            if plot_chain:

                # Extract log likelihood values for each walker for all chains
                log_likelihood_values = np.zeros((nwalkers, n_steps))

                for i in range(nwalkers):
                    for j in range(n_steps):
                        params = samples[j, i]
                        log_likelihood_values[i, j] = log_likelihood(params, t1, t2, s1, s2, intensity_model)

                fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
                labels = ["lg_N", "T"]
                
                for i in range(2):
                    ax = axes[i]
                    for j in range(nwalkers):
                        ax.plot(samples[:, j, i], alpha=0.3)
                    ax.set_xlim(0, len(samples))
                    ax.set_ylabel(labels[i])

                for j in range(nwalkers):
                    axes[2].plot(log_likelihood_values[j])

                axes[2].set_ylabel('log_likelihood')

                axes[-1].set_xlabel("step number")

                fig.suptitle('chains_r_'+r_v_info[0]+'_v_'+r_v_info[-1])

                figname = 'chain_r_'+r_v_info[0]+'_v_'+r_v_info[-1]+'.jpg'
                
                if not os.path.isdir(chain_plot_path):
                    os.mkdir(chain_plot_path)
                figpath = os.path.join(chain_plot_path, figname)
                fig.savefig(fname = figpath, dpi=300, format='jpeg')

                if show_chains:
                    plt.show()
                else:
                    plt.close()
            # Now flatten chains for further analysis
            
            flattened_samples = sampler.get_chain(discard=burn_in, thin=thin_by, flat=return_flat)

            # autocorr_values = sampler.get_autocorr_time()

            return flattened_samples, 1
                        
        else:
            print("Case where t1 and t2 are iterable is not yet implemented. Work in progress.")
            return
    else:
        print("Other estimators are not yet implemented")
        return