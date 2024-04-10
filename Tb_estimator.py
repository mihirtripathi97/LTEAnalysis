import numpy as np
import emcee as em


def log_likelihood(params, Y1, Y2, s1, s2, model):
    
    lg_N, T = params
    N = 10**lg_N     # We convert lg_N back to N

    # Y1 --> Tb(3-2), Y2 --> Tb(2-1)
    Y1_predicted = model.get_intensity(line = 'c18o', Ju = 3, Ncol = N, Tex = T, delv = 0.5, Xconv = 1.e-7)
    Y2_predicted = model.get_intensity(line = 'c18o', Ju = 2, Ncol = N, Tex = T, delv = 0.5, Xconv = 1.e-7)



    # Compute the log likelihood using normal distributions
    log_likelihood_Y1 = -0.5 * (np.log(2 * np.pi * s1**2) + (Y1 - Y1_predicted)**2 / s1**2)
    log_likelihood_Y2 = -0.5 * (np.log(2 * np.pi * s2**2) + (Y2 - Y2_predicted)**2 / s2**2)
    
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
def log_posterior(params, Y1, Y2, s1, s2, bounds, model):
    log_prior_value = log_prior(params, bounds)
    if np.isinf(log_prior_value):
        return log_prior_value
    return log_prior_value + log_likelihood(params, Y1, Y2, s1, s2, model)



def estimate_params(t1:float, t2:float, s1:float, s2:float, estimator:str={'mcmc', 'scipy'},
                    initial_params:list = None, bounds:list = None, args:dict = None, initial_scatter:float = 0.1,
                    nwalkers:int = 100, n_steps:int = 1000, burn_in:int = 100, thin_by:int = 15, 
                    return_flat:bool = False, intensity_model = None) -> dict :
    """
    mcmc estimator
    For now let's make sure that N is supplied as log_10(N)

    Parameters:
    -----------

    t1          :   `float`, Tb of J = 3-2 line.
    t2          :   `float`, Tb of J = 2-1 line.
    s1          :   `float`, sigma for t1
    s2          :   `float`, sigma for t2
    estimator   :   `str`, Type of estimator 
    """
    if estimator == 'mcmc':
        if isinstance(t1, float) and isinstance(t2, float):
            
            ndim = len(initial_params)
            initial_scatter = 0.1

            p0 = np.array(initial_params, dtype=float) + initial_scatter * np.random.randn(nwalkers, ndim)

            args = (t1, t2, s1, s2, bounds, intensity_model)

            sampler = em.EnsembleSampler(nwalkers, ndim, log_posterior, args=args)
            
            sampler.run_mcmc(p0, n_steps, progress=True)

            # Extract samples
            # samples = sampler.get_chain()

            # Flatten samples for further analysis
            flattened_samples = sampler.get_chain(discard=burn_in, thin=thin_by, flat=return_flat)

            return flattened_samples
                
            
        else:
            print("Case where t1 and t2 are iterable is not yet implemented. Work in progress.")
            return
    else:
        print("Other estimators are not yet implemented")
        return