from niseq.power.bootstrap import bootstrap_predictive_power_1samp
from niseq.spending_functions import PocockSpendingFunction
from niseq import sequential_cluster_test_1samp

import numpy as np
import json
import mne
import os

################################################################################
## Simulation parameters
################################################################################

# input and output directories relative to working directory
ERPCORE_DIR = 'erpcore'
RESULTS_DIR = 'results'

# ERP effect for this simulation
EFFECT_NAME = 'MMN'

# simulation params
N_SIMULATIONS = 10000
N_FIXED = 30
N_LOOKS = 3
ALPHA = .01
CLUSTER_THRESHOLD = 2.

################################################################################
## Main
################################################################################

# load data and format for clustering
erp_fpath = os.path.join(ERPCORE_DIR, '%s_ave.fif.gz'%EFFECT_NAME)
evokeds = mne.read_evokeds(erp_fpath, verbose = False)
X = np.stack([evo.get_data() for evo in evokeds], axis = 0)
X = np.transpose(X, (0, 2, 1)) # observations x time x channels

# construct channel adjacency matrix
adj, _ = mne.channels.find_ch_adjacency(evokeds[0].info, "eeg")

# fixed-sample power analysis
res_fixed = bootstrap_predictive_power_1samp(
    X, sequential_cluster_test_1samp,
    [N_FIXED], N_FIXED,
    alpha = ALPHA,
    adjacency = adj,
    threshold = CLUSTER_THRESHOLD,
    n_simulations = N_SIMULATIONS,
    seed = 0, n_jobs = -1
)
power_fixed = res_fixed['cumulative_power'][-1]

def get_inflation_factor(beta, n_looks = 3, spending_type = 'asP'):
    '''
    Computes inflation factor for a sequential design with evenly spaced looks.

    Arguments
    ---------
    beta : float
        The Type II error of the fixed sample design
        that's power you're trying to match.
    n_looks : int
        The number of (evenly spaced) looks at the data.
    spending_type : 'asP' | 'asOF', default: 'asP'
        'asP' uses a Pocock spending function (default)
        'asOF' uses an OBrien Fleming spending function

    Returns
    --------
    inflation : float
        The factor by which one must multiply the sample size of a fixed
        sample design with Type II error ``beta`` to yield the maximum sample
        size required for a sequential test of the specified design to
        acheive the same Type II error.

    Notes
    -------
    Requires R to be installed in your environment.

    '''
    from rpy2 import robjects
    fv = robjects.r(
        '''
        if (!('rpact' %%in%% installed.packages())) {
            install.packages(
                'rpact', type = 'source',
                repos = 'http://cran.us.r-project.org'
            )
        }
        design <- rpact::getDesignGroupSequential(
            kMax = %d, beta = %f, typeOfDesign = '%s'
        )
        chars <- rpact::getDesignCharacteristics(design)
        chars$inflationFactor
        '''%(n_looks, beta, spending_type)
    )
    return fv[0]

# use inflation factor to predict max n needed to match fixed n's power
inflation = get_inflation_factor(1 - power_fixed, N_LOOKS)
n_max = int(np.floor(N_FIXED * inflation) + 1) # round up
look_times = [i*n_max//N_LOOKS for i in range(1, N_LOOKS + 1)]

# sequential design power analysis
results = bootstrap_predictive_power_1samp(
    X, sequential_cluster_test_1samp,
    look_times, n_max,
    alpha = ALPHA,
    adjacency = adj,
    threshold = CLUSTER_THRESHOLD,
    spending_func = PocockSpendingFunction(ALPHA, n_max),
    n_simulations = N_SIMULATIONS,
    seed = 0, n_jobs = -1
)

# store simulation results
results['n_fixed'] = N_FIXED
results['power_fixed'] = power_fixed
results['inflation_factor'] = inflation
results['test_func_kwargs']['adjacency'] = type(results['test_func_kwargs']['adjacency']).__name__
results_fname = '%s.json'%EFFECT_NAME
results_fpath = os.path.join(RESULTS_DIR, results_fname)
with open(results_fpath, "w") as f:
    json.dump(results, f, indent = 4)
