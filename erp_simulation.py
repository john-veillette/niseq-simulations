from niseq.power.bootstrap import bootstrap_predictive_power_1samp
from niseq.spending_functions import PocockSpendingFunction
from niseq import sequential_permutation_t_test_1samp
from niseq import sequential_cluster_test_1samp

import numpy as np
import json
import mne
import os
import argparse
from mne.utils import check_random_state

def _compute_covariance(evoked):
    _raw = mne.io.RawArray(evoked.get_data(), evoked.info)
    return mne.compute_raw_covariance(_raw, verbose = False)

def _get_lowpass_coefs(evoked):
    h_freq = evoked.info['lowpass']
    iir_params = mne.filter.create_filter(
        None, evoked.info['sfreq'],
        None, h_freq,
        method = 'iir',
        iir_params = dict(output = 'ba'),
        verbose = False
    )
    return iir_params['a']

def get_noise_simulator(evoked, sigma = 1.):
    '''
    Returns a function that generates spatially colored MV white noise,
    IIR filtered to produce temporal autocorrelation.
    '''
    from mne.simulation.evoked import _generate_noise
    ev = evoked.copy() # makes a version w/ ave reference projection
    ev.set_eeg_reference('average', projection = True)
    # computes covariance mtx to model spatial autocorreltion
    cov = _compute_covariance(ev)
    # gathers necessary metadata
    info = ev.info
    iir_filter = _get_lowpass_coefs(ev)
    n_times = ev.times.size
    def f(seed):
        noise, _ = _generate_noise(
            info, cov, iir_filter,
            seed, n_times
        )
        return sigma * noise.T
    return f


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


def run_simulations(
    EFFECT_NAME = 'MMN',
    TEST_TYPE = 'cluster',
    N_SIMULATIONS = 1000,
    N_FIXED = 30,
    N_LOOKS = 2,
    ALPHA = .05,
    NOISE = 0., # add noise if nonzero
    N_JOBS = -1,
    CLUSTER_THRESHOLD = 2., # only used if TEST_TYPE == 'cluster'
    ERPCORE_DIR = 'erpcore',
    RESULTS_DIR = 'results'):
    '''
    Runs a fixed-sample power analysis by simulation, bootstrapping from
    specified effect in ERPCORE dataset. Then, computes an inflation factor
    to design a sequential test with (hopefully) matched power, and finally
    runs a power analysis for that sequential design.
    '''

    assert(TEST_TYPE in ['tmax', 'cluster'])

    # load data and format for clustering
    erp_fpath = os.path.join(ERPCORE_DIR, '%s_ave.fif.gz'%EFFECT_NAME)
    evokeds = mne.read_evokeds(erp_fpath, verbose = False)
    X = np.stack([evo.get_data() for evo in evokeds], axis = 0)
    X = np.transpose(X, (0, 2, 1)) # observations x time x channels

    test_args = dict(
        alpha = ALPHA,
        n_simulations = N_SIMULATIONS,
        seed = 0, n_jobs = N_JOBS
    )
    if TEST_TYPE == 'tmax':
        test_args['test_func'] = sequential_permutation_t_test_1samp
    elif TEST_TYPE == 'cluster':
        adj, _ = mne.channels.find_ch_adjacency(evokeds[0].info, "eeg")
        test_args['test_func'] = sequential_cluster_test_1samp
        test_args['adjacency'] = adj
        test_args['threshold'] = CLUSTER_THRESHOLD

    if NOISE != 0.:
        evoked = mne.grand_average(evokeds)
        noise_func = get_noise_simulator(evoked, sigma = NOISE)
        test_func = test_args['test_func']
        def noisy_test_func(X, **kwargs):
            assert('seed' in kwargs)
            n = X.shape[0]
            seed = kwargs['seed']
            rng = check_random_state(seed)
            seeds = rng.randint(1, np.iinfo(np.int32).max - 1, n)
            noise = np.stack([noise_func(s) for s in seeds])
            return test_func(X + noise, **kwargs)
        test_args['test_func'] = noisy_test_func

    # fixed-sample power analysis
    res_fixed = bootstrap_predictive_power_1samp(
        X,
        look_times = [N_FIXED],
        n_max = N_FIXED,
        **test_args
    )
    power_fixed = res_fixed['cumulative_power'][-1]
    print('Fixed sample power: %f'%power_fixed)

    # use inflation factor to predict max n needed to match fixed n's power
    inflation = get_inflation_factor(1 - power_fixed, N_LOOKS)
    n_max = int(np.floor(N_FIXED * inflation) + 1) # round up
    look_times = [i*n_max//N_LOOKS for i in range(1, N_LOOKS + 1)]

    # sequential design power analysis
    results = bootstrap_predictive_power_1samp(
        X,
        look_times = look_times,
        n_max = n_max,
        spending_func = PocockSpendingFunction(ALPHA, n_max),
        **test_args
    )

    # store simulation results
    results['n_fixed'] = N_FIXED
    results['power_fixed'] = power_fixed
    results['inflation_factor'] = inflation
    results['noise_param'] = NOISE
    if TEST_TYPE == 'cluster':
        results['test_func_kwargs']['adjacency'] = type(results['test_func_kwargs']['adjacency']).__name__
    results_fname = '%s_%s.json'%(EFFECT_NAME, TEST_TYPE)
    results_fpath = os.path.join(RESULTS_DIR, results_fname)
    with open(results_fpath, "w") as f:
        json.dump(results, f, indent = 4)



if __name__ == "__main__":
    run_simulations()
