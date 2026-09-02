"""
Runs Bayesian calibration of (kp, wind_factor) for each of the 5
named glaciers via project/bayes_calibrate.py, batching every MCMC
step's full walker ensemble into one multi-point PEBSI config run.

@author: clairevwilson
"""
from project.bayes_calibrate import calibrate_glacier, GLACIERS, PARAM_NAMES

if __name__ == '__main__':
    for glacier_name in GLACIERS:
        chain = calibrate_glacier(glacier_name)
        means = chain.mean(axis=0)
        stds = chain.std(axis=0)
        for name, m, s in zip(PARAM_NAMES, means, stds):
            print(f'{glacier_name}: {name} = {m:.3f} +/- {s:.3f}')
