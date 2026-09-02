"""
Small smoke test for project/bayes_calibrate.py: a handful of
walkers, a handful of points, a couple of MCMC steps, one glacier.
Just checks the config-build -> PEBSI-run -> loss-slicing ->
emcee-step round trip works end to end, not real calibration.

@author: clairevwilson
"""
from project.bayes_calibrate import calibrate_glacier

if __name__ == '__main__':
    chain = calibrate_glacier(
        'gulkana', n_points=50, n_walkers=4, n_steps=2, n_burn=0,
        start_date='2019-04-01 00:00', end_date='2020-04-19 23:00')
    print('Smoke test finished, chain shape:', chain.shape)
    print(chain)
