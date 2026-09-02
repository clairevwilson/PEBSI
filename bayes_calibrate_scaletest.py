"""
Timing test at real calibration scale: default n_points/n_walkers/
temporal_chunk_years/date range from project/bayes_calibrate.py, but
only 2 MCMC steps. Not meant to produce a usable posterior -- just to
read per-call wall-clock ("Simulation completed in ___ seconds") off
the log at the scale the real run will actually use, before committing
to a multi-hundred-step job.

@author: clairevwilson
"""
from project.bayes_calibrate import calibrate_glacier

if __name__ == '__main__':
    chain = calibrate_glacier('gulkana', n_steps=2, n_burn=0)
    print('Scale test finished, chain shape:', chain.shape)
