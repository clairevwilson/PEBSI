"""
Pre-populates the multi-glacier scattered-site cache before submitting
batch_jobs/grid_calibrate.sh's array -- all 10 jobs pool the same
5-glacier point set, and would otherwise race to compute/write the
same cache file on first use. Run this once (no GPU needed) and let it
finish before submitting the array job.

@author: clairevwilson
"""
from project.bayes_calibrate import GLACIERS, translate_rgi
from project.scatter_sites import get_multiglacier_scattered_sites

N_POINTS = 3000

if __name__ == '__main__':
    rgi_ids = [translate_rgi[g]['6'] for g in GLACIERS]
    site_names, site_rgi_ids, site_glacier_names, metadata_fn = \
        get_multiglacier_scattered_sites(GLACIERS, rgi_ids, N_POINTS)

    print(f'{len(site_names)} points cached at {metadata_fn}')
    for glacier_name in GLACIERS:
        n_g = sum(1 for g in site_glacier_names if g == glacier_name)
        print(f'  {glacier_name}: {n_g} points')
