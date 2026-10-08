"""
Point-to-point ITM runner for RecordTV transmitter -> municipality paths.

Faster version of the original script:
  * computes each unique transmitter-receiver path only once
    (year, cod_mun and erp do not affect the propagation calculation),
  * runs paths in parallel across CPU cores,
  * writes the results once at the end instead of appending row by row,
  * no longer writes the debug shapefiles on every iteration.

Output columns are identical to the original p2p_results.csv.
"""
import os
import math

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from itmlogic.misc.qerfi import qerfi
from itmlogic.preparatory_subroutines.qlrpfl import qlrpfl
from itmlogic.statistics.avar import avar
from terrain_module import terrain_p2p

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
RAW_DATA_FOLDER = '/Users/marcelamello/Dropbox/Evangelical Churches in Brazil/ITMlogic'
RESULTS_FOLDER = os.path.join(RAW_DATA_FOLDER, 'results')
INPUT_FILE = os.path.join(RAW_DATA_FOLDER, 'top_10_distances_RecordTV.csv')
OUTPUT_FILE = os.path.join(RESULTS_FOLDER, 'p2p_results.csv')
DEM_FILE = '/Users/marcelamello/Documents/GitHub/itmlogic/results/output_combined.tif'

N_WORKERS = max(1, (os.cpu_count() or 2) - 1)   # leave one core free
RECEIVER_HEIGHT_M = 1.3

# ---------------------------------------------------------------------------
# Constants (computed once instead of once per row)
# ---------------------------------------------------------------------------
QC = [10, 50, 90]                          # confidence levels
QR = [10, 50, 90]                          # reliability levels
ZC = qerfi([x / 100 for x in QC])
ZR = qerfi([x / 100 for x in QR])
DB = 8.685890                              # conversion factor to dB
LOSS_COLS = [f'propagation_loss_dB_R{r}_C{c}' for r in QR for c in QC]

INPUT_COLS = ['latitude_transmitter', 'longitude_transmitter',
              'latitude_receiver', 'longitude_receiver',
              'fmhz', 'height_transmitter', 'year', 'cod_mun', 'erp']
PATH_KEYS = INPUT_COLS[:6]                 # what the ITM calculation depends on
OUTPUT_COLS = ['latitude_transmitter', 'longitude_transmitter',
               'latitude_receiver', 'longitude_receiver',
               'year', 'cod_mun', 'erp', 'distance_km', 'free_space'] + LOSS_COLS


def itmlogic_p2p(fmhz, height_tx, distance_km, profile):
    """Run ITM in point-to-point mode for one path. Same parameters as before."""
    pfl = [len(profile) - 1, 0, *profile]
    pfl[1] = distance_km * 1000 / pfl[0]   # range step in meters

    prop = {
        'fmhz': fmhz,
        'hg': [height_tx, RECEIVER_HEIGHT_M],
        'ipol': 1,          # vertical polarization
        'd': distance_km,
        'eps': 15,          # terrain relative permittivity
        'sgm': 0.001,       # terrain conductivity (S/m)
        'klim': 5,          # continental temperate climate
        'ens0': 301,        # surface refractivity
        'lvar': 5,
        'gma': 157e-9,      # inverse Earth radius
        'pfl': pfl,
        'kwx': 0,
        'wn': fmhz / 47.7,
        'klimx': 0,
        'mdvarx': 11,
    }
    prop['ens'] = prop['ens0']
    prop['gme'] = prop['gma'] * (1 - 0.04665 * math.exp(prop['ens'] / 179.3))

    zq = complex(prop['eps'], 376.62 * prop['sgm'] / prop['wn'])
    zgnd = np.sqrt(zq - 1)
    if prop['ipol'] != 0:
        zgnd = zgnd / zq
    prop['zgnd'] = zgnd

    prop = qlrpfl(prop)
    fs = DB * np.log(2 * prop['wn'] * prop['dist'])

    out = {'free_space': fs}
    # Same order of avar calls as the original (avar keeps state in prop)
    for zr, r in zip(ZR, QR):
        for zc, c in zip(ZC, QC):
            avar1, prop = avar(zr, 0, zc, prop)
            out[f'propagation_loss_dB_R{r}_C{c}'] = fs + avar1
    out['distance_km'] = prop['d']
    return out


def run_path(path):
    """Worker: terrain profile + ITM for one unique transmitter-receiver path."""
    lat_t, long_t, lat_r, long_r, fmhz, height_tx = path
    line = {
        'type': 'Feature',
        'geometry': {'type': 'LineString',
                     'coordinates': [(long_t, lat_t), (long_r, lat_r)]},
        'properties': {'id': 'terrain path'},
    }
    try:
        profile, distance_km, _ = terrain_p2p(DEM_FILE, line)
        result = itmlogic_p2p(fmhz, height_tx, distance_km, profile)
        result['error'] = None
    except Exception as e:                 # one bad path no longer kills the run
        result = {c: np.nan for c in ['distance_km', 'free_space'] + LOSS_COLS}
        result['error'] = repr(e)
    return result


if __name__ == '__main__':                 # required for multiprocessing on macOS
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    data = pd.read_csv(INPUT_FILE).iloc[:, :9].astype(float)
    data.columns = INPUT_COLS

    paths = data[PATH_KEYS].drop_duplicates().reset_index(drop=True)
    print(f'{len(data):,} rows -> {len(paths):,} unique paths, {N_WORKERS} workers')

    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, res in enumerate(ex.map(run_path,
                                       paths.itertuples(index=False, name=None),
                                       chunksize=50), start=1):
            results.append(res)
            if i % 1000 == 0 or i == len(paths):
                print(f'  {i:,}/{len(paths):,} paths done')

    paths = pd.concat([paths, pd.DataFrame(results)], axis=1)
    out = data.merge(paths, on=PATH_KEYS, how='left')

    n_err = out['error'].notna().sum()
    if n_err:
        out.loc[out['error'].notna(), PATH_KEYS + ['error']].to_csv(
            os.path.join(RESULTS_FOLDER, 'p2p_errors.csv'), index=False)
        print(f'WARNING: {n_err} rows failed; see p2p_errors.csv')

    out[OUTPUT_COLS].to_csv(OUTPUT_FILE, index=False)
    print(f'Wrote {len(out):,} rows to {OUTPUT_FILE}')
