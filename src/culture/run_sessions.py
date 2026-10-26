"""Run the fixed metric pipeline on every session of DANDI 002020 or Wagenaar 2006.

usage: python -I src/culture/run_sessions.py dandi|wagenaar <data_dir> <out_dir> [nproc]
"""
from __future__ import annotations

import glob
import os
import pickle
import sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import PARAMS, analyze  # noqa: E402

GRID_DANDI = dict(x0=0.0, y0=0.0, w=100.0, nx=39, ny=21)
GRID_WAG = dict(x0=-100.0, y0=-100.0, w=200.0, nx=8, ny=8)
MIN_RATE_PER_CH = 30000 / 1000 / 1800  # = 'nearly empty' rule (<30k spikes for ~1000 ch x 30 min)

# Wagenaar/meabench hw2cr.m (Potter lab help page): hardware channel -> column, row (0-based)
_COLS = [3, 3, 3, 3, 2, 2, 1, 2, 1, 0, 1, 0, 2, 1, 0, 0, 1, 2, 0, 1, 0, 1, 2, 1, 2, 2, 3, 3, 3, 3,
         4, 4, 4, 4, 5, 5, 6, 5, 6, 7, 6, 7, 5, 6, 7, 7, 6, 5, 7, 6, 7, 6, 5, 6, 5, 5, 4, 4, 4, 4]
_ROWS = [6, 7, 5, 4, 7, 6, 7, 5, 6, 6, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 1, 1, 2, 0, 1, 0, 3, 2, 0, 1,
         1, 0, 2, 3, 0, 1, 0, 2, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 5, 7, 6, 7, 4, 5, 7, 6]

VARIANTS = {
    'base': {},
    'thr_x0.5': dict(thr_mult=2.5, floor_frac=0.05, floor_min=2.5),
    'thr_x2': dict(thr_mult=10.0, floor_frac=0.20, floor_min=10.0),
    'K_x0.5': dict(K=10),
    'K_x2': dict(K=40),
    'part_x0.5': dict(min_part=0.10),
    'part_x2': dict(min_part=0.40),
}


def load_dandi(fn):
    import h5py
    with h5py.File(fn, 'r') as f:
        g = f['processing/ecephys/detected_spikes']
        t = g['time_from_first_spike_seconds'][:]
        c = g['source_channel'][:].astype(np.int64)
        m = f['processing/ecephys/source_mapping']
        a = f['processing/ecephys/channel_electrode_associations']
        sc = a['source_channel'][:].astype(np.int64)
        row = a['source_mapping_row'][:]
        x, y = m['source_x'][:][row], m['source_y'][:][row]
        T = float(f['processing/ecephys/conversion_provenance/duration_s'][0])
    lut = np.full(max(c.max(), sc.max()) + 1, -1, np.int64)
    lut[sc] = np.arange(len(sc))
    ci = lut[c]
    ok = ci >= 0
    order = np.argsort(t[ok], kind='stable')
    return t[ok][order], ci[ok][order], x.astype(float), y.astype(float), T, int((~ok).sum()), sc


def load_wag(fn):
    import pandas as pd
    df = pd.read_csv(fn, sep=r'\s+', header=None, usecols=[0, 1], names=['t', 'c'], compression='bz2')
    t = df['t'].to_numpy(float)
    c = df['c'].to_numpy(np.int64)
    ok = (c >= 0) & (c < 60)
    t, c = t[ok], c[ok]
    order = np.argsort(t, kind='stable')
    t, c = t[order], c[order]
    t = t - t.min() if len(t) else t
    x = np.array(_COLS, float) * 200.0
    y = np.array(_ROWS, float) * 200.0
    T = float(np.ceil(t.max())) if len(t) else 0.0
    return t, c, x, y, T, int((~ok).sum()), np.arange(60)


def job(args):
    ds, fn, outdir = args
    base = os.path.basename(fn)
    stem = base.replace('_ecephys.nwb', '').replace('.spk.txt.bz2', '')
    if os.path.exists(os.path.join(outdir, stem + '.pkl')):
        return stem, 'skip (exists)'
    try:
        if ds == 'dandi':
            t, c, x, y, T, n_unres, chid = load_dandi(fn)
            sub = base.split('_')[0].replace('sub-', '')
            ymd = base.split('_')[1].split('-')[-1]
            meta = dict(session=base.replace('_ecephys.nwb', ''), culture=sub,
                        date=str(date(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6]))), n_unresolved_spikes=n_unres)
            grid = GRID_DANDI
        else:
            t, c, x, y, T, n_unres, chid = load_wag(fn)
            b, cu, dv = base.replace('.spk.txt.bz2', '').split('-')
            meta = dict(session=f'{b}-{cu}-{dv}', culture=f'{b}-{cu}', batch=int(b), div=int(dv))
            grid = GRID_WAG
        meta['n_spikes_total'] = len(t)
        meta['excluded_empty'] = bool(len(t) < MIN_RATE_PER_CH * len(x) * max(T, 1))
        rows, extras = {}, {}
        if not meta['excluded_empty']:
            for vn, mod in VARIANTS.items():
                P = dict(PARAMS, **mod)
                r, e = analyze(t, c, x, y, T, grid, P)
                rows[vn] = {**meta, **r}
                if vn == 'base':
                    extras = {k: v for k, v in e.items()}
        else:
            rows['base'] = dict(meta)
        res = dict(rows=rows, extras=extras, chan_x=x, chan_y=y, chan_id=chid)
        tmp = os.path.join(outdir, meta['session'] + '.pkl.tmp')
        with open(tmp, 'wb') as fh:
            pickle.dump(res, fh)
        os.replace(tmp, os.path.join(outdir, meta['session'] + '.pkl'))
        return meta['session'], 'ok'
    except Exception as exc:  # report and continue
        import traceback
        return base, 'ERR ' + repr(exc) + traceback.format_exc()


def main():
    ds, ddir, odir = sys.argv[1], sys.argv[2], sys.argv[3]
    nproc = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    os.makedirs(odir, exist_ok=True)
    if ds == 'dandi':
        files = sorted(glob.glob(os.path.join(ddir, 'sub-*', '*.nwb')))
        files.sort(key=os.path.getsize, reverse=True)
    else:
        files = sorted(glob.glob(os.path.join(ddir, '*.spk.txt.bz2')))
        files.sort(key=os.path.getsize, reverse=True)
    if len(sys.argv) > 5:
        files = [f for f in files if any(s in f for s in sys.argv[5].split(','))]
    with Pool(nproc) as pool:
        for name, status in pool.imap_unordered(job, [(ds, f, odir) for f in files]):
            print(name, status[:2000], flush=True)


if __name__ == '__main__':
    main()
