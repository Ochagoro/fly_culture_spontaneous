"""Controls added after the first pass (see notes, section 'controls added after first look'):

(a) C2a random-window null: centroid of K consecutive spikes starting at a random position inside the
    same burst (keeps the spatial averaging of a K-spike centroid and the local wave-front structure,
    tests whether the *beginning* of the burst is special). 50 repetitions.
(b) C3 electrode-density control: number of active channels within 200 um of each bin centre
    (routing density) and partial Spearman of ignition vs local activity density given it.
(c) C4 non-stationarity control: inter-burst m after subtracting a centred 2-s moving average within
    each inter-burst segment.

usage: python -I src/culture/extra_controls.py <project_root> dandi|wagenaar [nproc]
"""
from __future__ import annotations

import glob
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import PARAMS, entropy_conc, m_tau, partial_spearman, runs, spearman  # noqa: E402
from run_sessions import GRID_DANDI, GRID_WAG, load_dandi, load_wag  # noqa: E402

DETREND_S = 2.0


def job(args):
    ds, pkl, root = args
    r = pickle.load(open(pkl, 'rb'))
    row = r['rows']['base']
    ex = r['extras']
    if row.get('excluded_empty'):
        return None
    s = row['session']
    if ds == 'dandi':
        fn = os.path.join(root, 'data/culture/dandi002020/002020', s.split('_')[0], s + '_ecephys.nwb')
        t, c, x, y, T, _, _ = load_dandi(fn)
        grid = GRID_DANDI
    else:
        fn = os.path.join(root, 'data/culture/potter_dense', s + '.spk.txt.bz2')
        t, c, x, y, T, _, _ = load_wag(fn)
        grid = GRID_WAG
    P = PARAMS
    rng = np.random.default_rng(1)
    rate = np.bincount(c, minlength=len(x)) / T
    act = rate >= P['active_rate']
    keep = act[c]
    t, c = t[keep], c[keep]
    out = dict(session=s, culture=row['culture'])
    st, en = ex.get('burst_start', np.zeros(0)), ex.get('burst_end', np.zeros(0))
    # (c) detrended inter-burst m
    dt4 = P['mr_bin']
    n4 = int(np.ceil(T / dt4)) + 1
    a4 = np.bincount((t / dt4).astype(np.int64), minlength=n4).astype(float)
    d = np.zeros(n4 + 1, int)
    if len(st):
        np.add.at(d, np.clip(np.floor((st - P['excl_pad']) / dt4).astype(int), 0, n4), 1)
        np.add.at(d, np.clip(np.ceil((en + P['excl_pad']) / dt4).astype(int), 0, n4), -1)
    excl = np.cumsum(d)[:n4] > 0
    s_, e_ = runs(~excl)
    ok = (e_ - s_) * dt4 >= P['min_seg']
    w = int(DETREND_S / dt4)
    pieces = [a4[a:b] - ndimage.uniform_filter1d(a4[a:b], w, mode='nearest') for a, b in zip(s_[ok], e_[ok])]
    pieces_m = [a4[a:b] - a4[a:b].mean() for a, b in zip(s_[ok], e_[ok])]
    if pieces:
        out['m_ib_detr2s'], out['tau_ib_detr2s_ms'] = m_tau(np.concatenate(pieces), P['kmax'], dt4)
        out['m_ib_segmean'] = m_tau(np.concatenate(pieces_m), P['kmax'], dt4)[0]
    whole = a4 - ndimage.uniform_filter1d(a4, w, mode='nearest')
    out['m_all_detr2s'] = m_tau(whole, P['kmax'], dt4)[0]
    if row.get('n_bursts_ign', 0) < P['min_bursts']:
        return out
    # (a) random-window null for ignition centroids
    K = P['K']
    bid_first = np.searchsorted(t, st)
    bid_end = np.searchsorted(t, en)
    nsp = bid_end - bid_first
    sel = ex['ign_burst_idx']
    fb, nk = bid_first[sel], nsp[sel]
    gw, x0, y0, nx, ny = grid['w'], grid['x0'], grid['y0'], grid['nx'], grid['ny']

    def tobin(xx, yy):
        ix = np.clip(((xx - x0) // gw).astype(int), 0, nx - 1)
        iy = np.clip(((yy - y0) // gw).astype(int), 0, ny - 1)
        return ix * ny + iy

    n_sup = int(ex['support'].sum())
    H0, C0 = entropy_conc(tobin(ex['ign_x'], ex['ign_y']), n_sup, nx * ny)
    spread = lambda X, Y: float(np.sqrt(((X - X.mean()) ** 2 + (Y - Y.mean()) ** 2).mean()))  # noqa: E731
    Hs, Cs, Ss, Ds = [], [], [], []
    for _ in range(P['n_null']):
        j = fb + (rng.random(len(fb)) * (nk - K + 1)).astype(int)
        idx = j[:, None] + np.arange(K)[None, :]
        gx, gy = x[c[idx]].mean(1), y[c[idx]].mean(1)
        h, cc = entropy_conc(tobin(gx, gy), n_sup, nx * ny)
        Hs.append(h)
        Cs.append(cc)
        Ss.append(spread(gx, gy))
        Ds.append(np.median(np.hypot(gx - ex['ign_x'], gy - ex['ign_y'])))
    Hs, Cs = np.array(Hs), np.array(Cs)
    out.update(ign_conc=C0, ign_entropy=H0, ign_conc_rw=float(Cs.mean()), ign_entropy_rw=float(Hs.mean()),
               ign_conc_rw_p=float((np.sum(Cs >= C0) + 1) / (len(Cs) + 1)),
               ign_entropy_rw_p=float((np.sum(Hs <= H0) + 1) / (len(Hs) + 1)),
               ign_spread_um=spread(ex['ign_x'], ex['ign_y']), ign_spread_rw_um=float(np.mean(Ss)),
               median_dist_first_vs_rw_um=float(np.mean(Ds)))
    # (b) electrode-density control at bin level
    ac = ex['ch_active']
    sb = ex['bin_index']
    bxc = x0 + (sb // ny + 0.5) * gw
    byc = y0 + (sb % ny + 0.5) * gw
    nch200 = (np.hypot(bxc[:, None] - x[None, ac], byc[:, None] - y[None, ac]) <= P['dens_radius'] + 1e-6).sum(1).astype(float)
    ib, dens = ex['bin_ign'], ex['bin_dens']
    out.update(rho_bin_nch200=spearman(ib, nch200), prho_bin_dens_given_nch=partial_spearman(ib, dens, nch200),
               rho_bin_meanlocalrate=spearman(ib, dens / np.maximum(nch200, 1)))
    return out


def main():
    root, ds = sys.argv[1], sys.argv[2]
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    pkls = sorted(glob.glob(os.path.join(root, 'results/culture/cache', ds, '*.pkl')))
    with Pool(nproc) as pool:
        rows = [r for r in pool.imap_unordered(job, [(ds, p, root) for p in pkls]) if r is not None]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(root, 'results/culture', f'controls_{ds}.csv'), index=False)
    print(ds, len(df))


if __name__ == '__main__':
    main()
