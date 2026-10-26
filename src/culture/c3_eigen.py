"""C3 predictor (v): leading (Perron) eigenvector of an inter-burst functional coupling matrix.

Coupling = Pearson correlation of 20-ms spike counts between active channels, computed only on
inter-burst bins (outside [burst start - 100 ms, burst end + 100 ms], segments >= 1 s, i.e. the
same exclusion as C4); channels need >= 30 inter-burst spikes; negative values -> 0; diagonal 0;
pairs closer than 50 um -> 0 (duplicate detections of one neuron on neighbouring electrodes).
The eigenvector component of each channel is compared with its share of the first K spikes of
bursts and, per 100-um bin, with the smoothed ignition probability -- alongside the plain
inter-burst rate on exactly the same channels / bins.

usage: python -I src/culture/c3_eigen.py <project_root> dandi|wagenaar [nproc]
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
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import PARAMS, partial_spearman, runs, spearman  # noqa: E402
from run_sessions import GRID_DANDI, GRID_WAG, load_dandi, load_wag  # noqa: E402

EP = dict(bin=0.020, min_spikes=30, min_dist_um=50.0)


def eig_component(t, ch, x, y, T, st, en, act):
    dt = EP['bin']
    nb = int(np.ceil(T / dt)) + 1
    d = np.zeros(nb + 1, int)
    pad = PARAMS['excl_pad']
    if len(st):
        np.add.at(d, np.clip(np.floor((st - pad) / dt).astype(int), 0, nb), 1)
        np.add.at(d, np.clip(np.ceil((en + pad) / dt).astype(int), 0, nb), -1)
    excl = np.cumsum(d)[:nb] > 0
    s_, e_ = runs(~excl)
    okseg = (e_ - s_) * dt >= PARAMS['min_seg']
    keepbin = np.zeros(nb, bool)
    for a, b in zip(s_[okseg], e_[okseg]):
        keepbin[a:b] = True
    kb = np.flatnonzero(keepbin)
    remap = np.full(nb, -1)
    remap[kb] = np.arange(len(kb))
    bi = (t / dt).astype(int)
    sel = act[ch] & (remap[bi] >= 0)
    C = len(x)
    n_ib = np.bincount(ch[sel], minlength=C)
    use = act & (n_ib >= EP['min_spikes'])
    cu = np.flatnonzero(use)
    if len(cu) < 10 or len(kb) < 100:
        return None
    lut = np.full(C, -1)
    lut[cu] = np.arange(len(cu))
    s2 = sel & (lut[ch] >= 0)
    X = sparse.csr_matrix((np.ones(s2.sum()), (lut[ch[s2]], remap[bi[s2]])), shape=(len(cu), len(kb)))
    n = len(kb)
    mu = np.asarray(X.sum(1)).ravel() / n
    G = (X @ X.T).toarray() / n
    cov = G - np.outer(mu, mu)
    sd = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    R = cov / np.outer(sd, sd)
    R[R < 0] = 0
    np.fill_diagonal(R, 0)
    D = np.hypot(x[cu, None] - x[None, cu], y[cu, None] - y[None, cu])
    R[D < EP['min_dist_um']] = 0
    w, V = np.linalg.eigh(R)
    v = np.abs(V[:, -1])
    comp = np.full(C, np.nan)
    comp[cu] = v
    rate_ib = np.full(C, np.nan)
    rate_ib[cu] = n_ib[cu] / (n * dt)
    # participation ratio of the eigenvector (how many channels carry it)
    pr = float((v ** 2).sum() ** 2 / (v ** 4).sum())
    return comp, rate_ib, dict(eig_n_channels=int(len(cu)), eig_lambda1=float(w[-1]), eig_lambda2=float(w[-2]),
                               eig_participation_ratio=pr, eig_ib_seconds=float(n * dt))


def job(args):
    ds, pkl, root = args
    r = pickle.load(open(pkl, 'rb'))
    row = r['rows']['base']
    ex = r['extras']
    if row.get('excluded_empty') or row.get('n_bursts_ign', 0) < PARAMS['min_bursts']:
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
    rate = np.bincount(c, minlength=len(x)) / T
    act = rate >= PARAMS['active_rate']
    res = eig_component(t, c, x, y, T, ex['burst_start'], ex['burst_end'], act)
    if res is None:
        return dict(session=s, culture=row['culture'])
    comp, rate_ib, info = res
    ac = ex['ch_active']
    fk = ex['ch_first_share']
    ev, rv = comp[ac], rate_ib[ac]
    ok = np.isfinite(ev)
    # routing-density control: number of active channels within 200 um
    nch_ch = (np.hypot(x[ac, None] - x[None, ac], y[ac, None] - y[None, ac]) <= PARAMS['dens_radius'] + 1e-6).sum(1).astype(float)
    out = dict(session=s, culture=row['culture'], **info,
               prho_ch_eig_given_nch=partial_spearman(fk[ok], ev[ok], nch_ch[ok]),
               rho_ch_nch=spearman(fk[ok], nch_ch[ok]),
               rho_ch_eig=spearman(fk[ok], ev[ok]), rho_ch_ibrate_same=spearman(fk[ok], rv[ok]),
               prho_ch_eig_given_ibrate=partial_spearman(fk[ok], ev[ok], rv[ok]),
               rho_eig_vs_ibrate=spearman(ev[ok], rv[ok]))
    # per bin: mean eigenvector component of channels in the bin
    w, x0, y0, ny, nx = grid['w'], grid['x0'], grid['y0'], grid['ny'], grid['nx']
    ix = np.clip(((x - x0) // w).astype(int), 0, nx - 1)
    iy = np.clip(((y - y0) // w).astype(int), 0, ny - 1)
    chbin = ix * ny + iy
    okc = np.isfinite(comp)
    sb = ex['bin_index']
    num = np.bincount(chbin[okc], weights=comp[okc], minlength=nx * ny)[sb]
    numr = np.bincount(chbin[okc], weights=rate_ib[okc], minlength=nx * ny)[sb]
    den = np.bincount(chbin[okc], minlength=nx * ny)[sb]
    eb = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    rb = np.where(den > 0, numr / np.maximum(den, 1), np.nan)
    ib = ex['bin_ign']
    bxc = x0 + (sb // ny + 0.5) * w
    byc = y0 + (sb % ny + 0.5) * w
    nch_b = (np.hypot(bxc[:, None] - x[None, ac], byc[:, None] - y[None, ac]) <= PARAMS['dens_radius'] + 1e-6).sum(1).astype(float)
    out.update(prho_bin_eig_given_nch=partial_spearman(ib, eb, nch_b), rho_bin_eig_vs_nch=spearman(eb, nch_b))
    out.update(rho_bin_eig=spearman(ib, eb), rho_bin_ibrate_same=spearman(ib, rb),
               prho_bin_eig_given_ibrate=partial_spearman(ib, eb, rb))
    return out


def main():
    root, ds = sys.argv[1], sys.argv[2]
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    pkls = sorted(glob.glob(os.path.join(root, 'results/culture/cache', ds, '*.pkl')))
    with Pool(nproc) as pool:
        rows = [r for r in pool.imap_unordered(job, [(ds, p, root) for p in pkls]) if r is not None]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(root, 'results/culture', f'c3_eigen_{ds}.csv'), index=False)
    print(ds, len(df))


if __name__ == '__main__':
    main()
