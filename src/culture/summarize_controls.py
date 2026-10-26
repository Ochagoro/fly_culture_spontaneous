"""Summaries of the eigenvector predictor (C3 v), the added controls and the onset-composition check.

usage: python -I src/culture/summarize_controls.py <project_root>   -> results/culture/summary_controls.json
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = sys.argv[1]
RES = os.path.join(ROOT, 'results', 'culture')


def w(x):
    x = pd.Series(x, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 6:
        return dict(n=int(len(x)))
    return dict(n=int(len(x)), median=float(x.median()), q25=float(x.quantile(.25)), q75=float(x.quantile(.75)),
                frac_pos=float(np.mean(x > 0)), p=float(stats.wilcoxon(x).pvalue))


def cult(x, cu):
    m = pd.Series(x, dtype=float).groupby(cu).median().dropna()
    k = int((m > 0).sum())
    return dict(n_cultures=int(len(m)), n_pos=k, sign_p=float(stats.binomtest(k, len(m)).pvalue) if len(m) else np.nan)


def dev(df, col):
    r = []
    for _, g in df.groupby('culture'):
        g = g.dropna(subset=[col])
        if len(g) >= 5 and g[col].nunique() > 1:
            r.append(stats.spearmanr(g['day'], g[col])[0])
    r = np.array(r)
    return dict(n_cultures=int(len(r)), n_pos=int((r > 0).sum()), median_rho=float(np.median(r)),
                sign_p=float(stats.binomtest(int((r > 0).sum()), len(r)).pvalue),
                wilcoxon_p=float(stats.wilcoxon(r).pvalue) if len(r) >= 5 else np.nan)


out = {}
for ds in ['dandi', 'wagenaar']:
    R = {}
    base = pd.read_csv(os.path.join(RES, f'sessions_{ds}.csv'))
    E = pd.read_csv(os.path.join(RES, f'c3_eigen_{ds}.csv'))
    C = pd.read_csv(os.path.join(RES, f'controls_{ds}.csv')).merge(base[['session', 'day']], on='session')
    # C3 (v) eigenvector
    R['eigen'] = {k: dict(**w(E[k]), culture=cult(E[k], E['culture'])) for k in
                  ['rho_ch_eig', 'rho_ch_ibrate_same', 'prho_ch_eig_given_ibrate', 'prho_ch_eig_given_nch', 'rho_eig_vs_ibrate',
                   'rho_bin_eig', 'rho_bin_ibrate_same', 'prho_bin_eig_given_ibrate', 'prho_bin_eig_given_nch', 'rho_bin_eig_vs_nch']}
    for lv in ['ch', 'bin']:
        d = E[f'rho_{lv}_eig'] - E[f'rho_{lv}_ibrate_same']
        R['eigen'][f'{lv}_eig_minus_ibrate'] = dict(**w(d), frac_eig_beats_rate=float(np.mean(d.dropna() > 0)),
                                                    culture=cult(d, E['culture']))
    R['eigen']['n_channels_median'] = float(E['eig_n_channels'].median())
    R['eigen']['participation_ratio_median'] = float(E['eig_participation_ratio'].median())
    # C2a random-window null
    g = C.dropna(subset=['ign_conc_rw'])
    R['C2a_randwin'] = dict(conc_obs=float(g.ign_conc.median()), conc_rw=float(g.ign_conc_rw.median()),
                            conc_minus_rw=dict(**w(g.ign_conc - g.ign_conc_rw), culture=cult(g.ign_conc - g.ign_conc_rw, g.culture)),
                            frac_sessions_p_lt_0_05=float(np.mean(g.ign_conc_rw_p < 0.05)),
                            entropy_obs=float(g.ign_entropy.median()), entropy_rw=float(g.ign_entropy_rw.median()),
                            entropy_minus_rw=dict(**w(g.ign_entropy - g.ign_entropy_rw), culture=cult(g.ign_entropy - g.ign_entropy_rw, g.culture)),
                            spread_obs=float(g.ign_spread_um.median()), spread_rw=float(g.ign_spread_rw_um.median()),
                            spread_minus_rw=w(g.ign_spread_um - g.ign_spread_rw_um),
                            dist_first_vs_rw_um=float(g.median_dist_first_vs_rw_um.median()))
    R['C3_routing_density'] = dict(rho_bin_nch200=dict(**w(g.rho_bin_nch200), culture=cult(g.rho_bin_nch200, g.culture)),
                                   prho_bin_dens_given_nch=dict(**w(g.prho_bin_dens_given_nch), culture=cult(g.prho_bin_dens_given_nch, g.culture)),
                                   rho_bin_meanlocalrate=dict(**w(g.rho_bin_meanlocalrate), culture=cult(g.rho_bin_meanlocalrate, g.culture)))
    # C4 detrended
    R['C4_detrended'] = dict(m_ib_detr2s=w(C.m_ib_detr2s), m_ib_segmean=w(C.m_ib_segmean), m_all_detr2s=w(C.m_all_detr2s),
                             tau_ib_detr2s_ms_median=float(C.tau_ib_detr2s_ms.replace(np.inf, np.nan).median()),
                             frac_m_ib_detr_ge_1=float(np.mean(C.m_ib_detr2s >= 1)),
                             dev_m_ib_detr2s=dev(C, 'm_ib_detr2s'), dev_m_all_detr2s=dev(C, 'm_all_detr2s'))
    # onset composition: share of first-K spikes from the top-10 % out-of-burst-rate channels,
    # vs their share of out-of-burst spikes and of in-burst spikes
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, 'cache', ds, '*.pkl'))):
        r = pickle.load(open(f, 'rb'))
        ex, row = r['extras'], r['rows']['base']
        if 'ch_first_share' not in ex:
            continue
        fk, so, si = ex['ch_first_share'], ex['ch_out'], ex['ch_in']
        top = np.argsort(so)[::-1][:int(np.ceil(0.1 * len(so)))]
        rows.append(dict(session=row['session'], culture=row['culture'], first=float(fk[top].sum()),
                         out=float(so[top].sum() / so.sum()), inb=float(si[top].sum() / max(si.sum(), 1e-12))))
    O = pd.DataFrame(rows)
    O.to_csv(os.path.join(RES, f'onset_composition_{ds}.csv'), index=False)
    R['onset_composition_top10pct_out_channels'] = dict(
        share_first=float(O['first'].median()), share_out=float(O['out'].median()), share_in=float(O['inb'].median()),
        first_minus_out=dict(**w(O['first'] - O['out']), culture=cult(O['first'] - O['out'], O['culture'])),
        first_minus_in=dict(**w(O['first'] - O['inb']), culture=cult(O['first'] - O['inb'], O['culture'])))
    out[ds] = R
json.dump(out, open(os.path.join(RES, 'summary_controls.json'), 'w'), indent=1)
print(json.dumps(out, indent=0)[:100000])
