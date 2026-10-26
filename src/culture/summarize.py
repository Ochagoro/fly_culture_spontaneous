"""Cross-session statistics (C2b stability, C3, C4, C5, C6, C7, sensitivity) and output files.

usage: python -I src/culture/summarize.py <project_root>
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

ROOT = sys.argv[1]
RES = os.path.join(ROOT, 'results', 'culture')
CACHE = os.path.join(RES, 'cache')
MINB = 20
OUT = {}  # headline numbers -> results/culture/summary.json


def load(ds):
    rows, extras = {}, {}
    for f in sorted(glob.glob(os.path.join(CACHE, ds, '*.pkl'))):
        r = pickle.load(open(f, 'rb'))
        for vn, row in r['rows'].items():
            rows.setdefault(vn, []).append(row)
        extras[r['rows']['base']['session']] = r['extras']
    dfs = {vn: pd.DataFrame(v) for vn, v in rows.items()}
    return dfs, extras


def add_time(df, ds):
    if ds == 'dandi':
        d = pd.to_datetime(df['date'])
        df['day'] = (d - d.groupby(df['culture']).transform('min')).dt.days
    else:
        df['day'] = df['div']
    return df.sort_values(['culture', 'day']).reset_index(drop=True)


def wsr(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5 or np.all(x == 0):
        return dict(n=int(len(x)), median=float(np.median(x)) if len(x) else np.nan, p=np.nan)
    return dict(n=int(len(x)), median=float(np.median(x)), q25=float(np.percentile(x, 25)),
                q75=float(np.percentile(x, 75)), frac_pos=float(np.mean(x > 0)),
                p=float(stats.wilcoxon(x).pvalue))


def culture_sign(df, col, ref=None):
    """Per-culture median of col (or col - ref); sign consistency across cultures."""
    v = df[col] - (df[ref] if ref else 0)
    med = v.groupby(df['culture']).median().dropna()
    npos = int((med > 0).sum())
    p = stats.binomtest(npos, len(med)).pvalue if len(med) else np.nan
    return dict(n_cultures=int(len(med)), n_pos=npos, sign_p=float(p), median_of_culture_medians=float(med.median()))


def dev_trend(df, col, min_n=5):
    rhos = {}
    for cu, g in df.groupby('culture'):
        g = g[np.isfinite(g[col].astype(float))]
        if len(g) >= min_n and g[col].nunique() > 1:
            rhos[cu] = float(stats.spearmanr(g['day'], g[col])[0])
    r = np.array(list(rhos.values()))
    r = r[np.isfinite(r)]
    npos = int((r > 0).sum())
    return dict(n_cultures=int(len(r)), n_pos=npos, median_rho=float(np.median(r)) if len(r) else np.nan,
                sign_p=float(stats.binomtest(npos, len(r)).pvalue) if len(r) else np.nan,
                wilcoxon_p=float(stats.wilcoxon(r).pvalue) if len(r) >= 5 else np.nan, per_culture=rhos)


def corr_maps(a, b, mask):
    x, y = a[mask], b[mask]
    if mask.sum() < 10 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def stability(df, extras, w, x0):
    ok = df[df['n_bursts_ign'] >= MINB]
    sess = list(ok['session'])
    info = {s: (c, d) for s, c, d in zip(ok['session'], ok['culture'], ok['day'])}

    def pair(s1, s2):
        e1, e2 = extras[s1], extras[s2]
        m = e1['support'] & e2['support']
        p1, p2 = e1['ign_pmap'], e2['ign_pmap']
        a1, a2 = e1['burst_density_map'], e2['burst_density_map']
        h1 = np.array(np.unravel_index(np.argmax(p1), p1.shape))
        h2 = np.array(np.unravel_index(np.argmax(p2), p2.shape))
        jac = (e1['support'] & e2['support']).sum() / max((e1['support'] | e2['support']).sum(), 1)
        return dict(s1=s1, s2=s2, culture1=info[s1][0], culture2=info[s2][0], day1=info[s1][1], day2=info[s2][1],
                    r_ign=corr_maps(p1, p2, m), r_act=corr_maps(a1, a2, m),
                    r_resid=corr_maps(p1 / max(p1.sum(), 1e-12) - a1, p2 / max(p2.sum(), 1e-12) - a2, m),
                    hot_dist_um=float(np.hypot(*(h1 - h2)) * w), support_jaccard=float(jac))

    rows = []
    for cu, g in ok.groupby('culture'):
        ss = list(g.sort_values('day')['session'])
        for s1, s2 in zip(ss[:-1], ss[1:]):
            rows.append({**pair(s1, s2), 'kind': 'consecutive'})
    for s1, s2 in combinations(sess, 2):
        if info[s1][0] != info[s2][0]:
            rows.append({**pair(s1, s2), 'kind': 'between_culture'})
    return pd.DataFrame(rows)


def stab_stats(P):
    out = {}
    cons, betw = P[P.kind == 'consecutive'], P[P.kind == 'between_culture']
    for k in ['r_ign', 'r_act', 'r_resid', 'hot_dist_um', 'support_jaccard']:
        a, b = cons[k].dropna(), betw[k].dropna()
        out[k] = dict(consecutive_median=float(a.median()), between_median=float(b.median()),
                      n_cons=int(len(a)), n_between=int(len(b)),
                      mwu_p=float(stats.mannwhitneyu(a, b).pvalue) if len(a) > 2 and len(b) > 2 else np.nan)
        # culture-level: each culture's mean consecutive value vs mean of its between-culture pairs
        cm = cons.groupby('culture1')[k].mean()
        bm = {c: betw[(betw.culture1 == c) | (betw.culture2 == c)][k].mean() for c in cm.index}
        d = np.array([cm[c] - bm[c] for c in cm.index])
        d = d[np.isfinite(d)]
        out[k]['culture_level_n_greater'] = int((d > 0).sum())
        out[k]['culture_level_n'] = int(len(d))
    return out


def main():
    os.makedirs(os.path.join(RES, 'diag'), exist_ok=True)
    allres = {}
    for ds, w, x0 in [('dandi', 100.0, 0.0), ('wagenaar', 200.0, -100.0)]:
        dfs, extras = load(ds)
        for vn in dfs:
            dfs[vn] = add_time(dfs[vn], ds)
        base = dfs['base']
        base.to_csv(os.path.join(RES, f'sessions_{ds}.csv'), index=False)
        R = {}
        inc = base[~base['excluded_empty']]
        ign = inc[inc['n_bursts_ign'] >= MINB]
        R['n'] = dict(sessions_total=int(len(base)), cultures_total=int(base['culture'].nunique()),
                      sessions_excluded_empty=int(base['excluded_empty'].sum()),
                      sessions_included=int(len(inc)), cultures_included=int(inc['culture'].nunique()),
                      sessions_ign=int(len(ign)), cultures_ign=int(ign['culture'].nunique()),
                      bursts_total=int(inc['n_bursts'].sum()), bursts_ign=int(ign['n_bursts_ign'].sum()),
                      sessions_with_zero_bursts=int((inc['n_bursts'] == 0).sum()))
        # C1
        R['C1'] = {k: dict(median=float(inc[k].median()), q25=float(inc[k].quantile(.25)), q75=float(inc[k].quantile(.75)),
                           n=int(inc[k].notna().sum()))
                   for k in ['burst_rate_per_min', 'median_burst_dur_ms', 'median_participation', 'frac_out_of_burst',
                             'frac_time_in_burst', 'n_active', 'mean_rate_active', 'n_small_events']}
        # C2a
        R['C2a'] = dict(
            conc_minus_null=wsr(ign['ign_conc'] - ign['ign_conc_null']),
            conc=dict(median=float(ign['ign_conc'].median()), null_median=float(ign['ign_conc_null'].median())),
            frac_sessions_conc_p_lt_0_05=float((ign['ign_conc_p'] < 0.05).mean()),
            entropy_minus_null=wsr(ign['ign_entropy'] - ign['ign_entropy_null']),
            entropy=dict(median=float(ign['ign_entropy'].median()), null_median=float(ign['ign_entropy_null'].median())),
            conc_win_minus_null=wsr(ign['ign_conc_win'] - ign['ign_conc_null']),
            earliest_conc_minus_null=wsr(ign['earliest_conc'] - ign['earliest_conc_null']),
            earliest=dict(median=float(ign['earliest_conc'].median()), null_median=float(ign['earliest_conc_null'].median())),
            spread_minus_null=wsr(ign['ign_spread_um'] - ign['ign_spread_null_um']),
            spread=dict(median=float(ign['ign_spread_um'].median()), null_median=float(ign['ign_spread_null_um'].median())),
            log_compactness=wsr(np.log(ign['compactness_ratio'])),
            compactness_median=float(ign['compactness_ratio'].median()),
            radius=dict(first=float(ign['ign_radius_um'].median()), rand=float(ign['rand_radius_um'].median())),
            culture_sign_conc=culture_sign(ign, 'ign_conc', 'ign_conc_null'),
            culture_sign_entropy=culture_sign(ign, 'ign_entropy', 'ign_entropy_null'),
        )
        # C2b stability
        P = stability(inc, extras, w, x0)
        P.to_csv(os.path.join(RES, f'ignition_stability_pairs_{ds}.csv'), index=False)
        R['C2b'] = stab_stats(P) if len(P) else {}
        # C3
        R['C3'] = {k: dict(**wsr(ign[k]), **{'culture': culture_sign(ign, k)})
                   for k in ['rho_bin_out', 'rho_bin_in', 'rho_bin_dens', 'rho_bin_lat', 'prho_bin_out_given_in',
                             'prho_bin_dens_given_in', 'rho_ch_out', 'rho_ch_in', 'rho_ch_dens', 'rho_ch_lat',
                             'prho_ch_out_given_in', 'prho_ch_dens_given_in', 'rho_ch_pre', 'rho_ch_enrich_outin']}
        R['C3']['out_vs_in_bin'] = wsr(ign['rho_bin_out'] - ign['rho_bin_in'])
        R['C3']['out_vs_in_ch'] = wsr(ign['rho_ch_out'] - ign['rho_ch_in'])
        # C4
        hasb = inc[inc['n_bursts'] >= 1]
        R['C4'] = dict(m_all=wsr(inc['m_all'] - 0), m_ib=wsr(inc['m_ib'] - 0),
                       m_all_minus_m_ib=wsr(inc['m_all'] - inc['m_ib']),
                       m_all_minus_m_ib_with_bursts=wsr(hasb['m_all'] - hasb['m_ib']),
                       tau_all_ms_median=float(np.nanmedian(inc['tau_all_ms'].replace(np.inf, np.nan))),
                       tau_ib_ms_median=float(np.nanmedian(inc['tau_ib_ms'].replace(np.inf, np.nan))),
                       frac_m_all_ge_0_99=float((inc['m_all'] >= 0.99).mean()),
                       frac_m_ib_ge_0_99=float((inc['m_ib'] >= 0.99).mean()),
                       frac_m_all_ge_1=float((inc['m_all'] >= 1).mean()),
                       frac_m_ib_ge_1=float((inc['m_ib'] >= 1).mean()),
                       rho_m_all_vs_frac_time_in_burst=float(stats.spearmanr(inc['frac_time_in_burst'], inc['m_all'], nan_policy='omit')[0]),
                       rho_m_ib_vs_burst_rate=float(stats.spearmanr(inc['burst_rate_per_min'], inc['m_ib'], nan_policy='omit')[0]),
                       kmax_sens=dict(m_ib_k25=float(inc['m_ib_k25'].median()), m_ib_k50=float(inc['m_ib'].median()),
                                      m_ib_k100=float(inc['m_ib_k100'].median()), m_all_k25=float(inc['m_all_k25'].median()),
                                      m_all_k50=float(inc['m_all'].median()), m_all_k100=float(inc['m_all_k100'].median())))
        # C5 development
        cols = ['burst_rate_per_min', 'median_burst_dur_ms', 'median_participation', 'frac_out_of_burst',
                'mean_rate_active', 'n_active', 'm_all', 'm_ib', 'ib_rate_hz_per_ch']
        R['C5'] = {k: dev_trend(inc, k) for k in cols}
        ig2 = ign.assign(conc_excess=ign['ign_conc'] - ign['ign_conc_null'],
                         ent_excess=ign['ign_entropy'] - ign['ign_entropy_null'],
                         log_compact=np.log(ign['compactness_ratio']))
        for k in ['conc_excess', 'ent_excess', 'log_compact', 'ign_spread_um', 'rho_bin_dens', 'rho_ch_out']:
            R['C5'][k] = dev_trend(ig2, k, min_n=4)
        # C6 halves
        R['C6'] = {}
        for k in ['burst_rate', 'frac_out', 'm_all', 'm_ib', 'rate']:
            d = inc[f'h2_{k}'] - inc[f'h1_{k}']
            R['C6'][k] = dict(**wsr(d), h1_median=float(inc[f'h1_{k}'].median()), h2_median=float(inc[f'h2_{k}'].median()),
                              culture=culture_sign(inc.assign(_d=d), '_d'))
        bb = inc[inc['n_bursts'] >= MINB]
        R['C6']['m_ib_bursting_sessions'] = wsr(bb['h2_m_ib'] - bb['h1_m_ib'])
        R['C6']['m_all_bursting_sessions'] = wsr(bb['h2_m_all'] - bb['h1_m_all'])
        # sensitivity
        sens = []
        for vn, d in dfs.items():
            d = d[~d['excluded_empty']]
            g = d[d['n_bursts_ign'] >= MINB]
            sens.append(dict(variant=vn, n_sessions_ign=len(g), bursts_total=int(d['n_bursts'].sum()),
                             burst_rate=d['burst_rate_per_min'].median(), frac_out=d['frac_out_of_burst'].median(),
                             participation=d['median_participation'].median(), dur_ms=d['median_burst_dur_ms'].median(),
                             m_all=d['m_all'].median(), m_ib=d['m_ib'].median(),
                             conc=g['ign_conc'].median(), conc_null=g['ign_conc_null'].median(),
                             frac_conc_gt_null=(g['ign_conc'] > g['ign_conc_null']).mean(),
                             compactness=g['compactness_ratio'].median(),
                             rho_bin_dens=g['rho_bin_dens'].median(), rho_bin_out=g['rho_bin_out'].median(),
                             rho_bin_in=g['rho_bin_in'].median(), rho_ch_out=g['rho_ch_out'].median(),
                             rho_ch_in=g['rho_ch_in'].median(), prho_ch_out_given_in=g['prho_ch_out_given_in'].median(),
                             dev_rho_m_ib=dev_trend(d, 'm_ib')['median_rho'],
                             dev_rho_burst_rate=dev_trend(d, 'burst_rate_per_min')['median_rho'],
                             c6_dm_ib=np.nanmedian(d['h2_m_ib'] - d['h1_m_ib']),
                             c6_dburst_rate=np.nanmedian(d['h2_burst_rate'] - d['h1_burst_rate'])))
        S = pd.DataFrame(sens)
        S.to_csv(os.path.join(RES, f'sensitivity_{ds}.csv'), index=False)
        R['sensitivity'] = S.to_dict(orient='records')
        allres[ds] = R
        # maps npz
        if ds == 'dandi':
            ss = list(ign['session'])
            np.savez_compressed(
                os.path.join(RES, 'ignition_maps_dandi.npz'),
                session=np.array(ss), culture=ign['culture'].astype(str).to_numpy().astype('U'), day=ign['day'].to_numpy(),
                ign_pmap=np.stack([extras[s]['ign_pmap'] for s in ss]),
                ign_count=np.stack([extras[s]['ign_count_map'] for s in ss]),
                burst_density_map=np.stack([extras[s]['burst_density_map'] for s in ss]),
                support=np.stack([extras[s]['support'] for s in ss]),
                ign_x=np.concatenate([extras[s]['ign_x'] for s in ss]),
                ign_y=np.concatenate([extras[s]['ign_y'] for s in ss]),
                ign_session_index=np.concatenate([np.full(len(extras[s]['ign_x']), i) for i, s in enumerate(ss)]),
                grid_x_edges=np.arange(0, 3901, 100.0), grid_y_edges=np.arange(0, 2101, 100.0),
                note=np.array('ign_pmap: Gaussian-smoothed (sigma 100 um) fraction of bursts whose ignition point '
                              '(centroid of first 20 spikes) falls in each 100-um bin; axis0=x, axis1=y; '
                              'day = days since the culture\'s first session (DIV unknown)'))
        else:
            ss = list(ign['session'])
            np.savez_compressed(
                os.path.join(RES, 'ignition_maps_wagenaar.npz'),
                session=np.array(ss), culture=ign['culture'].astype(str).to_numpy().astype('U'), day=ign['day'].to_numpy(),
                ign_pmap=np.stack([extras[s]['ign_pmap'] for s in ss]),
                ign_count=np.stack([extras[s]['ign_count_map'] for s in ss]),
                support=np.stack([extras[s]['support'] for s in ss]))
    with open(os.path.join(RES, 'summary.json'), 'w') as fh:
        json.dump(allres, fh, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print(json.dumps(allres, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))[:200])


if __name__ == '__main__':
    main()
