"""Diagnostic figures and example-session arrays.

usage: python -I src/culture/figures.py <project_root>
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import PARAMS, analyze  # noqa: E402
from run_sessions import GRID_DANDI, GRID_WAG, load_dandi, load_wag  # noqa: E402

ROOT = sys.argv[1]
RES = os.path.join(ROOT, 'results', 'culture')
DIAG = os.path.join(RES, 'diag')
DD = os.path.join(ROOT, 'data', 'culture', 'dandi002020', '002020')
WD = os.path.join(ROOT, 'data', 'culture', 'potter_dense')
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})


def pick_examples(df, n_per, cultures=None):
    ok = df[(~df['excluded_empty']) & (df['n_bursts_ign'] >= 100)]
    out = []
    for cu in (cultures or ok['culture'].unique()):
        g = ok[ok['culture'] == cu]
        if len(g):
            med = g['n_spikes'].median()
            out.append(g.iloc[np.argmin(np.abs(g['n_spikes'] - med))]['session'])
        if len(out) >= n_per:
            break
    return out


def export_example(ds, session, df):
    if ds == 'dandi':
        cu = session.split('_')[0]
        fn = os.path.join(DD, cu, session + '_ecephys.nwb')
        t, c, x, y, T, _, chid = load_dandi(fn)
        grid = GRID_DANDI
    else:
        fn = os.path.join(WD, session + '.spk.txt.bz2')
        t, c, x, y, T, _, chid = load_wag(fn)
        grid = GRID_WAG
    r, e = analyze(t, c, x, y, T, grid, PARAMS, detail=True)
    rate = np.bincount(c, minlength=len(x)) / T
    act = rate >= PARAMS['active_rate']
    keep = act[c]
    t_, c_ = t[keep], c[keep]
    tq = np.round(t_ * 20000).astype(np.uint32) if ds == 'dandi' else None
    fnout = os.path.join(RES, f'example_{ds}_{session}.npz')
    np.savez_compressed(
        fnout,
        spike_time_s=(t_.astype(np.float64) if tq is None else np.zeros(0)),
        spike_frame_20kHz=(tq if tq is not None else np.zeros(0, np.uint32)),
        spike_channel_index=c_.astype(np.int16), channel_x_um=x, channel_y_um=y, channel_id=chid,
        channel_active=act, duration_s=T,
        burst_start_s=e['burst_start'], burst_end_s=e['burst_end'],
        ign_x_um=e.get('ign_x', np.zeros(0)), ign_y_um=e.get('ign_y', np.zeros(0)),
        ign_burst_index=e.get('ign_burst_idx', np.zeros(0, int)), earliest_channel_index=e.get('earliest_ch', np.zeros(0, int)),
        pop_count_10ms=e['c10'].astype(np.int32), pop_count_4ms=e['a4'].astype(np.int32),
        excluded_4ms_bins=e['excl4'], ign_pmap=e.get('ign_pmap', np.zeros(0)), support=e.get('support', np.zeros(0)),
        note=np.array('spike_frame_20kHz = round(time_from_first_spike_seconds*20000) (DANDI); '
                      'spike_time_s for Wagenaar. Only active channels (>=0.1 Hz). Ignition = centroid of first 20 spikes.'))
    print(fnout, os.path.getsize(fnout) / 1e6, 'MB')
    return t_, c_, x, y, T, e, r


def raster_fig(name, t, c, x, y, e, t0, span, fn):
    fig, axs = plt.subplots(3, 1, figsize=(10, 6.5), gridspec_kw=dict(height_ratios=[3, 1, 2.2]))
    sel = (t >= t0) & (t < t0 + span)
    order = np.argsort(np.argsort(x + 1e-3 * y))  # rank channels by x position
    axs[0].scatter(t[sel], order[c[sel]], s=0.2, c='k', lw=0, rasterized=True)
    for a, b in zip(e['burst_start'], e['burst_end']):
        if t0 <= a < t0 + span:
            for ax in axs[:2]:
                ax.axvspan(a, b, color='tab:green', alpha=0.25, lw=0)
    axs[0].set_ylabel('channel (sorted by x)')
    axs[0].set_xlim(t0, t0 + span)
    tt = np.arange(len(e['c10'])) * 0.01
    s2 = slice(int(t0 / 0.01), int((t0 + span) / 0.01))
    axs[1].plot(tt[s2], e['c10'][s2], lw=0.6, c='k')
    axs[1].set_ylabel('spikes / 10 ms')
    axs[1].set_xlim(t0, t0 + span)
    axs[1].set_xlabel('time (s)')
    ax = axs[2]
    ax.scatter(x, y, s=2, c='0.8', lw=0)
    if 'ign_x' in e:
        ax.scatter(e['ign_x'], e['ign_y'], s=4, c='tab:red', alpha=0.4, lw=0, label='ignition points (all bursts)')
        ax.legend(loc='upper right', fontsize=7, frameon=False)
    ax.set_aspect('equal')
    ax.set_xlabel('x (um)')
    ax.set_ylabel('y (um)')
    fig.suptitle(name, fontsize=9)
    fig.tight_layout()
    fig.savefig(fn, dpi=130)
    plt.close(fig)


def main():
    dfd = pd.read_csv(os.path.join(RES, 'sessions_dandi.csv'))
    dfw = pd.read_csv(os.path.join(RES, 'sessions_wagenaar.csv'))
    # ---- examples (2 DANDI cultures with most ignition-valid sessions, 1 Wagenaar)
    okd = dfd[(~dfd['excluded_empty']) & (dfd['n_bursts_ign'] >= 20)]
    cults = list(okd['culture'].value_counts().index[:2])
    exd = pick_examples(dfd, 2, cults)
    exw = pick_examples(dfw[dfw['culture'] == '6-3'], 1) or pick_examples(dfw, 1)
    for ds, ss in [('dandi', exd), ('wagenaar', exw)]:
        for s in ss:
            t, c, x, y, T, e, r = export_example(ds, s, dfd if ds == 'dandi' else dfw)
            t0 = float(e['burst_start'][len(e['burst_start']) // 2]) - 10 if len(e['burst_start']) else 600
            raster_fig(f'{s}: {r["n_bursts"]} bursts, {r["burst_rate_per_min"]:.1f}/min, out-of-burst {r["frac_out_of_burst"]:.2f}, '
                       f'm_all={r["m_all"]:.3f}, m_ib={r["m_ib"]:.3f}', t, c, x, y, e, t0, 60,
                       os.path.join(DIAG, f'raster_{ds}_{s}.png'))
    # ---- ignition maps across days for one culture
    Z = np.load(os.path.join(RES, 'ignition_maps_dandi.npz'))
    for cu in cults:
        idx = np.flatnonzero(Z['culture'] == str(cu))
        n = len(idx)
        nc = 6
        nr = int(np.ceil(n / nc))
        fig, axs = plt.subplots(nr, nc, figsize=(2.4 * nc, 1.5 * nr), squeeze=False)
        for k, ax in enumerate(axs.ravel()):
            ax.axis('off')
            if k < n:
                i = idx[k]
                pm = np.where(Z['support'][i], Z['ign_pmap'][i], np.nan)
                ax.imshow(pm.T, origin='lower', extent=[0, 3900, 0, 2100], cmap='magma')
                ax.set_title(f'day {Z["day"][i]} (n={int(Z["ign_count"][i].sum())})', fontsize=7)
        fig.suptitle(f'culture {cu}: ignition probability (centroid of first 20 spikes; 100-um grid, sigma 100 um)', fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(DIAG, f'ignition_maps_culture_{cu}.png'), dpi=110)
        plt.close(fig)
    # ---- m vs day / DIV and C1 vs day
    for ds, df, xl in [('dandi', dfd, 'days since first session'), ('wagenaar', dfw, 'DIV')]:
        inc = df[~df['excluded_empty']]
        fig, axs = plt.subplots(2, 3, figsize=(12, 6))
        for k, (col, lab) in enumerate([('m_all', 'm (whole session)'), ('m_ib', 'm (inter-burst)'),
                                        ('burst_rate_per_min', 'bursts / min'), ('frac_out_of_burst', 'fraction of spikes outside bursts'),
                                        ('median_participation', 'median participation'), ('tau_ib_ms', 'tau inter-burst (ms)')]):
            ax = axs.ravel()[k]
            for cu, g in inc.groupby('culture'):
                v = g[col].replace(np.inf, np.nan)
                ax.plot(g['day'], v, '-o', ms=2, lw=0.7, alpha=0.8 if ds == 'dandi' else 0.4, label=str(cu) if ds == 'dandi' else None)
            ax.set_xlabel(xl)
            ax.set_ylabel(lab)
            if col.startswith('m_'):
                ax.axhline(1, c='k', lw=0.5, ls=':')
            if col == 'tau_ib_ms':
                ax.set_yscale('log')
        if ds == 'dandi':
            axs[0, 0].legend(fontsize=6, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(DIAG, f'development_{ds}.png'), dpi=120)
        plt.close(fig)
    # ---- ignition concentration vs null and C3 summary
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    for ds, df, mk in [('dandi', dfd, 'o'), ('wagenaar', dfw, '^')]:
        g = df[(~df['excluded_empty']) & (df['n_bursts_ign'] >= 20)]
        axs[0].scatter(g['ign_conc_null'], g['ign_conc'], s=8, marker=mk, alpha=0.6, label=ds)
        axs[1].scatter(g['rand_radius_um'], g['ign_radius_um'], s=8, marker=mk, alpha=0.6, label=ds)
    for ax in axs[:2]:
        lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lim, lim, 'k:', lw=0.6)
        ax.legend(frameon=False)
    axs[0].set_xlabel('top-10% concentration, null')
    axs[0].set_ylabel('top-10% concentration, observed')
    axs[1].set_xlabel('rms radius of K random burst spikes (um)')
    axs[1].set_ylabel('rms radius of first K spikes (um)')
    cols = ['rho_bin_out', 'rho_bin_in', 'rho_bin_dens', 'rho_bin_lat', 'rho_ch_out', 'rho_ch_in', 'rho_ch_dens',
            'rho_ch_lat', 'prho_ch_out_given_in']
    g = dfd[(~dfd['excluded_empty']) & (dfd['n_bursts_ign'] >= 20)]
    axs[2].boxplot([g[c].dropna() for c in cols], showfliers=False)
    axs[2].set_xticks(range(1, len(cols) + 1), [c.replace('rho_', '').replace('_given_', '|') for c in cols], rotation=60)
    axs[2].axhline(0, c='k', lw=0.5)
    axs[2].set_ylabel('Spearman rho with ignition (DANDI sessions)')
    fig.tight_layout()
    fig.savefig(os.path.join(DIAG, 'ignition_concentration_predictors.png'), dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    main()
