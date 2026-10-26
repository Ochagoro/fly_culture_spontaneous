"""Spontaneous-burst metrics for any (spike time, channel, x, y) list.

Shared definitions for the cultured-neuron half (DANDI 002020 HD-MEA and
Wagenaar et al. 2006 59-electrode MEA) and reusable on a 'virtual MEA' of a
simulated network. All parameters are fixed in PARAMS (set before looking at
cross-session results) and are identical for all sessions; only the spatial
grid differs between array types (it is given by the caller).

C1 network bursts, C2 ignition sites, C3 predictors of ignition, C4 branching
ratio (multistep regression, `mr_estimate` from src/spont.py), C6 halves.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spont import mr_estimate  # noqa: E402  (same estimator as the fly half)

PARAMS = dict(
    active_rate=0.1,      # Hz; channels below this rate are dropped from everything
    burst_bin=0.010,      # s; population spike count bin for burst detection
    thr_mult=5.0,         # theta_hi = max(thr_mult*median(count), floor_frac*N_active, floor_min)
    floor_frac=0.10,      # 10 % of active channels spiking in one 10-ms bin
    floor_min=5.0,        # absolute floor (spikes / 10 ms) for small arrays
    lo_frac=0.5,          # burst edges: contiguous bins >= lo_frac*theta_hi around a theta_hi bin
    merge_gap=0.100,      # s; bursts separated by < 100 ms are merged
    min_part=0.20,        # a network burst must recruit >= 20 % of active channels (added after QC of 4 traces)
    K=20,                 # ignition = centroid of the first K spikes of a burst
    ign_window=0.020,     # s; alternative ignition = centroid of spikes in first 20 ms
    min_bursts=20,        # sessions with fewer bursts are excluded from C2/C3
    sigma_um=100.0,       # Gaussian smoothing of ignition maps (micrometres)
    dens_radius=200.0,    # um; local activity density radius (C3 iii)
    min_lat_bursts=5,     # channel latency needs >= 5 bursts with a spike
    n_null=50,            # null repetitions
    mr_bin=0.004,         # s; bin for branching-ratio estimate
    kmax=50,              # 50 x 4 ms = 200 ms
    excl_pad=0.100,       # s; inter-burst segments exclude bursts +- 100 ms
    min_seg=1.0,          # s; inter-burst segments shorter than this are dropped
    seed=0,
)


# ----------------------------------------------------------------------------- helpers
def runs(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def m_tau(a, kmax, dt):
    if len(a) < 5 * kmax or a.var() == 0:
        return np.nan, np.nan
    m = mr_estimate(a, kmax, 1)[0]
    if not np.isfinite(m):
        return np.nan, np.nan
    tau = -dt / np.log(m) if 0 < m < 1 else np.inf
    return float(m), float(tau * 1000)  # tau in ms


def spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return np.nan
    return float(stats.spearmanr(a[ok], b[ok])[0])


def partial_spearman(a, b, c):
    """Spearman correlation of a and b controlling for c (rank residuals)."""
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    if ok.sum() < 10:
        return np.nan
    ra, rb, rc = (stats.rankdata(v[ok]) for v in (a, b, c))
    X = np.c_[np.ones(ok.sum()), rc]
    res = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]  # noqa: E731
    ea, eb = res(ra), res(rb)
    if ea.std() == 0 or eb.std() == 0:
        return np.nan
    return float(np.corrcoef(ea, eb)[0, 1])


def entropy_conc(bins, n_support, nbins_total):
    """Normalised entropy and top-10 % concentration of a list of bin indices."""
    cnt = np.bincount(bins, minlength=nbins_total).astype(float)
    p = cnt[cnt > 0] / cnt.sum()
    H = float(-(p * np.log(p)).sum() / np.log(n_support))
    top = int(np.ceil(0.1 * n_support))
    conc = float(np.sort(cnt)[::-1][:top].sum() / cnt.sum())
    return H, conc


# ----------------------------------------------------------------------------- bursts
def detect_bursts(t, T, n_active, P):
    dt = P['burst_bin']
    nb = int(np.ceil(T / dt)) + 1
    c = np.bincount((t / dt).astype(np.int64), minlength=nb).astype(float)
    med = float(np.median(c))
    th_hi = max(P['thr_mult'] * med, P['floor_frac'] * n_active, P['floor_min'])
    th_lo = P['lo_frac'] * th_hi
    s, e = runs(c >= th_lo)
    H = np.r_[0, np.cumsum(c >= th_hi)]
    keep = (H[e] - H[s]) > 0
    s, e = s[keep], e[keep]
    if len(s) > 1:
        mg = int(round(P['merge_gap'] / dt))
        new = np.r_[True, (s[1:] - e[:-1]) >= mg]
        gid = np.cumsum(new) - 1
        s = s[new]
        e = np.maximum.reduceat(e, np.flatnonzero(new))
        assert len(s) == gid.max() + 1
    return s * dt, e * dt, dict(median_count=med, th_hi=th_hi, th_lo=th_lo), c


def assign(t, starts, ends):
    idx = np.searchsorted(starts, t, side='right') - 1
    inb = idx >= 0
    inb[inb] = t[inb] < ends[idx[inb]]
    idx[~inb] = -1
    return idx


# ----------------------------------------------------------------------------- main
def analyze(t, ch, cx, cy, T, grid, P=PARAMS, detail=False):
    """t (s, sorted), ch (index into cx, cy), cx/cy (um), T duration (s).

    grid = dict(x0, y0, w, nx, ny) defines the fixed spatial grid (bin width w um).
    Returns (row dict of scalar metrics, extras dict of arrays).
    """
    rng = np.random.default_rng(P['seed'])
    C = len(cx)
    out = dict(T=T, n_spikes_raw=len(t), n_channels=C)
    rate = np.bincount(ch, minlength=C) / T
    act = rate >= P['active_rate']
    n_act = int(act.sum())
    keep = act[ch]
    t, ch = t[keep], ch[keep]
    out.update(n_active=n_act, n_spikes=len(t), mean_rate_active=float(rate[act].mean()) if n_act else np.nan)
    ex = {}
    if n_act < 5 or len(t) < 100:
        return out, ex

    # ---------------- C1
    st, en, thr, c10 = detect_bursts(t, T, n_act, P)
    out.update(thr)
    bid = assign(t, st, en)
    inb = bid >= 0
    if len(st):  # drop events that recruit < min_part of active channels (single-channel bursts etc.)
        key = np.unique(bid[inb].astype(np.int64) * C + ch[inb])
        part = np.bincount(key // C, minlength=len(st)) / n_act
        big = part >= P['min_part']
        out['n_small_events'] = int((~big).sum())
        st, en = st[big], en[big]
        bid = assign(t, st, en)
        inb = bid >= 0
    nb = len(st)
    dur = en - st
    out.update(n_bursts=nb, burst_rate_per_min=nb / (T / 60), frac_out_of_burst=float(1 - inb.mean()),
               median_burst_dur_ms=float(np.median(dur) * 1000) if nb else np.nan,
               frac_time_in_burst=float(dur.sum() / T))
    if nb:
        key = np.unique(bid[inb].astype(np.int64) * C + ch[inb])
        part = np.bincount(key // C, minlength=nb) / n_act
        nsp = np.bincount(bid[inb], minlength=nb)
        out.update(median_participation=float(np.median(part)), median_burst_spikes=float(np.median(nsp)))
        ibi = st[1:] - en[:-1]
        out.update(median_ibi_s=float(np.median(ibi)) if nb > 1 else np.nan,
                   cv_ibi=float(ibi.std() / ibi.mean()) if nb > 2 else np.nan)
    ex.update(burst_start=st, burst_end=en)

    # ---------------- C4 (whole session and inter-burst)
    dt4 = P['mr_bin']
    n4 = int(np.ceil(T / dt4)) + 1
    a4 = np.bincount((t / dt4).astype(np.int64), minlength=n4).astype(float)
    pad = P['excl_pad']
    d = np.zeros(n4 + 1, int)
    if nb:
        i0 = np.clip(np.floor((st - pad) / dt4).astype(int), 0, n4)
        i1 = np.clip(np.ceil((en + pad) / dt4).astype(int), 0, n4)
        np.add.at(d, i0, 1)
        np.add.at(d, i1, -1)
    excl = np.cumsum(d)[:n4] > 0
    s_, e_ = runs(~excl)
    ok = (e_ - s_) * dt4 >= P['min_seg']
    s_, e_ = s_[ok], e_[ok]
    out['ib_duration_s'] = float(((e_ - s_) * dt4).sum())

    def ib_series(lo, hi):
        pieces = [a4[max(a, lo):min(b, hi)] for a, b in zip(s_, e_) if min(b, hi) - max(a, lo) >= P['min_seg'] / dt4]
        return np.concatenate(pieces) if pieces else np.zeros(0)

    out['m_all'], out['tau_all_ms'] = m_tau(a4, P['kmax'], dt4)
    aib = ib_series(0, n4)
    out['m_ib'], out['tau_ib_ms'] = m_tau(aib, P['kmax'], dt4)
    out['m_ib_k25'] = m_tau(aib, P['kmax'] // 2, dt4)[0]
    out['m_ib_k100'] = m_tau(aib, P['kmax'] * 2, dt4)[0]
    out['m_all_k25'] = m_tau(a4, P['kmax'] // 2, dt4)[0]
    out['m_all_k100'] = m_tau(a4, P['kmax'] * 2, dt4)[0]
    out['ib_rate_hz_per_ch'] = float(aib.sum() / max(len(aib) * dt4, 1e-9) / n_act) if len(aib) else np.nan

    # ---------------- C6 halves (same burst windows, thresholds from whole session)
    h = T / 2
    for j, (lo, hi) in enumerate([(0, h), (h, T)]):
        sel = (t >= lo) & (t < hi)
        out[f'h{j+1}_burst_rate'] = float(((st >= lo) & (st < hi)).sum() / ((hi - lo) / 60))
        out[f'h{j+1}_frac_out'] = float(1 - inb[sel].mean()) if sel.any() else np.nan
        l4, h4 = int(lo / dt4), int(hi / dt4)
        out[f'h{j+1}_m_all'] = m_tau(a4[l4:h4], P['kmax'], dt4)[0]
        out[f'h{j+1}_m_ib'] = m_tau(ib_series(l4, h4), P['kmax'], dt4)[0]
        out[f'h{j+1}_rate'] = float(sel.sum() / (hi - lo) / n_act)

    if detail:
        ex['a4'] = a4
        ex['excl4'] = excl
        ex['c10'] = c10

    # ---------------- C2 ignition
    K = P['K']
    nsp = np.bincount(bid[inb], minlength=nb) if nb else np.zeros(0, int)
    first = np.searchsorted(t, st)
    okb = nsp >= K
    out['n_bursts_ign'] = int(okb.sum())
    w, x0, y0, nx, ny = grid['w'], grid['x0'], grid['y0'], grid['nx'], grid['ny']

    def tobin(x, y):
        ix = np.clip(((x - x0) // w).astype(int), 0, nx - 1)
        iy = np.clip(((y - y0) // w).astype(int), 0, ny - 1)
        return ix * ny + iy

    chbin = tobin(cx, cy)
    support = np.zeros(nx * ny, bool)
    support[chbin[act]] = True
    n_sup = int(support.sum())
    out['n_support_bins'] = n_sup
    ex['support'] = support.reshape(nx, ny)
    if okb.sum() < P['min_bursts']:
        return out, ex
    fb = first[okb]
    idx = fb[:, None] + np.arange(K)[None, :]
    px, py = cx[ch[idx]], cy[ch[idx]]
    gx, gy = px.mean(1), py.mean(1)
    # alternative: spikes in first ign_window after onset
    j1 = np.searchsorted(t, st[okb] + P['ign_window'])
    csx = np.r_[0, np.cumsum(cx[ch])]
    csy = np.r_[0, np.cumsum(cy[ch])]
    nwin = np.maximum(j1 - fb, 1)
    wx, wy = (csx[fb + nwin] - csx[fb]) / nwin, (csy[fb + nwin] - csy[fb]) / nwin
    earliest = ch[fb]
    ex.update(ign_x=gx, ign_y=gy, ign_win_x=wx, ign_win_y=wy, ign_burst_idx=np.flatnonzero(okb),
              earliest_ch=earliest)
    nbi = len(fb)
    ibins = tobin(gx, gy)
    H, conc = entropy_conc(ibins, n_sup, nx * ny)
    Hw, concw = entropy_conc(tobin(wx, wy), n_sup, nx * ny)
    out.update(ign_entropy=H, ign_conc=conc, ign_entropy_win=Hw, ign_conc_win=concw)
    # null P: ignition points drawn from spatial distribution of all burst spikes
    bsp = np.flatnonzero(inb)
    Hn, Cn = [], []
    for _ in range(P['n_null']):
        r = ch[bsp[rng.integers(0, len(bsp), nbi)]]
        hh, cc = entropy_conc(chbin[r], n_sup, nx * ny)
        Hn.append(hh)
        Cn.append(cc)
    Hn, Cn = np.array(Hn), np.array(Cn)
    out.update(ign_entropy_null=float(Hn.mean()), ign_conc_null=float(Cn.mean()),
               ign_entropy_z=float((H - Hn.mean()) / (Hn.std() + 1e-12)),
               ign_conc_z=float((conc - Cn.mean()) / (Cn.std() + 1e-12)),
               ign_conc_p=float((np.sum(Cn >= conc) + 1) / (len(Cn) + 1)))
    # earliest channel concentration (top 10 % of active channels) vs pool null
    topc = int(np.ceil(0.1 * n_act))
    ce = np.bincount(earliest, minlength=C)
    out['earliest_conc'] = float(np.sort(ce)[::-1][:topc].sum() / nbi)
    en_ = []
    for _ in range(P['n_null']):
        r = np.bincount(ch[bsp[rng.integers(0, len(bsp), nbi)]], minlength=C)
        en_.append(np.sort(r)[::-1][:topc].sum() / nbi)
    out['earliest_conc_null'] = float(np.mean(en_))
    # spread of ignition points (rms distance from their mean), vs null
    spread = lambda X, Y: float(np.sqrt(((X - X.mean()) ** 2 + (Y - Y.mean()) ** 2).mean()))  # noqa: E731
    out['ign_spread_um'] = spread(gx, gy)
    r = ch[bsp[rng.integers(0, len(bsp), nbi)]]
    out['ign_spread_null_um'] = spread(cx[r], cy[r])
    # compactness: rms radius of first K spikes vs K random spikes of the same burst
    rad = np.sqrt(((px - gx[:, None]) ** 2 + (py - gy[:, None]) ** 2).mean(1))
    nk = nsp[okb]
    ridx = fb[:, None] + (rng.random((nbi, K)) * nk[:, None]).astype(int)
    qx, qy = cx[ch[ridx]], cy[ch[ridx]]
    rrad = np.sqrt(((qx - qx.mean(1, keepdims=True)) ** 2 + (qy - qy.mean(1, keepdims=True)) ** 2).mean(1))
    out['ign_radius_um'] = float(np.median(rad))
    out['rand_radius_um'] = float(np.median(rrad))
    out['compactness_ratio'] = float(np.median(rad / np.maximum(rrad, 1e-9)))
    out['frac_ign_dist_lt200'] = float(np.mean(np.hypot(gx - qx.mean(1), gy - qy.mean(1)) < 200))

    # maps
    cnt = np.bincount(ibins, minlength=nx * ny).astype(float).reshape(nx, ny)
    sig = P['sigma_um'] / w
    pmap = ndimage.gaussian_filter(cnt / nbi, sig, mode='constant')
    bcnt = np.bincount(chbin[ch[bsp]], minlength=nx * ny).astype(float).reshape(nx, ny)
    amap = ndimage.gaussian_filter(bcnt / bcnt.sum(), sig, mode='constant')  # burst-spike density
    ex.update(ign_count_map=cnt, ign_pmap=pmap, burst_density_map=amap)
    hot = np.unravel_index(np.argmax(pmap), pmap.shape)
    out['hot_x'], out['hot_y'] = x0 + (hot[0] + 0.5) * w, y0 + (hot[1] + 0.5) * w

    # ---------------- C3 predictors (per spatial bin and per channel)
    T_in = float(dur.sum())
    T_out = T - T_in
    n_out = np.bincount(ch[~inb], minlength=C)
    n_in = np.bincount(ch[inb], minlength=C)
    r_out, r_in = n_out / T_out, n_in / max(T_in, 1e-9)
    # recruitment latency: first spike of each channel in each burst, relative to onset
    bkey = bid[inb].astype(np.int64) * C + ch[inb]
    _, fi = np.unique(bkey, return_index=True)
    fsp = bsp[fi]
    lat = t[fsp] - st[bid[fsp]]
    lch = ch[fsp]
    order = np.lexsort((lat, lch))
    lch_s, lat_s = lch[order], lat[order]
    bounds = np.r_[0, np.flatnonzero(np.diff(lch_s)) + 1, len(lch_s)]
    med_lat = np.full(C, np.nan)
    for a_, b_ in zip(bounds[:-1], bounds[1:]):
        if b_ - a_ >= P['min_lat_bursts']:
            med_lat[lch_s[a_]] = np.median(lat_s[a_:b_])
    okl = np.isfinite(med_lat)
    lrank = np.full(C, np.nan)
    lrank[okl] = (stats.rankdata(med_lat[okl]) - 1) / max(okl.sum() - 1, 1)
    # local density: summed out-of-burst rate of active channels within radius
    ac = np.flatnonzero(act)
    D = np.hypot(cx[ac, None] - cx[None, ac], cy[ac, None] - cy[None, ac])
    dens_ch = np.full(C, np.nan)
    dens_ch[ac] = (D <= P['dens_radius'] + 1e-6) @ r_out[ac]
    # per bin
    sb = np.flatnonzero(support)
    nch_b = np.bincount(chbin[ac], minlength=nx * ny)[sb]
    bmean = lambda v: np.bincount(chbin[ac], weights=v[ac], minlength=nx * ny)[sb] / nch_b  # noqa: E731
    out_b, in_b = bmean(r_out), bmean(r_in)
    lat_b = np.bincount(chbin[ac], weights=np.nan_to_num(lrank[ac]), minlength=nx * ny)[sb]
    lat_n = np.bincount(chbin[ac], weights=okl[ac].astype(float), minlength=nx * ny)[sb]
    lat_b = np.where(lat_n > 0, lat_b / np.maximum(lat_n, 1), np.nan)
    bxc = x0 + (sb // ny + 0.5) * w
    byc = y0 + (sb % ny + 0.5) * w
    Db = np.hypot(bxc[:, None] - cx[None, ac], byc[:, None] - cy[None, ac])
    dens_b = (Db <= P['dens_radius'] + 1e-6) @ r_out[ac]
    ign_b = pmap.ravel()[sb]
    out.update(rho_bin_out=spearman(ign_b, out_b), rho_bin_in=spearman(ign_b, in_b),
               rho_bin_dens=spearman(ign_b, dens_b), rho_bin_lat=spearman(ign_b, lat_b),
               prho_bin_out_given_in=partial_spearman(ign_b, out_b, in_b),
               prho_bin_dens_given_in=partial_spearman(ign_b, dens_b, in_b))
    # per channel: share of first-K spikes vs shares of out-of-burst / in-burst spikes
    fk = np.bincount(ch[idx].ravel(), minlength=C)[ac] / (K * nbi)
    pre = np.clip(fb[:, None] - np.arange(1, K + 1)[None, :], 0, None)
    pk = np.bincount(ch[pre].ravel(), minlength=C)[ac] / (K * nbi)
    so, si = r_out[ac], r_in[ac]
    out.update(rho_ch_out=spearman(fk, so), rho_ch_in=spearman(fk, si), rho_ch_dens=spearman(fk, dens_ch[ac]),
               rho_ch_lat=spearman(fk, lrank[ac]), prho_ch_out_given_in=partial_spearman(fk, so, si),
               prho_ch_dens_given_in=partial_spearman(fk, dens_ch[ac], si),
               rho_ch_pre=spearman(fk, pk),
               # enrichment of first-K share over in-burst share vs out/in ratio
               rho_ch_enrich_outin=spearman(np.log((fk + 1e-4) / (si / si.sum() + 1e-4)),
                                            np.log((so + 1e-3) / (si + 1e-3))))
    ex.update(bin_index=sb, bin_ign=ign_b, bin_out=out_b, bin_in=in_b, bin_dens=dens_b, bin_lat=lat_b,
              ch_first_share=fk, ch_out=so, ch_in=si, ch_lat=lrank[ac], ch_active=ac)
    return out, ex
