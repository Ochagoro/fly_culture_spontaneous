"""Spontaneous activity in the FlyWire whole-brain LIF model.

Extends the event-driven GPU simulator of the Shiu et al. (2024) model
(sim.py, validated against Brian 2 at r = 1.000) with what is needed to study
spontaneous dynamics:

* every neuron receives an independent Poisson drive of rate ``h`` (Hz) of
  suprathreshold kicks: a "spontaneous spike" in the sense of a driven
  branching process. Refractoriness is kept for all neurons (unlike the
  stimulated neurons of the original model).
* a per-trial gain ``gain[b]`` multiplies every synaptic weight, so one batch
  sweeps the distance to the critical point.
* optional Tsodyks-Markram short-term depression (``std_U``, ``std_tau``) of
  each neuron's outputs, which lets runaway activity terminate the way
  network bursts terminate in dissociated cultures.
* population activity is recorded in time bins for the whole brain and for
  neuron groups (e.g. cell classes or neuropils).
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np
import torch

from sim import PARAMS, Connectome, load_connectome  # noqa: F401


class SpontSim:
    def __init__(self, con: Connectome, params: dict | None = None, device: str = "cuda"):
        self.p = dict(PARAMS)
        if params:
            self.p.update(params)
        p = self.p
        self.dev = torch.device(device)
        self.n = con.n
        w = con.count.astype(np.float32) * con.sign[con.pre].astype(np.float32) * p["w_syn"]
        keep = w != 0
        pre = torch.as_tensor(con.pre[keep], device=self.dev)
        post = torch.as_tensor(con.post[keep], device=self.dev)
        w = torch.as_tensor(w[keep], device=self.dev)
        order = torch.argsort(pre, stable=True)
        self.rows = post[order].contiguous()
        self.vals = w[order].contiguous()
        deg = torch.bincount(pre, minlength=self.n)
        self.col_ptr = torch.zeros(self.n + 1, dtype=torch.int64, device=self.dev)
        self.col_ptr[1:] = torch.cumsum(deg, 0)
        dt = p["dt"]
        self.eg = math.exp(-dt / p["tau"])
        self.em = math.exp(-dt / p["t_mbr"])
        self.cgv = (p["tau"] / (p["tau"] - p["t_mbr"])) * (self.eg - self.em)
        self.delay = int(round(p["t_dly"] / dt))
        self.n_rfc = int(round(p["t_rfc"] / dt))

    @torch.no_grad()
    def run(self, gain, h: float | np.ndarray = 1.0, t_run: float = 2000.0, seed: int = 0,
            bin_ms: float = 1.0, groups: np.ndarray | None = None, n_groups: int = 0,
            std_U: float = 0.0, std_tau: float = 800.0, drive_mask: np.ndarray | None = None,
            stim: list | None = None, record_first: bool = False, record_events=None,
            count_from_ms: float = 0.0, grp_bin_ms: float | None = None,
            silence: list | None = None):
        """Run len(gain) trials in parallel.

        gain: synaptic gain per trial (multiplies all weights).
        h: spontaneous drive per neuron (Hz); scalar or per-trial array.
        groups: int label per neuron (0..n_groups-1) for grouped population activity.
        drive_mask: bool per neuron; only these receive the spontaneous drive.
        stim: optional per-trial list of (indices, t_on_ms, t_off_ms, rate_Hz) extra drive.
        record_events: optional bool mask of neurons whose spike events are kept.
        Returns dict with counts (B x N), pop (B x n_bins), grp (B x n_groups x n_bins).
        """
        p, dev, N = self.p, self.dev, self.n
        gain = torch.as_tensor(np.asarray(gain, np.float32), device=dev)
        B = gain.numel()
        dt = p["dt"]
        n_steps = int(round(t_run / dt))
        spb = int(round(bin_ms / dt))
        n_bins = n_steps // spb
        gspb = int(round((grp_bin_ms or bin_ms) / dt))
        n_gbins = n_steps // gspb
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed))
        h = np.broadcast_to(np.asarray(h, np.float32), (B,))
        ph = torch.as_tensor(h * dt * 1e-3, device=dev)[:, None]          # B x 1
        dmask = None
        if drive_mask is not None:
            dmask = torch.as_tensor(np.asarray(drive_mask, bool), device=dev)[None, :]

        v = torch.full((B, N), p["v_0"], device=dev)
        g = torch.zeros((B, N), device=dev)
        refr = torch.zeros((B, N), dtype=torch.int16, device=dev)
        counts = torch.zeros((B, N), dtype=torch.int32, device=dev)
        pop = torch.zeros((B, n_bins), dtype=torch.int32, device=dev)
        grp = None
        if groups is not None:
            gl = torch.as_tensor(np.asarray(groups, np.int64), device=dev)
            grp = torch.zeros((B * n_groups * n_gbins,), dtype=torch.int32, device=dev)
        first = torch.full((B, N), -1.0, device=dev) if record_first else None
        ev_mask = None
        events = []
        if record_events is not None:
            ev_mask = torch.as_tensor(np.asarray(record_events, bool), device=dev)
        use_std = std_U > 0
        if use_std:
            x = torch.ones((B, N), device=dev)
            erec = math.exp(-dt / std_tau)

        sil = None
        if silence is not None and any(len(x) for x in silence):
            sil = torch.zeros((B, N), dtype=torch.bool, device=dev)
            for b, idx in enumerate(silence):
                if len(idx):
                    sil[b, torch.as_tensor(np.asarray(idx, np.int64), device=dev)] = True
            sil = sil.view(-1)
        stim_list = []
        if stim:
            for b, items in enumerate(stim):
                for idx, t0, t1, rate in items:
                    stim_list.append((b, torch.as_tensor(np.asarray(idx, np.int64), device=dev),
                                      int(t0 / dt), int(t1 / dt), rate * dt * 1e-3))

        queue: deque = deque([(torch.zeros(0, dtype=torch.int64, device=dev),
                               torch.zeros(0, device=dev))] * self.delay)
        col_ptr, rows, vals = self.col_ptr, self.rows, self.vals
        v0, vth, vrst = p["v_0"], p["v_th"], p["v_rst"]
        c0 = int(round(count_from_ms / dt))

        for t in range(n_steps):
            active = refr <= 0
            v_new = v0 + (v - v0) * self.em + self.cgv * g
            g_new = g * self.eg
            v = torch.where(active, v_new, v)
            g = torch.where(active, g_new, g)
            refr = refr - 1
            # spontaneous drive: suprathreshold kick
            kick = torch.rand((B, N), device=dev, generator=gen) < ph
            if dmask is not None:
                kick &= dmask
            for b, idx, t0, t1, pr in stim_list:
                if t0 <= t < t1:
                    hit = torch.rand(idx.numel(), device=dev, generator=gen) < pr
                    kick[b, idx[hit]] = True
            spk = ((v > vth) | kick) & active
            arr, eff = queue.popleft()
            if arr.numel():
                b_idx = torch.div(arr, N, rounding_mode="floor")
                j = arr - b_idx * N
                start = col_ptr[j]
                length = col_ptr[j + 1] - start
                total = int(length.sum())
                if total:
                    seg = torch.repeat_interleave(torch.arange(arr.numel(), device=dev), length,
                                                  output_size=total)
                    offs = torch.arange(total, device=dev) - torch.repeat_interleave(
                        torch.cumsum(length, 0) - length, length, output_size=total)
                    e = start[seg] + offs
                    scale = gain[b_idx] * eff
                    g.view(-1).index_add_(0, b_idx[seg] * N + rows[e], vals[e] * scale[seg])
                g = torch.where(active, g, torch.zeros_like(g))
            v = torch.where(spk, torch.full_like(v, vrst), v)
            g = torch.where(spk, torch.zeros_like(g), g)
            refr = torch.where(spk, torch.full_like(refr, self.n_rfc), refr)
            if t >= c0:
                counts += spk.to(torch.int32)
            sflat = torch.nonzero(spk.view(-1)).squeeze(1)
            if use_std:
                x = 1 - (1 - x) * erec
                eff_out = x.view(-1)[sflat].clone()
                xv = x.view(-1)
                xv[sflat] = xv[sflat] * (1 - std_U)
            else:
                eff_out = torch.ones(sflat.numel(), device=dev)
            bi = t // spb
            gbi = t // gspb
            if bi < n_bins and sflat.numel():
                bb = torch.div(sflat, N, rounding_mode="floor")
                pop[:, bi] += torch.bincount(bb, minlength=B).to(torch.int32)
                if grp is not None and gbi < n_gbins:
                    jj = sflat - bb * N
                    grp.index_add_(0, (bb * n_groups + gl[jj]) * n_gbins + gbi,
                                   torch.ones_like(jj, dtype=torch.int32))
                if first is not None:
                    fv = first.view(-1)
                    newf = fv[sflat] < 0
                    fv[sflat[newf]] = t * dt
                if ev_mask is not None:
                    jj = sflat - bb * N
                    kept = sflat[ev_mask[jj]]
                    events.append((torch.full_like(kept, t), kept))
            if sil is not None and sflat.numel():
                ok = ~sil[sflat]
                sflat, eff_out = sflat[ok], eff_out[ok]
            queue.append((sflat, eff_out))

        out = dict(counts=counts.cpu().numpy(), pop=pop.cpu().numpy(), t_run=t_run,
                   bin_ms=bin_ms, gain=gain.cpu().numpy(), h=h)
        if grp is not None:
            out["grp"] = grp.view(B, n_groups, n_gbins).cpu().numpy()
            out["grp_bin_ms"] = grp_bin_ms or bin_ms
        if first is not None:
            out["first"] = first.cpu().numpy()
        if events:
            et = torch.cat([e[0] for e in events]).cpu().numpy()
            en = torch.cat([e[1] for e in events]).cpu().numpy()
            out["ev_t"], out["ev_b"], out["ev_n"] = et.astype(np.int32), en // N, en % N
        return out


# ---------------------------------------------------------------- analysis

def mr_estimate(a: np.ndarray, kmax: int = 40, k0: int = 1):
    """Multistep-regression estimate of the branching ratio (Wilting & Priesemann 2018).

    a: 1-D activity time series (counts per bin). Returns (m, r_k, b) where r_k are
    the regression slopes of a[t+k] on a[t] for k = k0..kmax and r_k = b * m**k.
    """
    a = np.asarray(a, np.float64)
    ks = np.arange(k0, kmax + 1)
    rk = np.empty(len(ks))
    for i, k in enumerate(ks):
        x, y = a[:-k], a[k:]
        vx = x.var()
        rk[i] = np.nan if vx == 0 else ((x - x.mean()) * (y - y.mean())).mean() / vx
    ok = rk > 0
    if ok.sum() < 3:
        return np.nan, rk, np.nan
    from scipy.optimize import curve_fit
    try:
        (b, m), _ = curve_fit(lambda k, b, m: b * m ** k, ks[ok], rk[ok],
                              p0=(rk[ok][0], 0.9), bounds=([0, 0], [10, 1.5]), maxfev=20000)
    except Exception:
        return np.nan, rk, np.nan
    return m, rk, b
