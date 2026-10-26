"""Scan DANDI 002020 sessions: spike counts, duration, mapping coverage (no metrics)."""
import sys, glob, os, h5py, numpy as np
root = sys.argv[1]
rows = []
for fn in sorted(glob.glob(os.path.join(root, 'sub-*', '*.nwb'))):
    with h5py.File(fn, 'r') as f:
        g = f['processing/ecephys/detected_spikes']
        n = g['time_from_first_spike_seconds'].shape[0]
        tmax = g['time_from_first_spike_seconds'][-1] if n else 0
        m = f['processing/ecephys/source_mapping']
        el = m['source_electrode'][:]; x = m['source_x'][:]; y = m['source_y'][:]
        ok = (el >= 0) & np.isfinite(x) & np.isfinite(y)
        dur = f['processing/ecephys/conversion_provenance/duration_s'][0]
        print(os.path.basename(fn), n, round(float(tmax), 1), round(float(dur), 1), ok.sum(), len(el), x[ok].min(), x[ok].max(), y[ok].min(), y[ok].max())
