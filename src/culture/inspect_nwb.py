import sys, h5py, numpy as np
f = h5py.File(sys.argv[1], 'r')
def show(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(name, obj.shape, obj.dtype)
    else:
        print(name + '/', dict(obj.attrs) if len(obj.attrs) < 6 else list(obj.attrs))
f.visititems(show)
g = f['processing/ecephys/detected_spikes']
t = g['time_from_first_spike_seconds'][:]
c = g['source_channel'][:]
a = g['amplitude_native'][:]
print('n', len(t), 't range', t.min(), t.max(), 'sorted', np.all(np.diff(t) >= 0))
print('chan unique', len(np.unique(c)), c[:10], 'amp', np.percentile(a, [1, 50, 99]))
m = f['processing/ecephys/source_mapping']
for k in m: print(k, m[k][:5])
print('nmap', len(m['source_channel']))
print('dt min', np.min(np.diff(np.unique(t))))
for k in ['session_start_time', 'session_description', 'general/subject/age', 'general/subject/description']:
    try: print(k, f[k][()])
    except Exception as e: print(k, 'NA')
