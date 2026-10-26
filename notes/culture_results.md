# Cultured-neuron half: where do spontaneous network bursts ignite, and how close to runaway?

(2026-10-27. Code: `src/culture/` — `metrics.py` (all definitions, one function `analyze(t, ch, x, y, T, grid)` usable on any spike list incl. a virtual MEA), `run_sessions.py` (loaders, runs), `summarize.py` (cross-session statistics), `figures.py`. Outputs: `results/culture/`.)

## 0. Fixed parameters (set before any cross-session result was seen; identical for all sessions)

| step | parameter | value |
|---|---|---|
| channels | active channel | rate ≥ 0.1 Hz over the session; all other channels/spikes dropped everywhere |
| exclusion | "nearly empty" session | total spikes < 30 000 × (n channels / 1000) × (T / 1800 s) (= < 30 k spikes for a ~1000-channel 30-min DANDI session, < ~1.8 k for a 59-electrode 30-min recording) |
| C1 | bin | 10 ms population spike count c(t) |
| C1 | high threshold θ_hi | max(5 × median c, 0.10 × N_active, 5) |
| C1 | burst extent | maximal run of bins ≥ θ_lo = 0.5 θ_hi containing ≥1 bin ≥ θ_hi; runs separated by < 100 ms merged |
| C1 | minimum recruitment | ≥ 20 % of active channels spike in the event (else it is a "small event", counted as out-of-burst) |
| C2 | ignition point | centroid (x, y) of the first K = 20 spikes at/after burst onset (spike-count weighted); also recorded: centroid of spikes in first 20 ms, earliest channel |
| C2 | bursts used | bursts with ≥ K spikes; session needs ≥ 20 such bursts for C2/C3 |
| C2 | grid | DANDI 100 µm (39 × 21 bins over 3900 × 2100 µm); Wagenaar 200 µm bins centred on electrodes (8 × 8) |
| C2 | support | bins containing ≥1 active channel (n_support) |
| C2 | concentration | fraction of bursts whose ignition point falls in the ceil(0.1 n_support) most frequent bins |
| C2 | entropy | −Σp log p / log n_support over bins |
| C2 | null (spec) | n_bursts ignition points drawn as single spikes from all burst spikes of the session (spatial distribution of burst activity), 50 repetitions |
| C2 | compactness | rms radius of the first K spikes / rms radius of K random spikes of the same burst (median over bursts) — tests that the first spikes are spatially local |
| C2 | maps | ignition count map / n_bursts, Gaussian σ = 100 µm |
| C2b | stability | consecutive analysable sessions of a culture: Pearson r of smoothed ignition maps over bins in both supports; distance between map maxima (hot spots); r of the residual (ignition map − burst-spike-density map); null = all between-culture session pairs |
| C3 | predictors | per support bin: smoothed ignition probability vs (i) mean out-of-burst rate per channel, (ii) mean in-burst rate per channel, (iii) summed out-of-burst rate of active channels within 200 µm of bin centre, (iv) mean normalised rank of channel median recruitment latency (channels with ≥5 bursts). Also per channel: share of the first-K spikes vs the same predictors; partial Spearman (out | in) |
| C4 | branching ratio | `mr_estimate` from `src/spont.py` (identical estimator to the fly half), 4-ms bins, k = 1…50 (200 ms), τ = −4 ms / ln m |
| C4 | inter-burst | bins outside [burst start − 100 ms, burst end + 100 ms], segments ≥ 1 s, concatenated |
| C6 | halves | first vs second 15 min (T/2), same burst windows and thresholds as whole session |
| sens. | ×0.5 / ×2 | θ_hi (multiplier, floor fraction and floor together), K (10/40), minimum recruitment (10 %/40 %), kmax (25/100) |

Day axis: DANDI 002020 does not store days in vitro; "day" = days since the culture's first recorded session (including sessions later excluded as empty). Wagenaar: DIV from the file name (batch-culture-DIV).

Wagenaar channel → position: hardware channel numbers 0–59 (Potter lab help page, "hw" numbering); converted to column/row with the lab's `hw2cr.m` tables (channel 14 = column 0, row 4 = the blank cell in the help-page table, i.e. the ground electrode, which carries no spikes); x = 200 µm × column, y = 200 µm × row.

**One change after looking at data (pipeline QC, before any cross-session result):** traces of 4 sessions (2 DANDI, 2 Wagenaar) showed that on 59-electrode arrays with a near-zero median count, θ_hi (≈5–6 spikes/10 ms) was crossed hundreds of times by single-channel bursts that are not network events (Wagenaar 2-1-20: 556 "bursts" with median 6 spikes and 9 % participation). I added the minimum-recruitment rule (≥ 20 % of active channels) and its ×0.5/×2 sensitivity. No other parameter was changed.

## Caveats (apply to everything below)

- **Unsorted channel-level detections.** DANDI spikes are threshold crossings per routed electrode, not neurons; one neuron can appear on several nearby electrodes and one electrode can carry several neurons. All "rates" are channel rates. Wagenaar data are also unsorted (one channel = one 30-µm electrode).
- **DIV unknown in DANDI 002020**: the day axis is days since the culture's first recorded session; cultures may have started recording at different DIV, so day 0 is not comparable across cultures (only within-culture trends are interpreted).
- **Electrode configuration changes between sessions** (~1000 of 26 400 electrodes routed; the selection differs from session to session). Ignition maps of two sessions are compared only on bins covered in both (median Jaccard overlap of covered bins is reported with the stability results). Local density (C3 iii) mixes true activity density with routing density.
- **Ignition point = centroid**: if the first 20 spikes come from two separated sites the centroid falls between them. The compactness ratio and the K = 10/40 and 20-ms-window variants are reported to bound this.
- **Branching ratio on population counts** of ~1000 channels: the MR estimator corrects for subsampling, but bursts (non-stationary, slow ramps) dominate the autocorrelation at 4–200 ms, so m on whole sessions is close to 1 for reasons unrelated to criticality; the inter-burst estimate removes bursts ± 100 ms but concatenation of segments adds junctions.

## 1. Data and n

| | DANDI 002020 (MaxOne HD-MEA) | Wagenaar 2006 dense (59-electrode MEA) |
|---|---|---|
| recordings / cultures | 129 / 7 | 527 / 30 (all analysed, no subsampling) |
| excluded "nearly empty" | 15 (all < 30 k spikes) | 1 |
| included | 114 sessions, 7 cultures | 526 recordings, 30 cultures |
| sessions with 0 bursts | 10 | 75 |
| network bursts (all included sessions) | 30 349 | 111 329 |
| sessions with ≥ 20 ignition-usable bursts (C2/C3) | 70 sessions, 7 cultures, 28 504 bursts | 363 recordings, 30 cultures, 91 670 bursts |
| active channels per session (median) | 426 (IQR 257–741) of ~1000 routed | 56 |

Statistics: per-session values are summarised as median [IQR]; "p" is a Wilcoxon signed-rank test across sessions (sessions of one culture are not independent, so each claim is also given at the culture level: number of cultures whose median goes the same way, with a two-sided sign test). Day = days since the culture's first session (DANDI) or DIV (Wagenaar).

## 2. Results

### C1 Network bursts
| | DANDI | Wagenaar |
|---|---|---|
| burst rate (/min) | 1.6 [0.23–8.2] (n = 113) | 2.4 [0.40–7.1] (n = 526) |
| median burst duration | 390 ms [190–635] | 150 ms [80–465] |
| median participation (fraction of active channels) | 0.94 [0.72–0.99] | 0.47 [0.34–0.83] |
| fraction of spikes outside bursts | 0.53 [0.16–0.92] | 0.81 [0.45–0.95] |
| fraction of time in bursts | 0.010 | 0.012 |

Sensitivity (θ_hi ×0.5 / ×1 / ×2): DANDI burst rate 2.3 / 1.6 / 1.2 per min, out-of-burst fraction 0.36 / 0.53 / 0.74, participation 0.89 / 0.94 / 0.94. Wagenaar burst rate 16.1 / 2.4 / 0.96 per min, participation 0.35 / 0.47 / 0.91. **The out-of-burst fraction and, for Wagenaar, the burst rate depend strongly on the threshold** (Wagenaar dense cultures have many small "burstlets"; median 111 sub-20 %-recruitment events per recording); the development trends below are robust to it.

### C2 Ignition sites
(a) **Locality and concentration (K = 20).**
- DANDI: the first 20 spikes of a burst are spatially local: their rms radius is 0.62× that of 20 random spikes of the same burst (median ratio; < 1 in 69/70 sessions; 621 vs 1076 µm; p = 4e-13). The onset centroid lies a median 0.82 mm from the centroid of a random 20-spike window of the same burst.
- The earliest channel is strongly concentrated: the top 10 % of channels supply the first spike of 81 % of bursts vs 48 % expected from the spatial distribution of burst spikes (91 % of sessions above null, p = 4e-12).
- Ignition-centroid concentration vs the specified null (points drawn from all burst spikes): **no difference** (0.56 vs 0.52, p = 0.26; entropy 0.76 vs 0.77, p = 0.43; 3/7 cultures above null). This null is not a like-for-like comparison (single spikes vs 20-spike centroids, see section 3), so I added a random-window null (centroid of 20 consecutive spikes at a random position in the same burst): observed concentration 0.56 vs 0.48 (higher in 67 % of sessions, p = 1.4e-5; significant at p < 0.05 within 51 % of sessions; but only 5/7 cultures, sign p = 0.45), entropy 0.76 vs 0.81 (p = 9e-6). Onset centroids are also *more* spread than random windows (760 vs 581 µm), i.e. onsets fall on several separate hot spots rather than on one focus.
- Wagenaar (59 electrodes, 1.4 mm): concentration looks very high against the specified null (0.84 vs 0.35, 99 % of sessions, 30/30 cultures) **but this is an averaging artefact**: the random-window null is even more concentrated (0.90; observed < null in 30/30 cultures), and concentration rises with K (0.71 / 0.84 / 0.91 for K = 10 / 20 / 40). Compactness ratio 0.98 (first 20 spikes are not more local than random ones at 200 µm pitch). The earliest channel is moderately concentrated (0.47 vs 0.35, 90 % of sessions, p = 7e-54).

(b) **Stability across consecutive sessions** (DANDI: 63 consecutive pairs, 7 cultures; null = 1674 between-culture pairs).
- Ignition-map correlation r = 0.43 consecutive vs −0.06 between cultures (Mann-Whitney p = 9e-18; higher in 7/7 cultures). Distance between hot spots 0.58 mm vs 1.75 mm (6/7 cultures closer).
- Burst-activity maps are more stable still (r = 0.80 vs −0.03), but the ignition-minus-activity residual map is also stable (r = 0.39 vs 0.02, p = 2e-15, 7/7 cultures): the ignition pattern is not just a copy of where burst spikes are.
- Caveat: consecutive sessions share more of the electrode routing (Jaccard of covered bins 0.49 vs 0.10 between cultures), and ignition probability tracks routing density (C3), so part of the stability may be routing.
- Wagenaar: r = 0.85 consecutive vs 0.48 between (30/30 cultures), hot spots 200 vs 283 µm; the high between-culture value again reflects centroids pulled to the array centre.

### C3 What predicts ignition
DANDI, Spearman ρ per session, median over 70 sessions [IQR], cultures with median > 0:

| predictor | per 100-µm bin vs ignition probability | per channel vs share of the first 20 spikes |
|---|---|---|
| (i) out-of-burst rate | **0.01** [−0.10, 0.10], p = 0.97, 3/7 | 0.46 [0.27, 0.63], 7/7 |
| (ii) in-burst rate | 0.16 [0.04, 0.30], 6/7 | 0.37 [0.18, 0.50], 7/7 |
| (iii) local density (summed out-of-burst rate ≤ 200 µm) | 0.29 [0.10, 0.43], 7/7 | 0.26 [0.16, 0.37], 7/7 |
| (iii) controlled for routing density (partial, given n channels ≤ 200 µm) | **−0.00** [−0.13, 0.13], p = 0.74, 3/7 | — |
| (iv) recruitment-latency rank (sanity check, partly circular) | −0.31, 0/7 | −0.65, 0/7 |
| (v) leading eigenvector of inter-burst coupling | **0.25** [0.03, 0.35], p = 6e-8, 7/7 | 0.30 [0.12, 0.46], 6/7 |
| (v) controlled for routing density (partial) | 0.14, p = 5e-5, 7/7 | 0.28, 6/7 |
| (i) given (ii) (partial) | −0.08 | 0.36 [0.24, 0.52], 7/7 |

- The bin-level "local density" effect is fully explained by how many electrodes were routed nearby (ignition probability vs number of active channels ≤ 200 µm: ρ = 0.40, 7/7 cultures). The plain inter-burst rate of a location does not predict where bursts ignite.
- **(v) Eigenvector (added at the coordinator's request; definition: Pearson correlation of 20-ms counts in inter-burst periods (same exclusion as C4), channels ≥ 30 inter-burst spikes (median 181 channels), negatives → 0, diagonal 0, pairs < 50 µm → 0 against duplicate detections).** Per bin it beats the plain inter-burst rate (computed on the same channels) in 88 % of sessions (56/64; median Δρ = +0.21, p = 3e-9; 7/7 cultures) and survives the routing-density control. Per channel it does not beat the rate (53 % of sessions, Δρ = +0.03, p = 0.41, 3/7 cultures).
- Channel level: channels that fire a lot between bursts contribute more of the first 20 spikes even at a given in-burst rate (partial ρ 0.36, 7/7). This is partly expected for tonic channels (the early part of a burst is sparse), so I checked composition: the top 10 % out-of-burst channels supply 23 % of first-20 spikes, against 13 % of in-burst spikes and 51 % of out-of-burst spikes (first > in-burst share in 7/7 cultures; first < background share in 7/7). The onset is enriched in background-active channels relative to the burst, but it is not simply background activity carrying on. First-20 composition correlates with the composition of the 20 spikes just before onset (ρ = 0.64).
- Wagenaar (363 recordings): bin level, out-of-burst rate 0.19 (27/30), in-burst 0.18, density 0.47 (partial given routing 0.30, 27/30), eigenvector 0.18 (ties with rate: beats it in 50 %). Channel level: out-of-burst rate 0.81, in-burst 0.80, partial out|in 0.55 (30/30), eigenvector 0.68 (rate wins in 69 %, 23/30 cultures). Bin-level Wagenaar maps carry the centroid artefact described above, so the channel-level numbers are the more reliable ones.

### C4 Distance to criticality (MR estimator, 4-ms bins, k ≤ 200 ms)
| | DANDI (n = 113) | Wagenaar (n = 526) |
|---|---|---|
| m whole session | 0.992 [0.985–0.997] | 0.986 [0.974–0.994] |
| m inter-burst | 0.995 [0.969–1.003] | 0.991 [0.972–1.000] |
| m whole − m inter-burst | 0.000, p = 0.99 | 0.000, p = 0.92 |
| τ (median), whole / inter-burst | 458 / 165 ms | 256 / 266 ms |
| sessions with m_ib ≥ 1 | 36 % | 26 % |
| m inter-burst after 2-s detrending (control) | 0.991, τ = 357 ms; ≥ 1 in 21 % | 0.983, τ = 202 ms; ≥ 1 in 10 % |
| kmax 25 / 50 / 100 (m_ib) | 0.996 / 0.995 / 0.994 | 0.982 / 0.991 / 0.993 |

Removing bursts (± 100 ms) does **not** lower m: inter-burst activity is itself near-critical by this estimator (m ≈ 0.99, τ of a few hundred ms), and this survives removal of slow (> 2 s) rate drift. Across sessions, m_ib is lower where bursts are more frequent (ρ = −0.69 DANDI, −0.30 Wagenaar), i.e. frequent bursting goes with **less** reverberating activity between bursts. m_all ≈ 1 in young asynchronous cultures (DANDI days 0–5: 0.997) probably reflects slow rate fluctuations rather than criticality.

### C5 Development (per-culture Spearman ρ vs day/DIV; cultures with ρ > 0 / n; sign test)
| metric | DANDI (7 cultures) | Wagenaar (30 cultures) |
|---|---|---|
| burst rate | ↑ 7/7, median ρ 0.77, p = 0.016 | ↑ 29/30, ρ 0.66, p < 1e-6 |
| fraction of spikes outside bursts | ↓ 0/7, ρ −0.79, p = 0.016 | ↓ 0/30, ρ −0.84 |
| median burst duration | ↓ 1/7, ρ −0.58, p = 0.13 (n.s.) | 10/30, ρ −0.21 (n.s.) |
| participation | 3/7 (n.s.) | 16/30 (n.s.) |
| mean firing rate | 4/7 (n.s.) | ↑ 29/30, ρ 0.85 |
| m whole session | ↓ 0/7, ρ −0.83, p = 0.016 | ↓ 0/30, ρ −0.79 |
| m inter-burst | ↓ 1/7, ρ −0.61 (Wilcoxon p = 0.03) | 10/30, ρ −0.22 (Wilcoxon p = 0.03, sign p = 0.10) |
| m inter-burst, 2-s detrended | ↓ 0/7, ρ −0.43, p = 0.016 | 16/30, ρ 0.04 (null) |
| ignition concentration excess, compactness, spread | no consistent trend (3/6, 1/6, 4/6) | weak (concentration excess 21/30, p = 0.04; others n.s.) |
| channel-level ρ(first-20 share, out-of-burst rate) | ↑ 6/6, p = 0.03 | 18/30 (n.s.) |

With maturation, bursts become more frequent and absorb a growing share of spikes, and the population moves *away* from m = 1 (whole session in both datasets; inter-burst in DANDI, not robustly in Wagenaar).

### C6 Within-session drift (second minus first 15 min)
- DANDI (113 sessions): burst rate 1.73 → 1.53/min (median Δ 0, p = 0.65); out-of-burst fraction Δ 0 (p = 0.49); m_ib Δ −0.0005 (p = 0.25, 2/7 cultures up); m_all Δ −0.0004 (p = 0.01, 0/7 cultures up).
- Wagenaar (526): burst rate Δ −0.04/min (p = 2e-9, 5/30 cultures up); firing rate down (p = 6e-11); m_ib Δ −0.0018 (p = 8e-8, 8/30 cultures up; sign p = 0.016); m_all Δ −0.0002 (p = 2e-4).
- **m does not rise during 30 min of spontaneous activity (no SHY-like build-up); if anything it falls slightly (fatigue-like).** The effects are tiny: Δm ≈ −0.002, or a ~25 % shorter inter-burst τ (466 → 346 ms, Wagenaar).

### C7 Replication on Wagenaar (summary)
Replicated: development (bursts more frequent, larger share of spikes in bursts, m_all falling), m ≈ 0.99 with whole ≈ inter-burst, onset over-representation of background-active channels, and stable within-culture maps. Not replicated or not testable: spatial concentration of ignition centroids beyond the K-spike averaging artefact (59 electrodes at 200 µm pitch are too coarse for a 20-spike centroid), and the bin-level advantage of the eigenvector predictor.

## 3. Controls added after the first look at cross-session results (stated for honesty)
1. **Random-window null for C2a** (`extra_controls.py`). The specified null draws single burst spikes, while the observed ignition point is a 20-spike centroid; averaging pulls centroids toward the centre of mass, so the comparison is biased toward "concentrated" on small arrays (Wagenaar) and toward "diffuse" when activity is dominated by a few very active electrodes (DANDI). Both nulls are reported; conclusions use the random-window null.
2. **Routing-density control for C3** (number of active channels within 200 µm). Added because the centroid of spikes is pulled toward densely routed regions.
3. **2-s detrended m** to check that inter-burst m ≈ 1 is not slow non-stationarity.
4. **Onset-composition check** (top-10 % background channels: share of first-20 spikes vs share of in-burst and out-of-burst spikes).
5. **C3 (v) eigenvector predictor**, requested by the coordinator after the first pass (`c3_eigen.py`).
No parameter of the fixed pipeline was changed after seeing results.

## 4. Verdict
In HD-MEA cortical cultures, spontaneous network bursts start locally: the first spikes cover ~60 % of the radius of a random set of burst spikes, and a small set of channels fires the first spike of most bursts. The onset locations form a few hot spots that recur from one day to the next within a culture (map r ≈ 0.4 vs ≈ 0 between cultures; hot spots ~0.6 mm apart vs 1.7 mm), but they are only modestly more concentrated than random 20-spike windows of the same bursts (and not more concentrated than the specified null). This "concentrated and stable" result is moderate and partly entangled with electrode routing. It does not replicate on the coarse 59-electrode arrays, where the apparent concentration is an averaging artefact. What predicts ignition is not the plain between-burst firing rate of a location (ρ ≈ 0 per bin; the "local density" effect disappears once routing density is controlled). It is the location's weight in the leading eigenvector of the between-burst functional-coupling matrix (ρ ≈ 0.25, beats the rate in 88 % of sessions and 7/7 cultures, survives the routing control). This is a coordinated-noise form of the noise-focusing idea and parallels the fly result, but the effect is modest. At channel level, background-active channels are over-represented at onset relative to their burst share. That support for noise focusing is weaker because the comparison mixes in sampling effects. The population sits close to, but just below, runaway (m ≈ 0.99, τ ≈ 0.2–0.4 s), both with and without bursts. With development it moves *away* from m = 1 as bursts become frequent and dominant (7/7 and 30/30 cultures for whole-session m). Within a 30-min session m stays flat (DANDI) or falls very slightly (Wagenaar, Δm ≈ −0.002), which is fatigue-like and not SHY-like.

## 5. Files
- `results/culture/sessions_dandi.csv`, `sessions_wagenaar.csv`: one row per session, all metrics (base parameters; `excluded_empty`, `n_bursts_ign` ≥ 20 flags C2/C3 eligibility; `day`).
- `results/culture/sensitivity_{dandi,wagenaar}.csv`: headline medians for θ ×0.5/×2, K 10/40, recruitment 10 %/40 %.
- `results/culture/summary.json` (C1–C6 statistics), `summary_controls.json` (eigenvector, random-window null, routing control, detrended m, onset composition).
- `results/culture/c3_eigen_*.csv`, `controls_*.csv`, `onset_composition_*.csv`, `ignition_stability_pairs_*.csv`: per-session / per-pair values behind the summaries.
- `results/culture/ignition_maps_dandi.npz` (70 sessions: smoothed maps `ign_pmap` [session, x, y] on the 100-µm grid, counts, burst-density maps, support masks, all ignition points, culture/day labels); `ignition_maps_wagenaar.npz`.
- Example sessions (spikes, channels, positions, burst windows, ignition points, 10-ms and 4-ms population counts): `example_dandi_sub-19880_ses-19880-240318.npz` (4.7 MB), `example_dandi_sub-23312_ses-23312-240504.npz` (9.3 MB), `example_wagenaar_6-3-21.npz` (6.4 MB).
- Diagnostics: `results/culture/diag/` — rasters with bursts and ignition points (3), ignition maps across days for cultures 19880 and 23312, development panels (m, burst rate, out-of-burst fraction, participation, τ) for both datasets, concentration-vs-null and predictor boxplots.
- Per-session cache (pickles with all per-burst arrays): `results/culture/cache/` (23 MB).
