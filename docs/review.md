# Review: batch processing readiness

## Scope

Reviewed every tracked file in the original repository at `8ce1690153582438387f2a89de387119a7dd8b6e`: workflow, Python analyzer, legend configuration, README, and ignore rules. This update prepares a testable batch pipeline; it does not publish a Public API or certify meteorological forecast accuracy.

## Findings and fixes

| Finding | Impact | Change |
|---|---|---|
| Closest-centroid greedy association | Neighbouring cells could exchange IDs and create spurious motion | Global assignment; predicted position, shifted shape IoU, area ratio; ambiguous/split/merge candidates block ETA |
| Gap divided by gap-closing speed, with a 45° heading gate | A cell can move closer while passing beside the target; shape growth can look like motion | Fit translational velocity to 3–5 consecutive observations; intersect translated pixel footprint with the actual target square |
| ETA beyond 60 minutes and historical tracks marked inbound | Consumers could interpret old or out-of-scope estimates as current alerts | Latest-frame status, ended tracks, 60-minute forecast cap, explicit reasons and nullable ETA |
| Track IDs allocated from zero every execution | T063/T071 across two output files did not necessarily refer to the same echo | Reuse IDs through exact overlapping scan + footprint fingerprints in a persisted state file |
| Only track summaries were exported | Could not inspect motion jumps or matching errors | Per-observation CSV with time, source file, fingerprint, centroid, approximate coordinates, dBZ bounds, gap, step speed, matching cost/IoU, shape-change flags and retrospective status |
| Every >=35 dBZ pixel persistent through ten scans was deleted | Could erase real slow/stationary intense echoes | Remove this rule; treat white as unclassified because of label ambiguity; preserve all other calibrated echo colours |
| dBZ interpolation anchored at sample rows instead of threshold marker boundaries | Systematic bias and misleading precision near cutoffs | Use marker boundaries, keep bin lower/upper bounds; classify using conservative lower bound |
| Configuration duplicated between JSON, analyzer, and inline workflow snippets | Geometry/threshold changes could diverge | One `config/radar.json`, validated image dimensions and legend on every frame |
| PNG signature alone accepted | A truncated/invalid image could enter analysis | Pillow format/dimension/verify/full-decode checks; file size limit |
| No freshness/continuity gate | Old scans or midnight gaps could yield apparently fresh ETA | UTC filename parsing, ICT output, frame count, actual scan intervals, future-time rejection and 18-minute age gate |
| Downloads overwrote cached files before validation | Failed source requests could destroy useful history | Bounded retries/timeouts and staging; newest 10 selected after validation |
| One restore candidate and failure at fewer than 10 scans | Expired artifacts or daily reset caused repeated failures | Try recent valid artifacts; preserve warm-up state and suppress forecasts until history is sufficient |
| Artifact cleanup unconditionally deleted other run artifacts | Concurrent/rerun jobs could delete the usable data bundle | Serialized pipeline; unique upload attempt names; prune only older matching workflow/branch artifacts after verifying the new upload |
| Unpinned Pillow and no tests | Dependency drift and regressions could go unnoticed | Pinned Python/dependencies/action commits, CI regression suite, Dependabot |
| ~500 lines of exploratory inline workflow code | Duplicate work, noisy logs and deprecated getdata calls | Removed colour inventory, palette-distance probes, duplicate area/palette diagnostics; run maintained Python scripts |
| Old calibration file and broad template ignore rules | Unused/confusing settings | Removed `config/legend_anchors.json`, replaced minimal `.gitignore`, documented outputs and operating procedure |

## Verification

- Regression tests exercise head-on arrivals, trajectories that miss despite a favourable heading, footprint-vs-centroid intersection, stationary/away echoes, history gaps, old tracks, ambiguous matches, crossing tracks, ID continuity, palette drift, white labels, real stationary coloured echoes, corrupt PNGs, UTC/ICT midnight, stale/future data, URL constraints, duplicate API records, timeout preservation, midnight image merge, artifact cleanup and full ten-frame output generation.
- Real input replay: ten original 1076 × 800 Sattahip PNGs supplied in the earlier artifact, through 21:18 ICT on 26 September 2026; decoding, tracking, CSV/JSON and diagnostic image generation completed.
- Live-source smoke test: API and ten PNGs from 00:00:03–00:54:03 ICT on 27 September 2026 downloaded and validated; analysis completed with correct approximately six-minute scan intervals.
- Real-input success demonstrates execution and format compatibility. It does **not** establish rain prediction accuracy or confirm that each association represents the same physical storm.

## Remaining release limits

1. Source georeferencing has not been independently verified; approximate latitude/longitude export is labelled as such.
2. PNG palette quantization, map overlays and non-palette pixels prevent recovery of complete raw reflectivity. White intense echoes cannot be distinguished from labels and remain unknown.
3. No ground-truth event dataset or measured false-alarm / missed-event / ETA error statistics exist yet. No calibrated probability or confidence percentage is supplied.
4. Storm growth, decay, splits/merges and unobserved new formation limit constant-velocity forecasts. Ambiguous results are retained as unknown rather than forced into an arrival estimate.
5. The workflow is manual plus code-change triggered. Continuous scheduling, monitoring and any public serving layer remain deployment work.

Operational acceptance must resolve these limits before using the result as an independent marine warning service.
