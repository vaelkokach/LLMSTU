### Table B — human-gold temporal events (gold_dedup)

**16 distinct gold episodes over 10 tracks; mean ± sd over 3 seeds; episode matching at tIoU 0.30.**

Detector and tracker are excluded by design — track identity comes from the manifest — so these numbers isolate the cue + event layers. This is a **diagnostic** human-gold set (two densely annotated segments), not a classroom-wide estimate: one episode is 6 percentage points of recall.

| system | frame acc | frame macro-F1 | F1@10 | F1@25 | F1@50 | edit | event P | event R | event F1 | onset MAE (s) | offset MAE (s) | dur MAE (s) | FA/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| transformer_552 | 0.738 ± 0.030 | 0.439 ± 0.081 | 0.309 ± 0.051 | 0.244 ± 0.042 | 0.169 ± 0.042 | 30.9 ± 8.2 | 0.560 ± 0.063 | 0.292 ± 0.036 | 0.383 ± 0.044 | 9.93 ± 5.37 | 1.33 ± 1.15 | 11.26 ± 6.36 | 9.35 ± 1.47 |
| transformer_556 | 0.782 ± 0.024 | 0.544 ± 0.045 | 0.307 ± 0.041 | 0.265 ± 0.051 | 0.211 ± 0.043 | 38.2 ± 2.7 | 0.622 ± 0.038 | 0.208 ± 0.036 | 0.312 ± 0.045 | 1.22 ± 0.69 | 8.85 ± 9.17 | 9.62 ± 8.34 | 5.10 ± 0.00 |
| transformer_563expr | 0.772 ± 0.016 | 0.535 ± 0.006 | 0.324 ± 0.064 | 0.286 ± 0.047 | 0.236 ± 0.049 | 41.1 ± 5.0 | 0.556 ± 0.120 | 0.229 ± 0.036 | 0.324 ± 0.055 | 4.83 ± 1.01 | 0.48 ± 0.48 | 5.30 ± 1.26 | 7.65 ± 2.55 |
| transformer_563dyn | 0.767 ± 0.044 | 0.538 ± 0.036 | 0.338 ± 0.033 | 0.300 ± 0.034 | 0.224 ± 0.073 | 40.0 ± 12.0 | 0.519 ± 0.128 | 0.208 ± 0.072 | 0.284 ± 0.063 | 1.25 ± 1.14 | 0.16 ± 0.27 | 1.41 ± 1.23 | 9.35 ± 5.89 |
| transformer_570 | 0.740 ± 0.093 | 0.511 ± 0.051 | 0.268 ± 0.050 | 0.237 ± 0.045 | 0.183 ± 0.066 | 28.8 ± 6.1 | 0.450 ± 0.180 | 0.167 ± 0.036 | 0.242 ± 0.065 | 2.00 ± 2.64 | 0.00 ± 0.00 | 2.00 ± 2.64 | 9.35 ± 5.31 |
| mstcn_556 | 0.788 ± 0.016 | 0.537 ± 0.013 | 0.321 ± 0.031 | 0.291 ± 0.039 | 0.150 ± 0.023 | 48.5 ± 1.1 | 0.554 ± 0.214 | 0.271 ± 0.036 | 0.353 ± 0.031 | 5.03 ± 5.60 | 3.95 ± 2.80 | 8.98 ± 7.11 | 11.05 ± 7.79 |
| mstcn_570 | 0.771 ± 0.009 | 0.502 ± 0.040 | 0.397 ± 0.017 | 0.362 ± 0.018 | 0.227 ± 0.035 | 59.9 ± 5.0 | 0.454 ± 0.009 | 0.312 ± 0.062 | 0.368 ± 0.047 | 10.93 ± 5.34 | 6.86 ± 2.22 | 17.44 ± 6.80 | 15.30 ± 2.55 |
| asrf_556 | 0.775 ± 0.025 | 0.527 ± 0.012 | 0.336 ± 0.005 | 0.277 ± 0.028 | 0.169 ± 0.020 | 36.4 ± 5.7 | 0.496 ± 0.143 | 0.229 ± 0.072 | 0.309 ± 0.082 | 1.15 ± 0.57 | 1.63 ± 2.35 | 2.53 ± 2.16 | 10.20 ± 5.10 |
| asrf_570 | 0.785 ± 0.055 | 0.521 ± 0.048 | 0.340 ± 0.028 | 0.291 ± 0.042 | 0.175 ± 0.051 | 53.4 ± 10.1 | 0.506 ± 0.113 | 0.229 ± 0.036 | 0.315 ± 0.056 | 2.19 ± 2.28 | 5.50 ± 6.91 | 7.35 ± 8.55 | 9.35 ± 2.94 |
| teacher | 0.874 ± 0.000 | 0.796 ± 0.000 | 0.720 ± 0.000 | 0.700 ± 0.000 | 0.670 ± 0.000 | 76.5 ± 0.0 | 0.588 ± 0.000 | 0.625 ± 0.000 | 0.606 ± 0.000 | 2.70 ± 0.00 | 5.39 ± 0.00 | 8.09 ± 0.00 | 17.85 ± 0.00 |
| majority | 0.341 ± 0.000 | 0.085 ± 0.000 | 0.090 ± 0.000 | 0.067 ± 0.000 | 0.022 ± 0.000 | 34.9 ± 0.0 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | n/a | n/a | n/a | 0.00 ± 0.00 |

Notes:
- `teacher` is the Qwen3-VL-family pseudo-labeller measured against the same human gold. It is an **empirical teacher benchmark**, not a theoretical ceiling.
- `majority` is the constant-dominant-class control.
- Boundary errors (onset/offset/duration) are computed over each system's own matched events and are therefore **not** directly comparable across systems with different recall; see the common-matched-subset block in `summary_s*.json` for the like-for-like comparison.
