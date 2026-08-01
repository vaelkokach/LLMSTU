### Table A — visible-cue classification (val split)

| model | dims | added feature block | seeds | accuracy | balanced acc | macro-F1 | macro-F1 95% CI (seeds) | macro-F1 95% CI (video bootstrap, seed 42) | macro-AUPRC | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| asrf | 556 | + head pose (4) | 3 | 0.7294 ± 0.0191 | 0.5319 ± 0.0283 | **0.5063 ± 0.0202** | [0.4834, 0.5292] | [0.3963, 0.5309] | 0.5155 | 0.0399 |
| asrf | 570 | + head pose + expression + dynamics (14) | 3 | 0.7515 ± 0.0055 | 0.5293 ± 0.0077 | **0.5071 ± 0.0096** | [0.4962, 0.5180] | [0.4047, 0.5483] | 0.5146 | 0.0198 |
| mstcn | 556 | + head pose (4) | 3 | 0.7575 ± 0.0012 | 0.5218 ± 0.0068 | **0.5027 ± 0.0050** | [0.4969, 0.5084] | [0.3939, 0.5413] | 0.5093 | 0.0262 |
| mstcn | 570 | + head pose + expression + dynamics (14) | 3 | 0.7752 ± 0.0160 | 0.4891 ± 0.0121 | **0.5038 ± 0.0121** | [0.4901, 0.5175] | [0.3941, 0.5447] | 0.5048 | 0.0211 |

### Per-class F1 (mean ± sd over seeds)

| config | screen_oriented | looking_away | head_down | turned_to_peer | phone_use | uncertain |
|---|---|---|---|---|---|---|
| asrf::556_hp | 0.842 ± 0.018 | 0.269 ± 0.030 | 0.684 ± 0.026 | 0.216 ± 0.014 | 0.471 ± 0.048 | 0.556 ± 0.027 |
| mstcn::556_hp | 0.862 ± 0.001 | 0.229 ± 0.030 | 0.697 ± 0.021 | 0.189 ± 0.004 | 0.509 ± 0.015 | 0.529 ± 0.019 |
| asrf::570_full | 0.856 ± 0.004 | 0.283 ± 0.020 | 0.685 ± 0.022 | 0.206 ± 0.032 | 0.465 ± 0.030 | 0.548 ± 0.004 |
| mstcn::570_full | 0.874 ± 0.010 | 0.231 ± 0.020 | 0.705 ± 0.010 | 0.187 ± 0.024 | 0.486 ± 0.036 | 0.541 ± 0.023 |

### Isolated feature contrasts (paired video-level bootstrap)

| contrast | Δ macro-F1 (seed means) | seeds with a significant Δ | verdict |
|---|---|---|---|
