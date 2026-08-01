### Table A — visible-cue classification (test split)

| model | dims | added feature block | seeds | accuracy | balanced acc | macro-F1 | macro-F1 95% CI (seeds) | macro-F1 95% CI (video bootstrap, seed 42) | macro-AUPRC | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| asrf | 556 | + head pose (4) | 3 | 0.7096 ± 0.0221 | 0.5127 ± 0.0348 | **0.4771 ± 0.0161** | [0.4588, 0.4953] | [0.4061, 0.5125] | 0.4734 | 0.0333 |
| asrf | 570 | + head pose + expression + dynamics (14) | 3 | 0.7348 ± 0.0227 | 0.5213 ± 0.0018 | **0.4924 ± 0.0187** | [0.4713, 0.5136] | [0.3985, 0.5156] | 0.4822 | 0.0309 |
| mstcn | 556 | + head pose (4) | 3 | 0.7589 ± 0.0023 | 0.5190 ± 0.0231 | **0.4998 ± 0.0114** | [0.4869, 0.5127] | [0.4363, 0.5449] | 0.4888 | 0.0317 |
| mstcn | 570 | + head pose + expression + dynamics (14) | 3 | 0.7651 ± 0.0182 | 0.4810 ± 0.0295 | **0.4881 ± 0.0226** | [0.4625, 0.5137] | [0.4447, 0.5398] | 0.4767 | 0.0246 |

### Per-class F1 (mean ± sd over seeds)

| config | screen_oriented | looking_away | head_down | turned_to_peer | phone_use | uncertain |
|---|---|---|---|---|---|---|
| asrf::556_hp | 0.828 ± 0.020 | 0.248 ± 0.012 | 0.612 ± 0.025 | 0.212 ± 0.010 | 0.425 ± 0.072 | 0.537 ± 0.047 |
| mstcn::556_hp | 0.865 ± 0.001 | 0.226 ± 0.016 | 0.641 ± 0.015 | 0.185 ± 0.009 | 0.539 ± 0.053 | 0.542 ± 0.021 |
| asrf::570_full | 0.846 ± 0.016 | 0.258 ± 0.020 | 0.607 ± 0.044 | 0.201 ± 0.033 | 0.500 ± 0.024 | 0.543 ± 0.023 |
| mstcn::570_full | 0.870 ± 0.010 | 0.215 ± 0.031 | 0.646 ± 0.012 | 0.197 ± 0.029 | 0.492 ± 0.078 | 0.508 ± 0.089 |

### Isolated feature contrasts (paired video-level bootstrap)

| contrast | Δ macro-F1 (seed means) | seeds with a significant Δ | verdict |
|---|---|---|---|
