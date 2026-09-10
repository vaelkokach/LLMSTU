# Experiment status — 2026-09-10

Current state and what is in flight. `FINDINGS.md` remains the long-form record;
this is the short answer to "where are we".

All numbers below are **validation**, MS-TCN, three seeds, and the test split is
**unspent** for every one of them.

---

## 1. The diagnosis

macro-F1 on the six cue classes sat at ~0.48 and no architecture moved it. Four
measurements say why, and they agree:

| evidence | value |
|---|---|
| `looking_away` under-credited by the precedence rule | true by its own rule **1313×**, allowed to be the label **479×** (2.74×) — every other class is 1.00–1.04× |
| records where more than one cue rule fires | **13.7%**, and 772 of the losses are one pattern: slumped **and** distracted, kept as `head_down` |
| face detected, `screen_oriented` vs `looking_away` | **92% vs 90%** — indistinguishable to every cheap signal (`head_down` is 8%) |
| head size in the encoder | a seated student's head lands on **~1.4 of CLIP-B/32's 49 patches**, at any camera distance, because the 224×224 resize normalises scale away |

So the two failing classes (`looking_away` 0.21, `turned_to_peer` 0.17) are separated
only by gaze, at a head size where neither the encoder nor the VLM annotator could
resolve it. Label-limited *and* resolution-limited; adding explicit head yaw
(553→556) buys +0.041, and 14 further dims buy nothing.

## 2. Results

### Coarser taxonomies (`llmstu_sequences_full_det`, 556_hp)

| taxonomy | macro-F1 | coverage | note |
|---|---|---|---|
| `cue6` (matched baseline) | 0.479 ± 0.007 | 100% | |
| **`onoff_reliable`** | **0.768 ± 0.016** | 91% | on_task 0.90, off_task 0.64 |
| `coarse3_reliable` | 0.706 ± 0.055 | 91% | **too unstable to report** — sd 0.055, CIs span [0.46, 0.85] |

`_reliable` abstains on `looking_away` and `turned_to_peer`. **0.768 over 2 classes at
91% coverage is not comparable to 0.479 over 6 at 100%** — averaging over fewer classes
is easier and abstention removes the hardest 9%. Report it as a different, better-posed
task, not as an improvement to the model.

### Head stream (`llmstu_sequences_head`, 1074_hp_head)

| | macro-F1 | looking_away | phone_use | uncertain |
|---|---|---|---|---|
| baseline | 0.479 ± 0.007 | 0.226 | 0.474 | 0.488 |
| **head stream** | **0.515 ± 0.015** | **0.271** | **0.540** | **0.535** |

A second CLIP pass over the head cropped from the full frame at native resolution
(+518 dims). **+0.036 macro-F1**, and the gains land where the diagnosis predicted —
head- and small-object-scale cues. `turned_to_peer` barely moved (+0.017), the one
prediction that did not hold.

### Partial labels — PRODEN (negative result, and a clean one)

0.378 ± 0.015, with **`head_down` = 0.000 in every run**.

Not a bug. `head_down` is the label on 773 records and **772 of them also carry
`looking_away` as a candidate**, so there is essentially no frame where `head_down` is
the only candidate. PRODEN concentrates on whichever candidate the model can already
support; with no unambiguous anchor the mass drains entirely to `looking_away`. That
violates the identifiability condition partial-label learning depends on — a
mechanistic result about when PLL applies to precedence-collapsed labels.

(An earlier run at 0.35 was additionally depressed by a reduction bug — `proden_loss`
divided by frame count where `F.cross_entropy(weight=...)` divides by the sum of target
weights, making it 2.21× too small. Fixed in `71fba1b`; `head_down` stayed at 0.000
either way.)

## 3. Deployed

* Space `WaelK/classroom-attention-cues-live` (t4-medium, private), running
  `ff_det/mstcn_553_facefound@s42` — the checkpoint `attention_runtime.yaml` names,
  T=0.924, display ≥0.46, alert ≥0.66.
* Verified end to end on real video: 300 frames, 6 students, 3.48 fps, calibrated
  abstention visibly gating output.
* `deploy/hf_space_live/app.py` splits artifacts by shape — weights to durable
  storage, session cache to ephemeral — because the durable mount is a bucket and
  900 small JPEGs stalled a boot.

## 4. In flight / next

| item | state |
|---|---|
| VLM fusion (`attention/fusion.py`, `vlm_grounder.py`) | built and tested, **not wired** into `pipeline_bridge` or the dashboard |
| Live RTSP/HTTP source | server side done (`693ef37`), **UI control missing** |
| Test split for the coarse taxonomies | unspent — the obvious next step for a reportable number |
| `coarse3_reliable` instability | needs more seeds or dropping |
| Option-order sensitivity of the VLM scorer | unmeasured; permute and check before quoting any agreement rate |

### Open threats (`THESIS_DEFENSIBILITY_REVIEW.md`)

* **A — ethics**: consent obtained per the author, but no document in the repo.
* **C — single annotator**: no inter-annotator agreement for LLMSTU.
* **D — pseudo-label provenance**: partly resolved. `llmstu/config.py:47` names
  `Qwen/Qwen3.5-27B`, which **does exist** on the Hub; the ambiguity is in the prose,
  not the config.
* **No temporal test number exists.** 23 of the detector's 27 test videos are in the
  cue model's training set (`FINDINGS.md` §3.4c). Any joint or fused number is
  validation-only until sequences are rebuilt under the detector's split.

### If the VLM fusion is picked up

The temporal model learned from Qwen3.5-27B's labels, so a Qwen-family VLM is not a
fully independent second opinion — agreement partly measures shared bias. The default
backend is `Qwen3-VL-4B-Instruct`, a different size never adapted to these labels,
which weakens the coupling without removing it. Run the VLM on **native-resolution
video**, never on the session cache: those JPEGs are 960 px wide against a 1918 px
source, which would put the head at ~16 px and reproduce the exact failure this whole
diagnosis is about.
