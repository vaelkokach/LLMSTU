# Second-annotator bundle — inter-annotator agreement

250 student crops to label. Needs only **Python 3** (standard library), no
install, no network, no GPU.

## Run it

```
tar -xzf gold_iaa_bundle.tar.gz -C <any empty directory>
cd <that directory>/tools/gold_annotator
python serve.py \
    --manifest ../../grounding_data/llmstu_tools/outputs/gold_candidates.jsonl \
    --annotator YOUR_NAME
```

Then open <http://localhost:8765/>. Keybindings and the **annotation guideline**
are in `tools/gold_annotator/README.md` — read the guideline before starting,
and do not change it while annotating.

When finished, send back the single file
`tools/gold_annotator/gold_annotations_YOUR_NAME.jsonl`.
It is labels only; it contains no images.

## What this measures, and the one rule that protects it

These 250 crops have **already been labelled once**, by a different annotator.
The point is not to produce more labels — it is to measure how much two people
labelling the same pixels agree, which is the number
`THESIS_DEFENSIBILITY_REVIEW.md` threat C says is missing.

**So label them independently. Do not look at the first annotator's file, and
do not ask what they chose.** Agreement between two people, one of whom has seen
the other's answers, is not agreement — it measures persuasion. If you are the
same person who did the first pass, leave enough time that you are not recalling
individual crops, and say so when you send the file back; a same-annotator
re-label is a useful intra-annotator number but it is not the inter-annotator
one, and it must be reported as what it is.

Three things are deliberately identical to the first pass, because a difference
in any of them would confound the comparison:

* **The same 250 crops**, drawn stratified across the six cue classes of the
  first annotator's labels, so the rare classes (`phone_use`, `turned_to_peer`,
  `uncertain`) are not left in single digits.
* **The pre-filled values are the model's pseudo-labels, not the first
  annotator's.** The tool pre-fills whatever the manifest carries; shipping the
  first pass's labels would turn this into a review of their work.
* **No source-frame context images.** The `f` toggle will show nothing. The
  first pass did not have them either — adding them here would give this pass an
  affordance the first did not have.

## One known limitation, so nobody is surprised by it later

Both passes see the same pre-filled model guesses, and the first annotator kept
the pre-filled value on about **89% of fields**. Some of the agreement this
measures will therefore be two people deferring to the same model rather than
two people independently seeing the same thing, so the resulting kappa is an
**upper bound** on true independent agreement.

That is a deliberate trade — the alternative, labelling blind, would measure a
different task than the first pass performed and the two would not be
comparable. The practical consequence for you is the next section: **treat the
pre-filled value as a guess to check, not as a default to accept.**

## Labelling notes

* Label **what is visible**, not what you infer someone is thinking. "Head is
  down" is a claim about pixels; "not paying attention" is not.
* If a crop genuinely cannot be read — heavily occluded, too small, face not
  visible — mark it `occluded` and use `unknown` rather than guessing. An honest
  `unknown` is more useful here than a confident wrong answer; there is a class
  for exactly this case and it is expected to fire on some of them.
* Disagreeing with the pre-filled value is the normal case, not a problem. The
  pre-fill is a model's guess and it is wrong roughly a quarter of the time.
* Do not skip crops to keep the numbers tidy. A systematically skipped hard case
  is the one thing that would quietly inflate the agreement.

## What happens to it

```
python tools/gold_annotator/compute_agreement.py \
    Gold_annotation_wael/gold_annotations_wael.jsonl \
    <your file>
```

Cohen's kappa and percentage agreement, per field, over the crops you both
labelled.
