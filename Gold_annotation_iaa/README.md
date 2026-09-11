# Second-annotator pass — inter-annotator agreement

Drop the file the second annotator sends back here, unchanged:

    Gold_annotation_iaa/gold_annotations_<their-name>.jsonl

It is the file `serve.py` wrote in their copy of the bundle
(`tools/gold_annotator/gold_annotations_<name>.jsonl`). Labels only — no images.

These are the **same 250 crops** the first annotator labelled, drawn stratified
across the six cue classes of their labels
(`tools/gold_annotator/gold_candidates_iaa.jsonl`, built by
`make_iaa_manifest.py`). That overlap is the whole point: the first two gold sets
shared only 3 crops, which is why no kappa could be computed from them
(FINDINGS §17.2).

## Then

    python tools/gold_annotator/compute_agreement.py \
        Gold_annotation_wael/gold_annotations_wael.jsonl \
        Gold_annotation_iaa/gold_annotations_<their-name>.jsonl

Cohen's kappa and percentage agreement per field, over the crops both annotated.

## How to read the result

Both passes were pre-filled from the same pseudo-labels, and the first annotator
kept the pre-filled value on 89.1% of fields. Some of the measured agreement is
therefore two people deferring to one model rather than two people independently
seeing the same thing, so **the kappa is an upper bound on independent
agreement**. Quote it that way. `make_iaa_manifest.py --blind` bounds it from the
other side if a range is ever needed.

If the second pass was done by the *same* person as the first, it is an
intra-annotator (test-retest) number, not an inter-annotator one, and must be
reported as that.
