"""A VLM's second opinion on each tracked student.

Interface
---------
A grounder answers one question: given a frame and the students' boxes, how
does each student score against the six cue phrases? Everything else -- the
tracker, the temporal model, the fusion policy -- is unchanged.

    scores = grounder.score_students(frame_bgr, boxes)   # [n_boxes, 6]

Rows are probability distributions over ``taxonomy.CUE_CLASSES``, in that
order. A backend that cannot judge a student returns a uniform row rather than
guessing, so the fusion layer sees "no opinion" instead of a confident wrong
one.

Why option-likelihood scoring instead of generating JSON
--------------------------------------------------------
The labelling pipeline had the VLM emit the full 10-field schema and then
collapsed it through ``map_record``. Faithful, but it costs up to 512 generated
tokens per student -- seconds each on a T4, against the pipeline's 3.48 fps.

This scores the six options in a single forward pass and reads the logits at
the answer position, giving a distribution over cues at roughly the cost of one
prefill. The trade is real and worth stating: option-letter scoring is
sensitive to option ORDER and to wording, in a way that free generation is not.
The order here is fixed to CUE_CLASSES and recorded in the output so a later
audit can re-run with a permuted order and measure the sensitivity rather than
assume it away.

Independence caveat
-------------------
The temporal model was trained on labels from Qwen3.5-27B. Querying a
Qwen-family VLM is therefore not a fully independent second opinion. Using a
smaller base model that was never adapted to these labels weakens the coupling;
it does not remove it. See fusion.py.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Sequence

import numpy as np

from .cue_phrases import phrases_in_class_order
from .taxonomy import CUE_CLASSES

NUM_CUES = len(CUE_CLASSES)
#: Letters presented to the model, one per cue, in CUE_CLASSES order.
OPTION_LETTERS = [chr(ord("A") + i) for i in range(NUM_CUES)]


class Grounder(Protocol):
    """Anything that can score students against the cue phrases."""

    def score_students(self, frame_bgr: np.ndarray,
                       boxes: Sequence[Sequence[float]]) -> np.ndarray:
        ...


def uniform_rows(n: int) -> np.ndarray:
    """`n` rows of "no opinion" -- the honest output when scoring fails."""
    return np.full((n, NUM_CUES), 1.0 / NUM_CUES, dtype=np.float32)


def build_prompt() -> str:
    """The multiple-choice question, built from the phrase mapping.

    Mirrors the labelling pipeline's guard clause: judge the student in the
    centre of the crop, describe visible behaviour, never infer identity.
    """
    opts = "\n".join(f"{L}. {p}" for L, p in
                     zip(OPTION_LETTERS, phrases_in_class_order()))
    return (
        "Look only at the student in the centre of this crop. Describe what is "
        "visible; do not guess identity, age, gender or ethnicity.\n\n"
        "Which ONE option best describes this student?\n"
        f"{opts}\n\n"
        f"Answer with a single letter ({OPTION_LETTERS[0]}-{OPTION_LETTERS[-1]})."
    )


def crop_students(frame_bgr: np.ndarray,
                  boxes: Sequence[Sequence[float]],
                  pad: float = 0.10) -> List[Optional[np.ndarray]]:
    """Crops for each box, padded slightly, or None where degenerate.

    Padding gives the model a little context around the student -- a phone or a
    neighbour can sit just outside a tight detector box, and both matter to the
    cue being judged.
    """
    h, w = frame_bgr.shape[:2]
    out: List[Optional[np.ndarray]] = []
    for b in boxes:
        x1, y1, x2, y2 = (float(v) for v in b[:4])
        px, py = (x2 - x1) * pad, (y2 - y1) * pad
        xa, ya = int(max(0, x1 - px)), int(max(0, y1 - py))
        xb, yb = int(min(w, x2 + px)), int(min(h, y2 + py))
        out.append(frame_bgr[ya:yb, xa:xb] if xb > xa and yb > ya else None)
    return out


class StubGrounder:
    """Deterministic fake, for tests and for running the UI without weights.

    Not random: a stub that returned noise would make a disagreement rate look
    like a measurement. This returns a fixed distribution derived from the box
    index, so tests can assert exact fusion outcomes.
    """

    def __init__(self, cue_ids: Optional[Sequence[int]] = None, conf: float = 0.7):
        self.cue_ids = cue_ids
        self.conf = float(conf)

    def score_students(self, frame_bgr: np.ndarray,
                       boxes: Sequence[Sequence[float]]) -> np.ndarray:
        n = len(boxes)
        rows = uniform_rows(n).copy()
        rest = (1.0 - self.conf) / (NUM_CUES - 1)
        for i in range(n):
            cid = (self.cue_ids[i] if self.cue_ids is not None
                   else i % NUM_CUES)
            rows[i, :] = rest
            rows[i, cid] = self.conf
        return rows


class QwenGrounder:
    """Qwen3-VL scoring the six options in one forward pass per student.

    Defaults to Qwen3-VL-4B-Instruct: about 8 GB in fp16, which fits a T4
    alongside the detector, and it is NOT the model that produced the training
    labels (that was Qwen3.5-27B) -- which is the point.

    fp16 rather than bf16 because a T4 is Turing and has no bf16 units; asking
    for bf16 there is silently emulated and slow.
    """

    def __init__(self,
                 model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
                 device: str = "cuda:0",
                 dtype: str = "float16",
                 max_students: int = 12):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.max_students = int(max_students)
        self._model = None
        self._proc = None
        self._letter_ids: Optional[List[int]] = None

    # -- lazy load: the dashboard must start without paying for the VLM -----
    def _ensure(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        td = getattr(torch, self.dtype)
        self._proc = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, torch_dtype=td).to(self.device).eval()

        # Token id of each option letter. Resolved once, and checked: if a
        # letter does not map to a single token the logit read would be
        # measuring a prefix, and every score after it would be wrong.
        tok = self._proc.tokenizer
        ids = []
        for L in OPTION_LETTERS:
            enc = tok.encode(L, add_special_tokens=False)
            if len(enc) != 1:
                raise RuntimeError(
                    f"option letter {L!r} tokenises to {len(enc)} tokens under "
                    f"{self.model_id}; option-likelihood scoring needs one "
                    f"token per letter.")
            ids.append(enc[0])
        self._letter_ids = ids

    def score_students(self, frame_bgr: np.ndarray,
                       boxes: Sequence[Sequence[float]]) -> np.ndarray:
        import cv2
        import torch
        from PIL import Image

        n = len(boxes)
        scores = uniform_rows(n)
        if n == 0:
            return scores
        self._ensure()

        crops = crop_students(frame_bgr, boxes)
        idx = [i for i, c in enumerate(crops) if c is not None][:self.max_students]
        if not idx:
            return scores

        prompt = build_prompt()
        images = [Image.fromarray(cv2.cvtColor(crops[i], cv2.COLOR_BGR2RGB))
                  for i in idx]
        msgs = [[{"role": "user", "content": [{"type": "image"},
                                              {"type": "text", "text": prompt}]}]
                for _ in idx]
        texts = [self._proc.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True) for m in msgs]
        inputs = self._proc(text=texts, images=images, return_tensors="pt",
                            padding=True).to(self.device)

        with torch.inference_mode():
            logits = self._model(**inputs).logits[:, -1, :].float()

        sel = logits[:, self._letter_ids]              # [k, 6]
        probs = torch.softmax(sel, dim=-1).cpu().numpy()
        for row, i in enumerate(idx):
            scores[i] = probs[row]
        return scores
