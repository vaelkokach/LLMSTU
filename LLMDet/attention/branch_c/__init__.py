"""Branch C — seat/task-relative cue segmentation for a fixed oblique classroom camera.

Named for what the data is, not for what the driving specification assumed. That
specification is written throughout for "fixed **overhead** multi-student video"
and calls the method OVERT (Overhead Viewpoint-invariant...). The corpus is not
overhead: 128 recordings from one room and one fixed corner-mounted camera at
roughly 20-30 degrees elevation, students at desks facing monitors with their
backs to the lens. See FINDINGS.md section 12.1 for the sampled evidence, and
BRANCH_C_PROTOCOL.md section 8 for the narrowed claim this package may support.

The contribution this package exists to test is **not** the machinery. 3DPCNet
(arXiv:2509.23455) already canonicalises pose to a body-centred frame via a
continuous 6D rotation mapped to SO(3), and RAC (arXiv:2605.16999) already has the
clean-corrupted reliability-ordering loss. What survived the novelty audit is
narrower and specific to this setting: representing head orientation relative to
the direction **that student's own monitor faces**, in a room where that direction
differs by seat.

Why that could matter here, stated as a falsifiable prediction rather than a
promise (BRANCH_C_PROTOCOL.md section 5):

FINDINGS section 11.10 established that roughly 80 percent of the existing
head-pose gain is the binary ``face_found`` flag, and that MediaPipe finds a face
on 92 percent of ``screen_oriented`` crops against 8 percent of ``head_down``
crops. In this room a face becomes detectable largely *because* the student has
turned away from the task. The strongest existing feature is therefore a
missingness artifact of one camera pose, not a behavioural signal - and camera-frame
yaw confounds cue with seat, because the yaw that means "looking at my monitor"
differs between a front-left seat and a right-hand one. Seat-frame yaw should not.

Modules:

- :mod:`canonical`  rotation utilities, causal reference frames, relative rotation
- :mod:`fusion`     observability-weighted expert fusion
- :mod:`losses`     the cue/boundary/consistency/reliability objective terms

Nothing here may become a dashboard default until every gate in
BRANCH_C_PROTOCOL.md section 7 has passed. Legacy Branch B stays the default.
"""
