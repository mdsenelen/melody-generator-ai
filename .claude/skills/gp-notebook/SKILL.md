---
description: GP4. Improve the Colab training notebook, fix the mood heuristic, and evaluate Claude API integration options.
disable-model-invocation: true
---

Read @docs/GUIDED-PASS.md, section GP4.

Independent of GP1 to GP3. Touches only the notebook and
`backend/app/model/colab_parity.py`. Safe to run in a parallel session on its own branch.

### Deliverable 1: `docs/notebook-claude-api.md`

Written comparison of Claude API integration options for the notebook, one section each:
data augmentation, labelling, output review. Per option: the concrete use, requests per
training run, and the effect on reproducibility of a thesis artifact. End with a
recommendation. This is a document, not a conversation.

### Deliverable 2: mood heuristic

`heuristic_mood_from_metrics` uses only tempo, key, and average pitch. Read its actual
thresholds and identify why neutral input lands in sad or happy.

Implement the extended-feature heuristic, not a trained classifier. Reason: a classifier
adds a training dependency and a reproducibility burden to a thesis artifact, for accuracy
we cannot yet measure. Add valence-relevant features that stay explainable: note density,
interval direction, mode strength, dynamic range, rhythmic regularity. Keep every
threshold named and commented.

Write the classifier alternative up in `docs/adr/` as a rejected option with the reason.

### Deliverable 3: training quality

Concrete improvements to CVAE and IDDM-PPO grounded in what the notebook does. Not a
rewrite. Each with the expected effect and the metric that would show it worked. Implement
the ones that are low risk and independently verifiable; list the rest in the report.

Verify: notebook runs top to bottom, mood output is sane across a spread of test clips.
Commit GP4 alone.
