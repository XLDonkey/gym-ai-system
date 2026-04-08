---
name: Review Clip
about: Pi-flagged low-confidence ONNX inference clip needing human annotation
labels: review-clip, training-data
---

## Clip Details

- **Machine ID:** <!-- e.g. xlf-pi-001 -->
- **Clip file:** <!-- e.g. data/review/xlf-pi-001/2026-04-08/20260408_143022 -->
- **Timestamp:** 
- **Predicted class:** <!-- what the model guessed -->
- **Confidence:** <!-- e.g. 0.38 — below 0.50 threshold -->

## How to annotate

1. Download the `*_meta.json` file from `data/review/`
2. Set `"true_class"` to the correct class ID (0–7):

| ID | Label | Description |
|----|-------|-------------|
| 0 | `no_person` | Nobody at the machine |
| 1 | `user_present` | Person nearby, not seated |
| 2 | `on_machine` | Seated, engaged, not yet lifting |
| 3 | `good_rep` | Full ROM, controlled, weight moving |
| 4 | `bad_rep` | Uncontrolled, bouncing, momentum |
| 5 | `false_rep` | Stretching, adjusting handle/seat/pin |
| 6 | `resting` | Seated between sets |
| 7 | `half_rep` | Partial ROM or single arm only |

3. Commit the updated `_meta.json`
4. When enough clips are reviewed, run `python3 train/train_pytorch.py --review data/review/`

## Notes
<!-- Why was the model confused? Unusual lighting, new user, machine angle? -->
