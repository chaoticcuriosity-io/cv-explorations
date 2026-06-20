## Fine-tuned air-vent detector (trained on the DGX Spark GB10)

| Model | Class | mAP@50 | mAP@[.5:.95] | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| YOLO11-s (base, COCO) | air_vent | — | — | — | — |
| **YOLO11-s (fine-tuned)** | air_vent | **0.40** | **0.26** | **0.58** | **0.43** |

*The base model has no `air_vent` class, so it cannot score on this task at all (—): it detects
**zero** vents. The fine-tuned model was trained for 80 epochs on 33 auto-labelled images
(~0.5 s/epoch on the GB10). Metrics are on a 9-image / 14-instance val split whose labels are
themselves Grounding DINO pseudo-labels, so they measure distillation fidelity, not human truth —
read them together with the qualitative before/after.*
