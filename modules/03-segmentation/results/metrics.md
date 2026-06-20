## Segmentation: speed & masks

| Model | Family | License | Latency (ms) | FPS | Masks/img |
| --- | --- | --- | --- | --- | --- |
| YOLO26-n-seg | instance (YOLO) | AGPL-3.0 | 23.79 | 42.0 | 7.0 |
| YOLO11-n-seg | instance (YOLO) | AGPL-3.0 | 22.9 | 43.7 | 6.5 |
| SAM 2.1 (base) | promptable | Apache-2.0 | 93.39 | 10.7 | n/a |

*Latency = mean over warmed-up single-image forwards on `NVIDIA GeForce RTX 3090 Ti | 24.0 GB | sm_86 | torch 2.11.0+cu128 / CUDA 12.8`. SAM 2.1 is prompted with the YOLO26-seg boxes (detect-then-segment). This module is qualitative + latency; mask-mAP on COCO is a deeper follow-up.*
