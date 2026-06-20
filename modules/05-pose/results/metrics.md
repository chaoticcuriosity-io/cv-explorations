## Pose estimation: speed

| Model | License | Latency (ms) | FPS | People/img |
| --- | --- | --- | --- | --- |
| YOLO26-n-pose | AGPL-3.0 | 21.52 | 46.5 | 3.2 |
| YOLO11-n-pose | AGPL-3.0 | 18.33 | 54.5 | 2.5 |

*17 COCO keypoints per person, single-stage (bottom-up). Latency = mean warmed-up single-image forward on `NVIDIA GeForce RTX 3090 Ti | 24.0 GB | sm_86 | torch 2.11.0+cu128 / CUDA 12.8`. Qualitative + latency; COCO keypoint OKS-AP is a deeper follow-up.*
