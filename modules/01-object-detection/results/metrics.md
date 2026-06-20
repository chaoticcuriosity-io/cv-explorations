## COCO detection benchmark

| Model | Family | License | mAP | mAP@50 | mAP@75 | mAP (small) | Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RF-DETR Medium | DETR (transformer) | Apache-2.0 | 59.4 | 78.2 | 63.7 | 31.6 | 36.3 | 27.5 |
| RF-DETR Nano | DETR (transformer) | Apache-2.0 | 51.3 | 69.9 | 54.2 | 22.4 | 36.43 | 27.5 |
| YOLO26-n | YOLO (NMS-free) | AGPL-3.0 | 43.0 | 58.1 | 47.2 | 20.1 | 18.75 | 53.3 |
| YOLO11-n | YOLO (anchor-free) | AGPL-3.0 | 37.6 | 52.1 | 40.5 | 17.9 | 18.0 | 55.6 |

*Evaluated on 200 COCO val2017 images; mAP via pycocotools-equivalent torchmetrics at IoU=.50:.95. Latency = mean over 30 warmed-up single-image forwards on `NVIDIA GeForce RTX 3090 Ti | 24.0 GB | sm_86 | torch 2.11.0+cu128 / CUDA 12.8`.*
