## Image retrieval (Imagenette)

| Model | Type | License | Recall@1 (%) | Precision@5 (%) | Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- |
| SigLIP 2 Base | image-text | Apache-2.0 | 99.8 | 99.2 | 20.51 | 48.8 |
| DINOv2 Base | self-supervised | Apache-2.0 | 99.3 | 98.8 | 19.45 | 51.4 |

*Retrieval on 600 Imagenette images (60/class). Recall@1 = nearest neighbour shares the class; Precision@5 = fraction of top-5 that do. No labels used at search time — purely embedding similarity. Latency on `NVIDIA GeForce RTX 3090 Ti | 24.0 GB | sm_86 | torch 2.11.0+cu128 / CUDA 12.8`. Figure uses DINOv2 Base.*
