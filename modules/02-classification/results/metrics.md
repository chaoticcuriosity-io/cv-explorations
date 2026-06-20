## Imagenette classification benchmark

| Model | Paradigm | License | Top-1 (%) | Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- |
| ConvNeXt-Tiny | Supervised | Apache-2.0 | 99.9 | 12.31 | 81.2 |
| SigLIP 2 Base | Zero-shot | Apache-2.0 | 99.4 | 23.4 | 42.7 |
| OpenCLIP ViT-B/32 | Zero-shot | MIT | 99.0 | 10.09 | 99.1 |

*Top-1 accuracy on 3925 Imagenette val images (10 classes). Latency = mean of 30 warmed-up single-image forwards on `NVIDIA GeForce RTX 3090 Ti | 24.0 GB | sm_86 | torch 2.11.0+cu128 / CUDA 12.8`. Zero-shot models were never trained on Imagenette — they classify by image-text similarity.*
