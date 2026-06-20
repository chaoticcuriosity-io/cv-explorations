## Monocular depth: speed

| Model | License | Latency (ms) | FPS |
| --- | --- | --- | --- |
| Depth Anything V2 (Small) | Apache-2.0 | 38.44 | 26.0 |
| Depth Anything V2 (Base) | Apache-2.0 | 78.19 | 12.8 |

*Relative (inverse) monocular depth; in the figure brighter = nearer. Latency = mean warmed-up single-image forward on `NVIDIA GeForce RTX 3090 Ti | 24.0 GB | sm_86 | torch 2.11.0+cu128 / CUDA 12.8`. Qualitative + latency; metric-depth eval (AbsRel/RMSE on NYU/KITTI) is a deeper follow-up.*
