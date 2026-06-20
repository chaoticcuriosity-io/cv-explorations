## Vision-language model

| Model | License | Latency (ms/answer, 32 tok) | Note |
| --- | --- | --- | --- |
| Qwen2.5-VL 3B | Apache-2.0 | 3729.64 | free-form text; qualitative |

*One model answers caption/reason/count/OCR purely from the prompt — no task-specific heads. Latency is per short answer on `NVIDIA GeForce RTX 3090 Ti | 24.0 GB | sm_86 | torch 2.11.0+cu128 / CUDA 12.8` and scales with the number of generated tokens. See the transcript for real outputs.*
