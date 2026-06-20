## Classical CV: operation timings (CPU)

| Operation | Latency (ms) |
| --- | --- |
| Canny edges | 0.76 |
| ORB detect+match | 8.23 |
| Farneback optical flow | 96.94 |

*All on CPU, no GPU, no training, no model download. Compare these millisecond timings to the deep models in modules 01–08 — classical CV is essentially free, which is why it still underpins calibration, SLAM front-ends, and stabilisation.*
