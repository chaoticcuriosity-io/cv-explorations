# assets/

Small, committed sample images and short video clips used as **qualitative inputs**
and for **failure-case analysis** across modules (the owner's own / robotics-relevant
imagery plus a few iconic test images).

Keep these small (the repo is public and cloned by readers). Large datasets are
downloaded on demand into `data/` (gitignored) via `scripts/fetch_data.py`.

Access from code with:

```python
from cvkit.data import sample_images
for path in sample_images():
    ...
```
