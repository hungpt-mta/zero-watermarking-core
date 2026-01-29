# Zero-Watermarking (Core Implementation)

This repository provides the **core implementation** of the zero-watermarking framework proposed in the accompanying paper.
The code corresponds to the **custom algorithms central to the main conclusions** of the study and is intended to support
algorithmic transparency, peer review, and reproducibility.

Specifically, the repository includes the following core components:

1. **Registration** – zero-watermark generation from a reference image  
   (`generate_zero_watermark(...)`)
2. **Authentication** – watermark extraction and verification from a (possibly attacked) image  
   (`authenticate(...)`)
3. **Geometric correction / registration** under geometric attacks using feature-based matching  
   (`correcting_image(...)`, based on `TempMatcher`)

The repository intentionally **does not include full experimental orchestration, datasets, or benchmarking scripts**.
Only the essential algorithms required to reproduce the results reported in the paper are provided.

---

## Repository structure

```
.
├─ src/
│  ├─ zero_watermarking_core.py   # core pipeline (register/authenticate/correct)
│  └─ utility.py                  # attacks + metrics + Arnold transform
├─ examples/
│  └─ ZeroWatermarking_Core_Implementation.ipynb
├─ requirements.txt
└─ README.md
```

## Quick start

```bash
pip install -r requirements.txt
```

Then import the core functions:

```python
from src.zero_watermarking_core import (
    generate_system_params_and_key, generate_user_keys,
    generate_zero_watermark, authenticate, correcting_image
)
```

## Notes

- The code expects to read/write intermediate files in subfolders under the working directory:
  `./keypoints`, `./descriptors`, `./ms_img`, `./os_img`, `./RCs`, etc.
- If you want a “pure in-memory” version (no intermediate files), you can refactor the I/O in a later step.

## License

This code is released under an open-source license and may be used for research and academic purposes.