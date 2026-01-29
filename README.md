# Zero-Watermarking (Core Implementation)

This repository provides the core implementation of a robust image zero-watermarking framework used in the associated research paper. The code includes all algorithmic components required for zero-watermark generation, verification (authentication), and image correction based on ROI selection and SIFT-based registration.

The implementation is intended to support transparency and reproducibility of the reported results, while keeping the repository focused and lightweight.

---

## Scope

The repository includes:

* Zero-watermark generation and verification (authentication)
* Image correction and registration using ROI selection and SIFT-based feature matching (TempMatcher)
* Core utilities required to run the benchmarking procedures described in the paper

The provided code is sufficient to run the benchmarking procedures reported in the associated paper; however, experimental datasets are not redistributed within this repository.

---

## Datasets

The experimental datasets used in this study are publicly available and were obtained from established repositories:

* **Medical images** were randomly selected from **The Cancer Imaging Archive (TCIA)** and can be accessed at:
  [https://nbia.cancerimagingarchive.net/nbia-search/](https://nbia.cancerimagingarchive.net/nbia-search/)

* **Standard color images** were obtained from the widely used **USC-SIPI image database**, available at:
  [http://sipi.usc.edu/database/](http://sipi.usc.edu/database/)

Due to repository policies and to keep this codebase lightweight, the datasets are not redistributed within this repository. Users can independently obtain the same or equivalent datasets from the above sources to reproduce the experiments.

All medical images from TCIA are fully de-identified and publicly released in accordance with applicable ethical and legal standards (including HIPAA). No additional permission or informed consent is required to reuse the images. The USC-SIPI images are publicly available benchmark images widely used for research and reproducibility purposes.

---

## Reproducibility

All algorithmic components required to reproduce the benchmarking results reported in the paper are included in this repository. Reproduction of the experimental results requires access to the publicly available datasets described above, which can be obtained directly from their original sources.

---

## Intended use

This code is provided for academic and research purposes only. It is intended to support reproducibility, verification, and further research on robust image zero-watermarking techniques.

---

## Repository structure

```
zero-watermarking-core/
├── src/          # Core implementation
├── examples/     # Example scripts for running the algorithms
├── requirements.txt
├── README.md
├── LICENSE
└── CITATION.cff
```

---

## Citation

If you use this code in your research, please cite the associated paper and this repository. Citation information is provided in the `CITATION.cff` file.

---

## License

This project is released under the **BSD 3-Clause License**. See the `LICENSE` file for details.
