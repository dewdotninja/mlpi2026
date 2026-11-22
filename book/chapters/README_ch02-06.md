# Chapters 2–6 (course Weeks 2–5) — draft chapters

Draft chapters for a robotics-and-automation machine learning textbook that replaces the course's
use of Raschka, Liu & Mirjalili, *Machine Learning with PyTorch and Scikit-Learn* (2022), for
Weeks 2–5.

Numbered to the 17-chapter book outline (Ch 2 data and preprocessing, Ch 3 classifiers,
Ch 4 regression, Ch 5 ensembles, Ch 6 unsupervised learning).

Each chapter folder holds:

* `chXX_*.md` — the chapter text (equations in LaTeX, figures referenced from `figures/`);
* `chXX_*.ipynb` — the companion notebook, executed, with all outputs embedded;
* `figures/` — every figure used by the text, written by the notebook (`savefig(...)`).

All data sets are **synthetic but physically built** and generated inside each notebook with a
fixed seed, so every notebook runs offline and every number in the text is reproducible. No book
figures or third-party data sets (Iris, Wine, Ames Housing, Kaggle) are used.

| week | chapter | running example | replaces |
|---|---|---|---|
| 2 | [Ch 2 — Data from machines](ch02_data_from_machines/ch02_data_from_machines.md) | vibration waveforms → condition indicators for 24 machines in four conditions; missing tachometer data, cable spikes, one-hot encoding, pipelines, and splitting by machine | Lect 2 preprocessing slides and Lab 2 (no notebook existed) |
| 2 | [Ch 3 — A tour of classifiers](ch03_classifiers/ch03_classifiers.md) | sorting washers, nuts and bushings on a conveyor from vision + load-cell measurements; reachable workspace of a 2-link arm (kernel SVM) | Lect 2 supplements 1–3 (Iris, Raschka ch. 3 figures) |
| 3 | [Ch 4 — Regression](ch04_regression/ch04_regression.md) | predicting CNC spindle power from cutting parameters (MRR physics, chatter outliers); Stribeck friction of a robot joint (LWR) | Lect 3 supplement 1 (Ames Housing, Kaggle salary data) |
| 4 | [Ch 5 — Ensemble learning](ch05_ensembles/ch05_ensembles.md) | diagnosing bearing vs gear-tooth wear in a robot wrist gearbox from 13 vibration/current indicators | Lect 4 supplement 1 (Wine, Iris, Raschka ch. 7 figures) |
| 5 | [Ch 6 — Unsupervised learning: PCA, LDA and clustering](ch06_unsupervised/ch06_unsupervised.md) | PCA/LDA on the gearbox indicators; k-means on a palletising cell's cycle log | Lect 5 supplements 1–2 (Wine, `make_blobs`) |

The gearbox data set is shared by Chapters 5 and 6 (same generator, same seed).

## Running the notebooks

Tested on Python 3.13 with numpy 2.5, pandas 3.0, scikit-learn 1.9, matplotlib 3.11 and
xgboost 3.4 (optional; Chapter 5 falls back to `HistGradientBoostingClassifier`). Nothing else
is required — in particular no `mlxtend`, `kagglehub` or `distutils`.

```bash
pip install numpy pandas scikit-learn matplotlib scipy xgboost
jupyter nbconvert --to notebook --execute --inplace ch03_classifiers/ch03_classifiers.ipynb
```

Run each notebook from its own folder so that `figures/` is written next to the chapter.
Approximate CPU run times: Ch 2 ≈ 40 s, Ch 3 ≈ 15 s, Ch 4 ≈ 10 s, Ch 5 ≈ 1.5 min, Ch 6 ≈ 10 s.
