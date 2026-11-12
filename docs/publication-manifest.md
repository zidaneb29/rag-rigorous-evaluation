# Proposed publication manifest

This allowlist contains the 60 files approved in Phase 7A and the MIT license authorized in Phase 7B. The intended repository is `zidaneb29/rag-rigorous-evaluation`.

## Included files

- `.gitignore`
- `.streamlit/config.toml`
- `LICENSE`
- `README.md`
- `configs/phase3.toml`
- `configs/phase4.toml`
- `configs/phase4b.toml`
- `configs/phase4c.toml`
- `configs/phase5.toml`
- `configs/pilot.toml`
- `configs/ui.toml`
- `docs/experiments.md`
- `docs/figures/paired-recall.svg`
- `docs/figures/quality-latency.svg`
- `docs/phase2.md`
- `docs/phase3.md`
- `docs/phase4.md`
- `docs/phase4b.md`
- `docs/phase4c.md`
- `docs/phase5.md`
- `docs/phase6.md`
- `docs/phase7a.md`
- `docs/publication-manifest.md`
- `docs/resources.md`
- `docs/roadmap.md`
- `docs/specification.md`
- `docs/technical-report.md`
- `docs/third-party-notices.md`
- `notebooks/phase4c.ipynb`
- `rag_eval/__init__.py`
- `rag_eval/__main__.py`
- `rag_eval/chunking.py`
- `rag_eval/corpus.py`
- `rag_eval/evaluation.py`
- `rag_eval/evidence_audit.py`
- `rag_eval/fusion.py`
- `rag_eval/generation.py`
- `rag_eval/phase3.py`
- `rag_eval/phase3_analysis.py`
- `rag_eval/phase3_figure.py`
- `rag_eval/phase4.py`
- `rag_eval/phase4_review.py`
- `rag_eval/phase4b.py`
- `rag_eval/phase4c.py`
- `rag_eval/phase4c_runtime.py`
- `rag_eval/phase5.py`
- `rag_eval/phase5_figures.py`
- `rag_eval/retrieval.py`
- `rag_eval/ui_data.py`
- `requirements-figures.txt`
- `requirements-ui.txt`
- `requirements.txt`
- `results/heldout-dashboard.json`
- `streamlit_app.py`
- `tests/test_phase2.py`
- `tests/test_phase3.py`
- `tests/test_phase4.py`
- `tests/test_phase4b.py`
- `tests/test_phase4c.py`
- `tests/test_phase5.py`
- `tests/test_phase6.py`

## Derived artifacts

- `results/heldout-dashboard.json` is the compact read-only dashboard snapshot. It projects the frozen Phase 5 report, aggregate timing and quality values, and limited IDs, ranks and scores for sampled cases. It includes no QASPER prose, question text, reference answers or original annotation strings.
- `docs/figures/paired-recall.svg` and `docs/figures/quality-latency.svg` are byte-for-byte copies of the verified Phase 5 plots. They contain aggregate measurements only.

| Artifact | SHA-256 |
| --- | --- |
| `results/heldout-dashboard.json` | `21fc4fcbe7dcf976d453f25ed366c31188764d257a8dd678ece19cec42c843f9` |
| `docs/figures/paired-recall.svg` | `43852be410980b6f40328d8b1f6ca3e8d4aca5eb18569bd13e003c9709d59583` |
| `docs/figures/quality-latency.svg` | `e970572f20bbd530505241c2e06ecc069327c9c41de5773add1322520d305b2f` |


## Excluded local material

- QASPER archives, raw paper and annotation files, downloaded model assets, and corpus or vector caches.
- Original run artifacts, checkpoints, review packets, Colab outputs, logs, audit staging files, screenshots and private backups.
- Virtual environments, installed packages, Python bytecode, local overrides, secrets and machine-specific settings.
- Git metadata and credentials. This manifest grants no rights over third-party assets or project code.
