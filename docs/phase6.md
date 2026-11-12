# Phase 6 local Streamlit application

The app presents evidence retrieval and measured evaluation under the title **Evidence Retrieval and Evaluation on QASPER**. It runs on the Windows CPU and uses the existing prepared artifacts.

## Launch

From the repository root, use Python 3.12 and install the UI dependencies into the project environment.

```powershell
python -m venv .venv
$appPython = '.\.venv\Scripts\python.exe'
& $appPython -m pip install -r requirements-ui.txt
$env:OPENBLAS_NUM_THREADS = '2'
$env:TOKENIZERS_PARALLELISM = 'false'
& $appPython -m streamlit run streamlit_app.py
```

Open [the local app](http://127.0.0.1:8501). The server binds to loopback and usage telemetry is disabled. After dependencies and assets are present, search and dashboard loading work offline. Source links open arXiv if the user chooses them. Stop the server with Ctrl+C.

### Local assets and recovery

- **Explorer.** The default configuration reads the checked QASPER train/dev archive, `artifacts/pilot-cold/report.json` and `chunks.json`, and the vector file named in that report under `artifacts/cache`. It uses the BGE tokenizer and weights from [Phase 2](phase2.md), and the optional MiniLM assets from [Phase 3](phase3.md).
- **Missing development index.** Obtain the pinned source and model assets using the Phase 2 and 3 setup guides. Run the existing offline CLI against new paths. Copy `configs/ui.toml` to an ignored local override and set its two development paths to the new directories:

```powershell
& $appPython -m rag_eval --cache artifacts/ui-dev-cache --output artifacts/ui-dev-prepared
Copy-Item configs/ui.toml artifacts/ui.local.toml
# In artifacts/ui.local.toml, set development_run and development_vectors to the two paths above.
$env:QASPER_UI_CONFIG = (Resolve-Path artifacts/ui.local.toml).Path
& $appPython -m streamlit run streamlit_app.py
```

- **Dashboard.** The app prefers the original Phase 5 report and verifies its protocol, population, rankings, evidence and chunk manifests. If the original report is absent, the app uses the bundled hash-pinned text-free dashboard snapshot. An inconsistent original report or corrupted snapshot produces an error. Dashboard use leaves the frozen results untouched.

## Architecture and data boundaries

`streamlit_app.py` renders the explorer and dashboard. `rag_eval/ui_data.py` handles asset validation, prepared-index loading and integration with BM25Okapi, BGE exact cosine search, RRF and MiniLM. `configs/ui.toml` sets local paths, BGE as the default, the frozen report hash and the Top-k limits. `.streamlit/config.toml` sets the loopback and visual defaults.

The explorer lists 20 prepared **development** papers and always restricts search to the selected paper. It reconstructs the development chunks once on the first search and verifies their order, offsets, tokenizer and implementation fingerprints. It opens cached BGE vectors read-only and builds the sparse index only on demand. The app does no embedding or model loading at startup. A resource lock serializes queries that share an index, and one ONNX session is retained at a time. MiniLM search first runs BGE and then replaces that session with the reranker. The displayed retrieval latency excludes model and index setup. The search-request time includes any setup done on that submission. Repeated identical submissions in a session reuse the displayed result. Changes to input artifact sizes or modification times invalidate the corresponding cached resource and result.

Expanded passages show the exact retrieved text span, method-specific score, paper, section, paragraph and chunk identifiers, and raw character offsets. The interface treats those passages as candidates for evidence inspection. It has no generation action.

The dashboard consumes saved Phase 5 values for both scopes. It validates the report and dependency hashes and reconciles the question, paper and eligibility counts. Charts and tables present ranking metrics, paired percentile intervals, median and p95 latency, vector build and storage measurements, and peak working set. The failure panel displays sampled question IDs, per-method ranks and annotated-evidence references. It omits held-out answer text and full paper content. Annotation matches describe benchmark evidence coverage and may omit supporting passages. Generator feasibility history remains in the project reports.

## Verification and limitations

The Windows server was started on a loopback test port, returned HTTP 200 for the health endpoint and root HTML, and was stopped after the check. It reached readiness in **2.29 seconds** at **62.0 MiB** idle working set before a browser session. The first launch exposed Streamlit's development-mode conflict with an explicit port under the target-installed test environment. Setting production mode in the project config fixed it.

On the same Windows laptop, AppTest rendered the explorer in **1.85 seconds** and the first dashboard view in **1.34 seconds**. One smoke-test question produced the following timings. These are UI verification measurements, separate from the held-out benchmark.

| Method | Retrieval | Search request |
| --- | ---: | ---: |
| BGE-small, first search | 16.34 ms | 2,191.12 ms |
| BM25, lazy sparse setup | 0.40 ms | 76.77 ms |
| RRF, warm base indexes | 11.25 ms | 12.92 ms |
| RRF + MiniLM, first reranked search | 1,217.06 ms | 1,527.67 ms |

Peak AppTest working set after the explorer and dashboard checks was **380.6 MiB**. It includes the in-process test harness and dashboard artifacts. Browser memory and sustained memory under concurrent use were outside the measurement. Service cost was $0.

Streamlit AppTest exercised the running script with the real retrieval assets and checked every method, all five requested result panels, paper-scope enforcement, identical-submit reuse and both dashboard scopes. These checks exercise the widget flow and backend. Browser rendering, keyboard navigation, narrow layouts and opening a source link still need a manual pass. No screenshots were captured. Use a desktop-width browser for the portfolio walkthrough and verify narrow-screen tables before publication.

All **50 tests passed**, including the 43 prior tests. Phase 2–5 measured artifact hashes and the Phase 4C checkpoint and packet matched their records. Verification corrected a transcription error in the Phase 4C document's checkpoint SHA-256. The checkpoint bytes and all results were preserved.

Phase 7 remains for review. It should cover a clean Windows setup, browser acceptance and screenshots, final artifact packaging and rights review, and publication authorization. Keep QASPER excerpts, weights and local experiment artifacts out of the public repository unless redistribution rights are established.
