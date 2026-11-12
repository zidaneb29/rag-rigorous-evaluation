# Phase 4C · Qwen 7B feasibility

## Status

**Sanity gate failed.** The revised binary-wheel notebook loaded Qwen2.5-7B-Instruct on a free Colab Tesla T4 and completed the three synthetic checks. It correctly answered the direct extraction case and declined the missing-information case. It also abstained on the supported inference. The original eight development questions and the independent confirmation set were never run. Full Phase 4 generation evaluation remains incomplete, and generator work is paused.

The original Colab setup spent over 40 minutes across two interrupted CUDA source-build attempts. The completed session used the pinned binary wheel.

## Imported results

| Synthetic case | Expected behavior | Recorded output | Prompt / output tokens | Completion time |
| --- | --- | --- | ---: | ---: |
| Direct extraction | Say the locker is blue and cite P1 | Answered “The locker is blue.” with P1 | 269 / 23 | 1.890 s |
| Simple inference | Say the parcel cannot be collected at 6 pm on Monday and cite P1 | Explicit abstention | 291 / 10 | 0.777 s |
| Missing information | Abstain because the owner's name is absent | Explicit abstention | 272 / 10 | 0.753 s |

All three outputs passed the structured contract and stopped normally. The sole citation resolved and directly supported the color claim. The inference passage says that the counter closes at 5 pm on Mondays and parcels can only be collected while it is open. The abstention therefore missed a supported negative answer. The automated importer verified structure and citation resolution. The semantic judgments above come from inspecting the saved passages and match the user's observations. Two of the three cases met their expected behavior. This synthetic sample only supports a decision about the predefined sanity gate.

| Runtime measurement | Recorded value |
| --- | --- |
| GPU and free memory before setup | Tesla T4, SM 7.5, 14,913 MiB free of 15,360 MiB |
| Driver and environment | NVIDIA 580.82.07, CUDA driver API 13.0, Linux x86_64, CPython 3.12.13 |
| Backend and build | llama-cpp-python 0.3.16, official CUDA 12.4 CPython 3.12 wheel, runtime reports SM 750 support |
| Wheel setup and model load | 92.74 s setup, 4.10 s model load |
| GPU offload probe | Four-token forward pass, notebook process GPU allocation rose from 0 to 4,558 MiB |
| Peak memory | 4,539 MiB device use during load, 4,571 MiB sampled during inference, 4,932,016 KiB process peak RSS |
| Completion time | 3.421 s total, p50 0.777 s, p95 1.779 s across three cases |
| Median end-to-end throughput | 12.86 output tokens/s, including prefill |

The checkpoint's model revision, both shard hashes, wheel digest, and inference settings matched the frozen protocol. The import validated the canonical packet digest, row digests, exact rendered prompts, token bounds, order, and GPU session metadata. The returned file's SHA-256 is `c89f4387598f454a1b01087d781b9ee505774514a3321692d0beb0ed6d0c9cd5`. Its canonical checkpoint digest is `b5ded5d5071a924f04592bd239d49c658356c2d2110b99cf34d481b7113f120b`. The imported copy is structurally identical to the original. The checkpoint contains three sanity rows and no approval to advance. Local validation checked the recorded GPU evidence without replaying inference.

The original [Phase 4 pilot](phase4.md) and [Phase 4B recovery](phase4b.md) remain the measured 1.5B comparisons. Balanced few-shot C produced three clearly correct, supported answers out of six and declined one of two native unanswerables. Any future 7B comparison will be descriptive because model size, hardware, and inference runtime differ.

## Frozen design

Use the official Apache-2.0 Qwen2.5-7B-Instruct Q4_K_M GGUF at revision `293ca9a10157b0e5fc5cb32af8b636a88bede891`. Both shard SHA-256 values remain pinned in [phase4c.toml](../configs/phase4c.toml). The weights total approximately 4.68 GB and download only inside Colab. The inference API remains in-process llama-cpp-python 0.3.16. The installation now uses the official **v0.3.16-cu124** Linux x86_64 wheel for CPython 3.11 or 3.12. Each wheel is about 551 MB and is verified by byte count and SHA-256 using [phase4c_runtime.py](../rag_eval/phase4c_runtime.py). Upstream builds with CUDA architectures set to `all`. The notebook uses binary-only pip commands with timeouts. The source digest in the original packet is historical metadata and is never used for installation. [Qwen release](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/commit/293ca9a10157b0e5fc5cb32af8b636a88bede891) · [CUDA wheel release](https://github.com/abetlen/llama-cpp-python/releases/tag/v0.3.16-cu124) · [Upstream build workflow](https://github.com/abetlen/llama-cpp-python/blob/v0.3.16/.github/workflows/build-wheels-cuda.yaml)

The wheel path changes the native build and CUDA libraries. It leaves model format, quantization, grammar, tokenization method, context limits, full GPU offload, greedy decoding and seed as specified. Each checkpoint records the wheel name and digest, setup time, GPU, driver, CUDA driver API, process GPU allocation, Python and dependency versions, backend build information, and model provenance. The importer rejects old source-build sessions and mixed wheel, GPU, driver or dependency versions within one checkpoint. Keep such results in separate runs. Even a pinned configuration can produce numerical differences across machines. Device memory peaks are sampled whole-device use at approximately 0.5-second intervals. Report process peak RSS separately. Tokens/s is generated tokens divided by completion wall time, including prompt prefill.

Freeze Phase 4B prompt C and the same JSON schema, five-passage cap, 1,250-token evidence allowance, 2,048-token context, 176-token completion limit, greedy decoding, and seed 42. Use the Qwen ChatML text, checked against all eight Phase 4B C rendered prompts and the loaded GGUF template family. The notebook checks all frozen inputs against the live tokenizer before inference. It stops on a budget mismatch. There is no automatic repacking, model substitution, or prompt selection.

| Stage | Sample | Advance only after |
| --- | --- | --- |
| Sanity | Three synthetic cases reused from Phase 4B. A stated color, a closing-hours inference, and a missing name. | Blue and the correct negative inference with P1 support, explicit abstention on the missing name, and three coherent structured records with valid citation IDs. |
| Original diagnostic | Same six reference-evidence questions and two BGE native negatives, with identical source spans and passage IDs. | At least four clearly correct, supported answers out of six, both native negatives declined, and valid structures and citation references throughout. |
| Confirmation | Twelve previously unused answerable questions with reference evidence and three previously unused native negatives with BGE evidence. | Apply the engineering decision criteria below. |

The confirmation sample was selected before generation from the Phase 3 development subset. Seed 4403 shuffles sorted IDs after omitting the original eight. Answerables are cycled across extractive, free-form and yes/no types and BGE top-five annotated-set coverage. Their smallest complete mapped reference set must fit the five-passage cap and a 3,800-character prefilter. The three remaining native negatives use the first five BGE passages. All IDs, ordering, spans, prompts and decoding settings are frozen in the packet. Three native negatives offer limited evidence about abstention. Reference-evidence performance also does not establish performance on naturally retrieved passages.

## Archived execution and import procedure

The following preserves the run protocol for reproducibility. The project has paused generator experiments.

1. Open [notebooks/phase4c.ipynb](../notebooks/phase4c.ipynb) using Colab's notebook upload menu. Under **Runtime → Change runtime type**, choose a **free T4 GPU**. If the default runtime uses Python 3.13, select the **2026.07** runtime version if offered. Google lists Python 3.12.13 for that runtime. The notebook requires Linux x86_64, an eligible Python ABI, a T4 with SM 7.5 and 8 GiB free, and CUDA driver API 12.4 or later. It verifies that the pinned wheel imports and that GPU offload and response grammar are available before weight download. If any check fails, stop and save the error. The wheel path requires no compiler, API key, paid resource, remote control, or web service. [Colab runtime versions](https://research.google.com/colaboratory/runtime-version-faq.html) · [Colab resource limits](https://research.google.com/colaboratory/faq.html)
2. Upload only `artifacts/phase4c/phase4c-packet.json` when prompted. It is 171,831 bytes and contains only the selected excerpts, questions, source coordinates and frozen settings. Keep `review-template.local.json` on the laptop. It contains benchmark references for adjudication. No full-paper corpus, local retrieval index, or private file is needed.
3. After model load, the notebook requests full GPU offload, runs a forward pass on a fixed non-evaluation string, and requires at least 3 GiB of additional GPU memory attributed to the notebook Python process by `nvidia-smi`. If the process cannot be observed, evaluation stops. After review, enable `RUN_EVALUATION` and run **sanity** first. Review the raw output against the table above. Use the notebook's approval field to start **original** only after sanity passes. Review the original outputs with the local rubric and references before approving **confirmation**. The notebook checks structure, citations and abstention automatically and requires explicit stage approval. Factual support requires review. Run each case once, in frozen order.
4. Download `phase4c-checkpoint.json` after each stage and before leaving the session. Each completed record is written atomically. On reset, upload the same packet, enable the resume cell, and upload the downloaded checkpoint. A new runtime records a new hardware session and continues with the first incomplete case. Rows since the last downloaded checkpoint cannot be recovered after a Colab reset.
5. Put the returned checkpoint under `artifacts/phase4c/` and run the import below. Preserve the download as received. The importer verifies packet and row digests, order, stage approvals, prompt text, settings and runtime metadata. It computes automatic status, citation, latency and throughput summaries and leaves the semantic gate pending.

```powershell
python -m rag_eval.phase4c export --output artifacts/phase4c
python -m rag_eval.phase4c import --packet artifacts/phase4c/phase4c-packet.json --results artifacts/phase4c/phase4c-checkpoint.json --output artifacts/phase4c-import
```

## Review and decision

For each output, mark correctness and completeness against compatible references. Check each substantive claim for support or contradiction in the exact visible passages, then assess each cited link. Mark absent, invalid and unsupported citation links separately. Judge abstention against the visible context and native label. Distinguish unclear context from a supported answer or an unsupported factual assertion. Keep reviewer rationale and provenance local. Automatic counts establish parse and citation-reference integrity. Assistant judgments and any independent human review should be labeled separately.

- **Pass** requires at least 8/12 clearly correct and supported confirmation answers, zero unsupported factual answers on native negatives, valid structures and resolving citation identifiers throughout, no systemic citation-support failures, and measured resource use that fits a bounded larger run. These are engineering thresholds.
- **Conditional pass** requires demonstrated semantic reliability in a clearly defined restricted slice and adequate runtime there. Document the failed criterion and restrict any follow-up accordingly.
- **Fail** applies if the model fails sanity, fails to show reliable grounded answering in the bounded development checks, or cannot run within available resources. Review whether the portfolio should emphasize retrieval and evaluation or use different infrastructure before considering another model.

## Local verification

All 39 unit tests passed after import. The frozen packet file retains SHA-256 `0bb3d5d37f35c734b14b63d7d61b2ae629ea4e6234c9b9ff97f04905946c5985` and canonical payload digest `b5e410eaa32b556ef0354cebb3dc6f2ecc8f22a89655c31464482311d1615649`. The original eight contexts and rendered prompts match Phase 4B C exactly. The fifteen confirmation IDs are disjoint from those eight. Prior Phase 2, 3, 4 and 4B artifacts still match their recorded checksums. Imported records and automatic summaries are under `artifacts/phase4c-import/`. No model or wheel was installed locally during import.
