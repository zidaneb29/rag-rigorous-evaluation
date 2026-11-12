# Phase 4 generation feasibility

## Decision

The local backend passed the throughput and structured-output checks and failed the grounding gate. It returned the same explicit abstention on all eight prompts, including all six answerable questions supplied with complete mapped reference evidence. Phase 4 stopped at feasibility. The larger paired experiment, evidence-removal runs, and downstream quality comparison were not launched. The held-out test data remained untouched.

The gate required eight completed rows, at least seven parseable citation-safe answers or abstentions, at least four answered reference cases, and median decode throughput of at least two tokens/s. The observed counts were eight parseable rows, zero answered reference cases, and 15.01 median tokens/s. The answer-count gate was a minimum capability check. It did not require or imply human-confirmed correctness.

## Model and runtime

| Item | Pilot configuration |
| --- | --- |
| Model | Official Qwen2.5-1.5B-Instruct GGUF, Apache-2.0, Q4_K_M, revision `91cad51170dc346986eccefdc2dd33a9da36ead9` |
| Model download | 1,117,320,736 bytes. SHA-256 `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e` |
| Runtime | llama.cpp b7300, commit `2960eb297`, official Windows x64 CPU ZIP, 17,005,418 bytes |
| Runtime archive SHA-256 | `f1e8cfe9b1e87371c6fcce57443f7425fae7fa8128ad0f58bf930627f4c4ae39` |
| Hardware | Intel i5-6300HQ, four CPU and batch threads, Haswell/AVX2 backend, zero GPU layers |
| Token limits | 2,048 total context, 1,250 additional evidence tokens, five whole passages, 176 generated tokens |
| Decoding | Greedy temperature 0, seed 42, structured JSON schema, prompt cache disabled per request |

The runner made one request at a time. The runtime log reported four internal slots and a 2,048-token context. The download also included small public release metadata and a 1 MiB transfer probe. Existing NumPy and psutil dependencies were reused. Service cost was **$0**. [Official GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/blob/91cad51170dc346986eccefdc2dd33a9da36ead9/qwen2.5-1.5b-instruct-q4_k_m.gguf) · [llama.cpp b7300](https://github.com/ggml-org/llama.cpp/releases/tag/b7300)

## Pilot and generation records

The runner verified the Phase 3 development-ranking, chunk and evidence-manifest hashes, then reconstructed the exact raw source spans. Seed 42 shuffled sorted question IDs. The six reference cases included two extractive, two free-form, and two yes/no questions, stratified by whether BGE retrieved a complete annotated set in its first five chunks. For each, the reference builder selected the fully mapped annotation using the fewest chunks, then least source text, then stable chunk IDs. Candidates whose selected reference needed more than five chunks were outside this pilot. The two native-unanswerable cases used BGE passages. The pilot contained eight unique questions and eight question–condition responses.

The generator adapter renders the model chat template and counts its exact tokens with llama.cpp. It passes those token IDs directly to inference. Context packing keeps whole passages in candidate order, logs skipped passage IDs and reasons, and fails an incomplete reference condition explicitly. Each visible passage has a local citation ID and an audited paper, section, paragraph and character span. The answer parser records raw outputs, abstentions, and malformed or invalid citations. Resolved citation IDs are restricted to passages visible in that prompt. A transactional SQLite checkpoint stores each completed question–condition result, with input and record hashes for replay.

The gated study scaffold defines BGE, RRF + MiniLM, and reference context under identical prompts, decoding and token budgets. It would select a disjoint development sample and reserve separate native-unanswerable and empty-context diagnostics. The reference arm measures behavior with annotated evidence and can still fail when annotations are incomplete or the model ignores evidence. It provides a diagnostic reference point.

## Measured results

| Pilot group | Questions | Explicit abstentions | Complete annotated sets packed | Context build p50 / p95 | Generation p50 / p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference, answerable | 6 | 6 | 6 | 9.80 / 14.87 ms | 4.59 / 7.03 s |
| BGE, native unanswerable | 2 | 2 | Outside evidence-metric eligibility | 41.89 / 49.13 ms | 13.05 / 13.08 s |

All eight completions stopped normally at EOS after 11 tokens. The reference prompts were 187–453 tokens, and the two BGE prompts were 831 and 886. Client and runtime prompt counts matched in every record. There was no runtime truncation. Explicit abstention was 6/6 on benchmark-answerable questions and 2/2 on native unanswerables. The latter two results provide little evidence of selectivity given the uniform response. The pilot yielded zero factual claims and zero citation links. Citation support and claim faithfulness have no denominator, and there is no measured BGE-versus-reranking downstream comparison.

One clear missed opportunity was the extractive question `45a5961a4e…` about the number of question types. Its sole visible passage on paper `1801.02073` directly says there are seven types. The output abstained without generating a claim. The two BGE responses to native negatives used the expected abstention form, but a human reviewer has yet to assess whether either supplied context could justify an answer. These examples illustrate the saved records and are not independent annotations.

The backend became ready in **2.16 seconds** on the successful invocation, excluding the GGUF checksum and runtime-version check. The server peaked at **1,797.25 MiB (1.76 GiB)** working set. The eight completion calls took **54.81 seconds** in total. Median decode throughput was **15.01 tokens/s** for short abstentions. This does not estimate throughput for longer technical answers. For the two BGE rows, saved Phase 3 retrieval latency had a 24.17 ms median. Adding the previously measured retrieval duration to current context construction and generation gave a **13.12 s** composed median. The pipeline did not rerun retrieval and generation in one timed call. Loading time is reported separately from all conditions. Generation records and checkpoints occupied 41,182 and 57,344 bytes, and the blinded review packet occupied 17,180 bytes.

The first launch loaded the model but its client could not connect over sandboxed loopback and timed out before recording a response. The successful run used localhost access with the same model and configuration. No inference was sent to a remote service.

## Human review and failure attribution

The shuffled review packet contains all eight responses, exact visible passages, and original answer references. The separate review key maps opaque IDs to conditions. No independent human judgments have been entered. Use the following rubric before inspecting the key.

- **Correctness.** Mark `correct`, `partial`, `incorrect`, `abstained`, or `unjudgeable`. For exact-matchable answers accept semantically equivalent values, names and yes/no phrasing. For free-form responses evaluate the required technical facts and material errors. Lexical overlap alone is insufficient.
- **Context sufficiency.** Mark `sufficient`, `insufficient`, or `uncertain` based on whether the visible passages establish the answer. A complete mapped annotation and sufficient visible context are distinct. Record why an answerable question was appropriately or inappropriately declined.
- **Claim faithfulness.** In output order, mark each substantive claim `supported`, `unsupported`, `contradicted`, or `uncertain`. A contradiction requires evidence against the claim. Record the supporting or contradictory passage.
- **Citation support.** In output order, mark each asserted claim–citation link `supporting`, `joint`, `unsupported`, `contradicted`, `invalid`, or `uncertain`. A joint link must be necessary alongside another cited passage. A claim with no asserted link is a missing citation. Syntactic resolution does not establish factual support.
- **Abstention.** Keep native unanswerability distinct from deliberate context removal. For benchmark-answerable prompts, interpret a decline alongside visible context sufficiency. Keep unsupported answers separate from abstentions and output errors.

Store reviewer rows as a JSON array with keys `review_id`, `correctness`, `context_sufficiency`, `claims` and `citation_links`. The last two are ordered lists of rubric labels and are empty for an abstention. Keep reviewer rationales separately or as an optional `notes` field. A second independent reviewer is optional. Single-reviewer results provide no inter-rater estimate. The review tool records deterministic parse and citation checks separately, and computes paired correctness and nDCG–correctness association only after labels are supplied. The current results support neither a downstream association nor a causal conclusion. In a future study, the records distinguish annotated evidence outside the recorded ranking, evidence retrieved but excluded by packing, and annotated evidence packed. Human judgment is needed to establish whether the model ignored sufficient evidence or whether a citation supported its claim.

## Verification and reproduction

All **30 unit tests passed**, including the 20 Phase 2 and 3 tests. New checks cover budget and rank order, source mapping, citation IDs, parsing, version validation, deterministic sampling and review manifests, record serialization, and checkpoint recovery across a simulated interruption. The real pilot replay recovered all eight completed records without starting llama-server. The artifact audit verified both earlier phases' recorded hashes, the saved prompt lengths and budgets, stable local citation IDs, and zero runtime truncation. A fresh inference repeat and a live mid-generation interruption were outside the verification.

Start from the [Phase 2 environment](phase2.md) and [Phase 3 run](phase3.md). Download the pinned official GGUF as `model/qwen/model.gguf` and unpack the pinned Windows CPU ZIP into `model/qwen/runtime`. Verify the archive SHA-256 above. The runner verifies the model hash and runtime version. The server binds only to `127.0.0.1`.

```powershell
& $pilotPython -m unittest discover -s tests -v
& $pilotPython -m rag_eval.phase4 pilot --output artifacts/phase4-pilot-reproduction
& $pilotPython -m rag_eval.phase4_review artifacts/phase4-pilot-reproduction
# Optional after a reviewer has completed the packet:
& $pilotPython -m rag_eval.phase4_review artifacts/phase4-pilot-reproduction --labels path/to/labels.json
```

Reusing the output directory resumes completed rows. Use a fresh directory to repeat actual inference. Greedy decoding and a fixed seed do not guarantee bitwise equivalence across runtimes or hardware. Raw source excerpts, checkpoints, logs and weights stay in ignored local directories. The record does not identify whether the prompt, schema, model capacity, quantization or their interaction caused uniform abstention.

## Recommendation for review

Before Phase 5, review the eight pilot packets and authorize a small follow-up feasibility check if desired. Compare the same checkpoint with a neutral prompt or schema that gives an answered example as well as an abstention example. Freeze the success criterion and keep these calibration cases out of the subsequent paired sample. If local quality remains inadequate, consider a larger openly licensed model on a free Colab GPU session, with pinned revisions and resumable exports. Free GPU access is conditional. Keep the paired comparison and broader reliability evaluation gated on a pilot that produces supported answers as well as selective abstentions.
