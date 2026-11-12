"""Local evidence retrieval and a read-only QASPER evaluation dashboard."""

import os
from pathlib import Path
from time import perf_counter

import altair as alt
import streamlit as st

from rag_eval.ui_data import (
    DASHBOARD_FILES, METHODS, SCOPES, DevelopmentCatalog, Explorer, load_config, load_dashboard, load_snapshot,
    read_toml, stamps,
)


ROOT = Path(__file__).resolve().parent
TITLE = "Evidence Retrieval and Evaluation on QASPER"
METRICS = {"recall@1": "Recall@1", "recall@5": "Recall@5", "recall@10": "Recall@10",
           "mrr@20": "MRR@20", "ndcg@5": "nDCG@5"}
PATTERNS = {
    "late_fused_evidence": "Evidence reached ranks 6–20",
    "dense_better": "BGE improved Recall@10 over BM25",
    "sparse_better": "BM25 improved Recall@10 over BGE",
    "fusion_gain": "Fusion beat both baselines",
    "fusion_loss": "Fusion fell below the stronger baseline",
    "rerank_gain": "Reranking improved nDCG@5",
    "rerank_loss": "Reranking lowered nDCG@5",
    "missed_fused_pool": "Fused pool missed all annotated paragraphs",
}


@st.cache_resource(max_entries=1, show_spinner=False)
def catalog_resource(data_dir, run_dir, source_stamp):
    return DevelopmentCatalog(data_dir, run_dir)


@st.cache_resource(max_entries=1, show_spinner=False)
def explorer_resource(source_stamp, index_stamp, model_dir, vector_dir, retrieval_path, _catalog):
    return Explorer(_catalog, model_dir, vector_dir, read_toml(retrieval_path))


@st.cache_resource(max_entries=1, show_spinner=False)
def dashboard_resource(directory, expected, evaluation_stamp):
    return load_dashboard(directory, expected)


@st.cache_resource(max_entries=1, show_spinner=False)
def snapshot_resource(path, expected, evaluation_stamp):
    return load_snapshot(path, expected)


def display_error(error):
    st.error(str(error))
    st.info("Restore the local assets or follow the preparation steps in docs/phase6.md.")


def explorer(config):
    st.header("Research paper explorer")
    st.caption("Search a selected development paper and inspect its ranked passages.")
    data, model, run, vectors = (ROOT / config[key] for key in
                                 ("data", "model", "development_run", "development_vectors"))
    source_stamp = stamps([data / "qasper-train-dev-v0.3.tgz", run / "report.json", run / "chunks.json"])
    try:
        catalog = catalog_resource(data, run, source_stamp)
        retrieval_path = ROOT / "configs/phase3.toml"
        index_stamp = stamps([model / "tokenizer.json", model / "model.onnx", model / "minilm/model.onnx",
                              model / "minilm/tokenizer.json", retrieval_path,
                              vectors / catalog.report["cache"]["file"]])
    except (OSError, ValueError, KeyError) as error:
        display_error(error)
        return
    st.caption(f"{len(catalog.papers)} development papers · Paper-scoped search · Local CPU")
    paper_ids = sorted(catalog.papers, key=lambda pid: (catalog.papers[pid]["title"].casefold(), pid))
    with st.form("search"):
        paper_id = st.selectbox("Research paper", paper_ids, key="paper",
                                format_func=lambda pid: f"{catalog.papers[pid]['title']} · {pid}")
        question = st.text_area("Question", max_chars=1000, height=82, key="question",
                                placeholder="What data did the authors use to evaluate their method?")
        method_col, depth_col = st.columns([2, 1])
        method = method_col.selectbox("Retrieval method", list(METHODS), key="method",
                                      index=list(METHODS).index(config["default_method"]),
                                      format_func=METHODS.get,
                                      help="BGE-small is the recommended default. MiniLM reranks up to 20 candidates and is slower.")
        top_k = depth_col.number_input("Top-k", min_value=1, max_value=config["max_top_k"],
                                       value=config["default_top_k"], key="top_k")
        submitted = st.form_submit_button("Search evidence", type="primary")
    request_key = (source_stamp, index_stamp, question.strip(), paper_id, method, top_k)
    saved = st.session_state.get("search_result")
    if submitted and (saved is None or saved[0] != request_key):
        st.session_state.pop("search_result", None)
        try:
            if not question.strip():
                raise ValueError("Enter a question to search the selected paper.")
            with st.spinner("Searching the selected paper…"):
                started = perf_counter()
                engine = explorer_resource(source_stamp, index_stamp, model, vectors, retrieval_path, catalog)
                result = engine.search(question, paper_id, method, top_k)
                result["interaction_ms"] = (perf_counter() - started) * 1000
            st.session_state["search_result"] = (request_key, result)
        except Exception as error:
            display_error(error)
            return
    saved = st.session_state.get("search_result")
    if saved is None or saved[0] != request_key:
        st.info("Select a paper and submit a question to see ranked evidence.")
        return
    result = saved[1]
    st.subheader("Ranked evidence")
    c1, c2, c3 = st.columns(3)
    c1.metric("Passages", len(result["passages"]))
    c2.metric("Retrieval", f"{result['query_ms']:.2f} ms")
    c3.metric("Search request", f"{result['interaction_ms']:.0f} ms")
    st.caption(f"{METHODS[method]} · Top {top_k} · Paper {paper_id}. Retrieval measures search and scoring. "
               "Search request also includes any index, cache, and model initialization. "
               "Scores use method-specific scales. Passages are candidates for inspection.")
    if method == "reranked":
        st.caption("MiniLM scores the first 20 fused candidates. Any remaining passages retain their RRF order.")
    if not result["passages"]:
        st.info("The selected paper has no indexed passages.")
    for passage in result["passages"]:
        with st.expander(f"{passage['rank']:02d} · {passage['section']} · "
                         f"{passage['score_label']} {passage['score']:.4f}"):
            st.text(passage["text"])
            st.caption(passage["title"])
            st.code(f"Paper {passage['paper_id']} · Paragraph {passage['paragraph_id']}\n"
                    f"Chunk {passage['chunk_id']} · Characters {passage['start']}–{passage['end']}", language=None)
            if passage["source"]:
                st.link_button("Open source paper", passage["source"])


def score_rows(group, metric):
    return [{"method": METHODS[method], "score": group["metrics"][method][metric]["mean"]}
            for method in METHODS]


def chart(records):
    return alt.Chart(alt.Data(values=records))


def quality(group):
    selected = st.selectbox("Ranking metric", list(METRICS), index=2, format_func=METRICS.get)
    bar = chart(score_rows(group, selected)).mark_bar(color="#00766e", cornerRadiusEnd=4).encode(
        x=alt.X("score:Q", title=METRICS[selected], scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("method:N", title=None, sort=list(METHODS.values())),
        tooltip=[alt.Tooltip("method:N", title="Method"), alt.Tooltip("score:Q", format=".3f")],
    ).properties(height=180)
    st.altair_chart(bar, width="stretch")
    st.dataframe([{"Method": METHODS[m], **{name: round(group["metrics"][m][key]["mean"], 3)
                                           for key, name in METRICS.items()}} for m in METHODS],
                 hide_index=True, width="stretch")
    st.caption("Recall@k measures the fraction of annotated paragraphs completed by the first k chunks. "
               "MRR@20 uses the first completed paragraph’s reciprocal rank. "
               "nDCG@5 discounts newly completed paragraphs by their rank.")


def paired(group, protocol):
    metric = st.selectbox("Difference metric", list(METRICS), index=2, format_func=METRICS.get)
    rows = []
    for key, values in group["paired"].items():
        left, right = key.split("-")
        v = values[metric]
        rows.append({"contrast": f"{METHODS[left]} − {METHODS[right]}", "delta": v["delta"],
                     "lower": v["ci95"][0], "upper": v["ci95"][1]})
    y = alt.Y("contrast:N", title=None, sort=[r["contrast"] for r in rows],
              axis=alt.Axis(labelLimit=300, labelPadding=8, labelFontSize=12))
    source = chart(rows)
    line = source.mark_rule(color="#00766e", strokeWidth=2).encode(x="lower:Q", x2="upper:Q", y=y)
    points = source.mark_point(color="#00766e", filled=True, size=85).encode(
        x=alt.X("delta:Q", title=f"Paired {METRICS[metric]} difference"), y=y,
        tooltip=[alt.Tooltip(f"{key}:Q", format="+.3f") for key in ("delta", "lower", "upper")],
    )
    zero = chart([{"zero": 0}]).mark_rule(color="#657772", strokeDash=[4, 4]).encode(x="zero:Q")
    st.altair_chart((zero + line + points).properties(height=244), width="stretch")
    st.dataframe([{"Contrast": r["contrast"], "Difference": round(r["delta"], 3),
                   "95% lower": round(r["lower"], 3), "95% upper": round(r["upper"], 3)}
                  for r in rows], hide_index=True, width="stretch")
    rules = protocol["rules"]
    st.caption(f"95% percentile intervals · {rules['bootstrap_draws']:,} paired paper-cluster draws · "
               f"Seed {rules['bootstrap_seed']} · {group['eligible_papers']} eligible papers. "
               "Each sampled paper contributes all its paired eligible questions.")


def efficiency(group, report):
    rows = [{"method": METHODS[m], "median_ms": group["latency_ms_p50_p95"][m][0],
             "p95_ms": group["latency_ms_p50_p95"][m][1],
             "recall": group["metrics"][m]["recall@10"]["mean"]} for m in METHODS]
    data = chart(rows)
    axes = {"x": alt.X("median_ms:Q", title="Warm latency (ms)", scale=alt.Scale(type="log"),
                       axis=alt.Axis(tickCount=5, format="~g", labelFontSize=12, titlePadding=8)),
            "y": alt.Y("recall:Q", title="Recall@10", scale=alt.Scale(domain=[0, 1]),
                       axis=alt.Axis(values=[0, .25, .5, .75, 1], format=".2f",
                                     labelFontSize=12, titlePadding=8)),
            "color": alt.Color("method:N", sort=list(METHODS.values()), scale=alt.Scale(
                domain=list(METHODS.values()), range=["#1a6e8e", "#cb5d2d", "#487438", "#81478f"]),
                               legend=alt.Legend(title=None, orient="bottom", columns=2,
                                                 labelLimit=160, labelFontSize=12))}
    span = data.mark_rule().encode(**axes, x2="p95_ms:Q")
    marks = data.mark_point(filled=True, size=100).encode(
        **axes, tooltip=["method:N", alt.Tooltip("median_ms:Q", format=".2f"),
                         alt.Tooltip("p95_ms:Q", format=".2f"), alt.Tooltip("recall:Q", format=".3f")])
    st.altair_chart((span + marks).properties(height=310), width="stretch")
    st.caption("Latency uses a log scale. Points show medians and lines extend to p95. Warm timings include query encoding, "
               "fusion and reranking where applicable. Setup and model loading are measured separately.")
    st.dataframe([{"Method": r["method"], "Median (ms)": round(r["median_ms"], 2),
                   "p95 (ms)": round(r["p95_ms"], 2)} for r in rows], hide_index=True, width="stretch")
    cards = st.columns(3)
    cards[0].metric("Vector build", f"{report['timings']['embedding_and_vector_index']['wall_s']:.2f} s")
    cards[1].metric("Vector storage", f"{report['vector_cache_bytes'] / 1e6:.2f} MB")
    cards[2].metric("Peak working set", f"{report['peak_working_set'] / 2**20:.1f} MiB"
                    if report["peak_working_set"] is not None else "Unavailable")


def cases(data):
    sample = data["report"]["failure_sample"]
    pattern = st.selectbox("Paper-scoped outcome", list(sample), format_func=PATTERNS.get)
    item = sample[pattern]
    st.caption(f"{item['count']} eligible questions matched this pattern. Examples were sampled with "
               "the frozen seed. Annotations may omit valid evidence.")
    if item["question_id"] is None:
        st.info("No eligible question matched this pattern in the recorded run.")
        return
    qid = item["question_id"]
    row = data["rankings"]["paper", qid]
    evidence = data["evidence"][qid]
    targets = set().union(*map(set, evidence["sets"]))
    st.code(f"Paper {item['paper_id']}\nQuestion {qid}", language=None)
    st.dataframe([{"Method": METHODS[m], "Recall@10": round(row["metrics"][m]["recall@10"], 3),
                   "nDCG@5": round(row["metrics"][m]["ndcg@5"], 3),
                   "Annotated paragraph windows in top 20": ", ".join(
                       str(i) for i, (cid, _) in enumerate(row["rankings"][m][:20], 1)
                       if data["chunks"][cid]["paragraph_id"] in targets) or "—"}
                  for m in METHODS], hide_index=True, width="stretch")
    with st.expander("Annotated evidence references"):
        for i, group in enumerate(evidence["sets"], 1):
            st.code(f"Evidence set {i}\n" + "\n".join(group), language=None)
        st.caption("Window ranks can show partial coverage. Recall requires the full annotated paragraph span.")


def dashboard(config):
    st.header("Evaluation dashboard")
    directory = ROOT / config["evaluation_run"]
    signature = stamps([directory / name for name in DASHBOARD_FILES])
    try:
        with st.spinner("Validating saved evaluation artifacts…"):
            if signature[0][1] is not None:
                data = dashboard_resource(directory, config["evaluation_report_sha256"], signature)
            else:
                snapshot = ROOT / config["evaluation_snapshot"]
                data = snapshot_resource(snapshot, config["evaluation_snapshot_sha256"], stamps([snapshot]))
    except (OSError, ValueError, KeyError, TypeError) as error:
        display_error(error)
        return
    report = data["report"]
    scope = st.radio("Retrieval scope", list(SCOPES), format_func=SCOPES.get, horizontal=True, key="scope")
    group = report["results"][scope]
    st.caption("Frozen QASPER v0.3 held-out sample. Paper scope supplies the target paper. "
               "Title-cued retrieval searches only the sampled candidate pool.")
    if signature[0][1] is None:
        st.caption("Source: bundled text-free snapshot of the frozen report.")
    cards = st.columns(4)
    for col, label, value in zip(cards, ("Sampled papers", "Questions", "Eligible", "Excluded"),
                                 (group["papers"], group["questions"], group["eligible"],
                                  group["questions"] - group["eligible"])):
        col.metric(label, value)
    reasons = report["eligibility"]
    st.caption(f"Excluded from quality metrics · {reasons.get('no_complete_text_set', 0)} incomplete text sets · "
               f"{reasons.get('answerability_tie', 0)} answerability ties · {reasons.get('unanswerable', 0)} unanswerable. "
               "All questions contribute to latency.")
    with st.expander("Evaluation configuration"):
        st.caption(f"Protocol {report['protocol_sha256']} · Population {report['population_sha256']}")
        fusion = data["protocol"]["phase3"]
        st.text(f"RRF depth {fusion['primary_fusion_depth']} · RRF constant {fusion['rrf_constant']} · "
                f"MiniLM pool {fusion['rerank_depth']}")
    for tab, function, args in zip(
        st.tabs(["Retrieval quality", "Paired differences", "Efficiency", "Failure analysis"]),
        (quality, paired, efficiency, cases),
        ((group,), (group, data["protocol"]), (group, report), (data,)),
    ):
        with tab:
            function(*args)


def main():
    st.set_page_config(page_title=TITLE, layout="wide")
    st.html("""<style>
      .block-container {max-width: 1120px; padding-top: 2.8rem;}
      [data-testid="stText"] {white-space: pre-wrap; overflow-wrap: anywhere;}
    </style>""")
    st.title(TITLE)
    st.caption("Inspect technical evidence and compare retrieval quality, uncertainty, and CPU cost.")
    try:
        config = load_config(Path(os.environ.get("QASPER_UI_CONFIG", ROOT / "configs/ui.toml")))
    except (OSError, ValueError, KeyError) as error:
        display_error(error)
        return
    page = st.sidebar.radio("View", ["Paper explorer", "Evaluation dashboard"], key="page")
    st.sidebar.caption("QASPER v0.3 · Offline retrieval")
    if page == "Paper explorer":
        explorer(config)
    else:
        dashboard(config)


main()
