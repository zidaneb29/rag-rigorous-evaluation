import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from rag_eval.chunking import Chunk
from rag_eval.corpus import sha256
from rag_eval.phase5 import digest
from rag_eval.ui_data import (
    Explorer, METHODS, dashboard_snapshot, load_config, load_dashboard, load_snapshot, source_url, stamps,
)


ROOT = Path(__file__).resolve().parents[1]


class UIAdapterTests(unittest.TestCase):
    def setUp(self):
        p = {"title": "Experiments", "abstract": "", "full_text": [
            {"section_name": "Results", "paragraphs": ["Data came from Reuters.", "We compared several methods."]},
        ]}
        papers = {"1901.00001": p, "1902.00002": {"title": "Other", "abstract": "",
                  "full_text": [{"section_name": "Data", "paragraphs": ["Other paper data."]}]}}
        self.chunks = [Chunk(f"{pid}:0:{i}@0:{len(t)}", pid, f"{pid}:0:{i}", 0, len(t), t)
                       for pid, paper in papers.items()
                       for i, t in enumerate(paper["full_text"][0]["paragraphs"])]
        manifest = [{k: v for k, v in vars(c).items() if k != "text"} for c in self.chunks]
        pilot = {"tokenizer_sha256": "fake", "max_tokens": 256, "overlap": 32}
        with patch("rag_eval.ui_data.verify"), patch("rag_eval.ui_data.Tokenizer.from_file"), patch(
            "rag_eval.ui_data.build_chunks", return_value=self.chunks
        ):
            self.engine = Explorer(SimpleNamespace(papers=papers, report={"config": pilot}, manifest=manifest),
                                   Path("model"), Path("cache"),
                                   {"primary_fusion_depth": 50, "rrf_constant": 60, "rerank_depth": 20})
        matrix = np.zeros((3, 384), dtype=np.float32)
        matrix[:, 0] = 1
        self.engine.vectors = matrix
        self.models = []

        def model(kind):
            self.models.append(kind)
            return SimpleNamespace(encode=lambda *a, **kw: matrix[:1],
                                   score=lambda query, passages: list(range(len(passages))))
        self.engine._model = model

    def test_all_methods_respect_scope_and_source_spans(self):
        for method in METHODS:
            result = self.engine.search("Where did the data come from?", "1901.00001", method, 5)
            self.assertEqual(len(result["passages"]), 2)
            self.assertEqual(result["method"], method)
            for row in result["passages"]:
                self.assertEqual(row["paper_id"], "1901.00001")
                self.assertEqual(row["section"], "Results")
                self.assertEqual(row["text"], self.engine.paragraphs[row["paragraph_id"]].text)
                self.assertEqual(row["source"], "https://arxiv.org/abs/1901.00001")
        self.assertEqual(self.models, ["dense", "dense", "dense", "reranked"])
        self.assertIsNone(source_url("https://example.com"))

    def test_ties_are_stable_and_invalid_inputs_are_rejected(self):
        result = self.engine.search("A question", "1901.00001", "dense", 2)
        ids = [p["chunk_id"] for p in result["passages"]]
        self.assertEqual(ids, sorted(ids))
        for args in (("", "1901.00001", "dense", 2), ("q", "missing", "bm25", 2),
                     ("q", "1901.00001", "generate", 2), ("q", "1901.00001", "dense", 21)):
            with self.assertRaises(ValueError):
                self.engine.search(*args)

    def test_config_and_file_stamp(self):
        config = load_config(ROOT / "configs/ui.toml")
        self.assertEqual(config["default_method"], "dense")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "artifact"
            missing = stamps([path])
            path.write_text("version 1")
            before = stamps([path])
            path.write_text("different bytes")
            self.assertNotEqual(before, stamps([path]))
            self.assertNotEqual(missing, before)


class DashboardTests(unittest.TestCase):
    def test_release_snapshot_matches_frozen_artifacts(self):
        config = load_config(ROOT / "configs/ui.toml")
        path = ROOT / config["evaluation_snapshot"]
        digest_value = config["evaluation_snapshot_sha256"]
        saved = load_snapshot(path, digest_value)
        self.assertEqual(saved["report"]["eligibility"]["eligible"], 126)
        self.assertEqual(saved["source_report_sha256"], config["evaluation_report_sha256"])
        with self.assertRaisesRegex(ValueError, "Checksum"):
            load_snapshot(path, "0" * 64)
        original = ROOT / config["evaluation_run"]
        if (original / "report.json").exists():
            validated = load_dashboard(original, config["evaluation_report_sha256"])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")),
                             dashboard_snapshot(validated, config["evaluation_report_sha256"]))

    def test_verified_report_and_corrupted_or_missing_assets(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = {"phase3": {"rerank_depth": 20}}
            population = {"selected_ids": ["p"]}
            status = {"question_id": "q", "paper_id": "p", "status": "eligible"}
            files = {"protocol.json": {"manifest": protocol, "sha256": digest(protocol)},
                     "population.json": {"manifest": population, "sha256": digest(population)},
                     "evidence.json": [status], "chunks.json": [],
                     "rankings.json": [status | {"setting": "paper"}]}
            for name, data in files.items():
                (root / name).write_text(json.dumps(data), encoding="utf-8")
            report_path = root / "report.json"
            report_path.write_text(json.dumps({
                "protocol_sha256": digest(protocol), "population_sha256": digest(population),
                "eligibility": {"eligible": 1},
                "results": {"paper": {"questions": 1, "papers": 1, "eligible": 1}},
                "files": {name: {"sha256": sha256(root / name)} for name in files},
            }), encoding="utf-8")
            expected = sha256(report_path)
            self.assertEqual(load_dashboard(root, expected)["report"]["results"]["paper"]["eligible"], 1)
            (root / "evidence.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "Checksum"):
                load_dashboard(root, expected)
            (root / "evidence.json").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Missing evidence.json"):
                load_dashboard(root, expected)

    def test_streamlit_dashboard_reads_recorded_scope_counts(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("Install requirements-ui.txt for Streamlit integration")
        if not (ROOT / "artifacts/phase5-heldout/report.json").exists():
            self.skipTest("Phase 5 artifacts unavailable")
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=20).run()
        app.radio(key="page").set_value("Evaluation dashboard").run()
        self.assertFalse(app.exception)
        self.assertEqual([m.value for m in app.metric[:4]], ["40", "156", "126", "30"])
        self.assertIn("Recall@10", app.dataframe[0].value.columns)
        paper = list(app.dataframe[0].value["Recall@10"])
        app.radio(key="scope").set_value("split-title").run()
        self.assertFalse(app.exception)
        self.assertNotEqual(paper, list(app.dataframe[0].value["Recall@10"]))

    def test_streamlit_missing_configuration_has_an_error_state(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("Install requirements-ui.txt for Streamlit integration")
        with patch.dict(os.environ, {"QASPER_UI_CONFIG": str(ROOT / "configs/missing-ui.toml")}):
            app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.error)

    def test_streamlit_explorer_keeps_a_result_on_identical_submit(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("Install requirements-ui.txt for Streamlit integration")
        needed = ("data/qasper-train-dev-v0.3.tgz", "artifacts/pilot-cold/report.json", "model/tokenizer.json")
        if any(not (ROOT / path).is_file() for path in needed):
            self.skipTest("Prepared development assets unavailable")
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=20).run()
        pid = app.selectbox(key="paper").value
        app.selectbox(key="method").set_value("bm25")
        app.text_area(key="question").set_value("What data did the authors use?")
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        result = app.session_state["search_result"][1]
        self.assertTrue(result["passages"])
        self.assertTrue(all(r["paper_id"] == pid for r in result["passages"]))
        with patch("rag_eval.ui_data.Explorer.search", side_effect=AssertionError("Duplicate inference")):
            app.button[0].click().run()
        self.assertFalse(app.error)
        self.assertEqual(app.session_state["search_result"][1], result)


if __name__ == "__main__":
    unittest.main()
