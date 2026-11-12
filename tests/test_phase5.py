import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rag_eval.phase3 import Checkpoints
from rag_eval.phase5 import (
    METHODS, METRICS, cached_or_query, choose_papers, digest, duplicate_audit, publish, summarize,
)


def paper(title, text):
    return {"title": title, "full_text": [{"paragraphs": text}]}


class Phase5Tests(unittest.TestCase):
    def test_audit_and_sampling_are_label_independent(self):
        train = {"duplicate": paper("First scientific paper", ["source paragraph"])}
        test = {"version": paper("First SCIENTIFIC Paper", ["source paragraph"]),
                "new": paper("Unrelated topic", ["other material"])}
        audit = duplicate_audit([("train", train), ("test", test)])
        self.assertEqual(audit[0]["reasons"], ["title", "body"])
        self.assertEqual(choose_papers(test, audit, 4, 42), ["new"])
        test["version"]["qas"] = [{"answers": ["ignored"]}]
        self.assertEqual(duplicate_audit([("train", train), ("test", test)]), audit)

    def test_frozen_artifact_is_order_independent_and_write_protected(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "lock.json"
            a = {"one": 1, "two": 2}
            b = {"two": 2, "one": 1}
            self.assertEqual(digest(a), digest(b))
            publish(path, a)
            publish(path, b)
            with self.assertRaisesRegex(ValueError, "Frozen"):
                publish(path, {"one": 2})
            self.assertEqual(json.loads(path.read_text()), a)

    def test_checkpoint_reuse_does_not_recompute(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "work.sqlite"
            first = Checkpoints(path, "scope")
            self.assertEqual(cached_or_query(first, "base", "paper", "q", lambda: [1]), ([1], False))
            first.db.close()
            replay = Checkpoints(path, "scope")
            self.assertEqual(cached_or_query(replay, "base", "paper", "q", self.fail), ([1], True))
            replay.db.close()

    def test_paired_population_and_metric_means(self):
        rules = {"settings": ["paper"], "bootstrap_seed": 42, "bootstrap_draws": 2000}
        def row(paper_id, status, values):
            metrics = {m: dict.fromkeys(METRICS, values[i]) if status == "eligible" else {}
                       for i, m in enumerate(METHODS)}
            return {"paper_id": paper_id, "status": status, "setting": "paper", "metrics": metrics,
                    "paragraph_only": metrics, "timing_ms": dict.fromkeys((*METHODS, "rerank_only"), 2)}
        rows = [row("a", "eligible", [0, 1, 1, 1]), row("b", "eligible", [1, 1, 1, 0]),
                row("b", "unanswerable", [0, 0, 0, 0])]
        report = summarize(rows, rules)["paper"]
        self.assertEqual(report["eligible"], 2)
        self.assertEqual(report["questions"], 3)
        self.assertEqual(report["metrics"]["dense"]["recall@10"], {"mean": 1, "n": 2})
        self.assertEqual(report["paired"]["dense-bm25"]["recall@10"],
                         {"delta": .5, "ci95": [0, 1], "papers": 2, "questions": 2})
        self.assertEqual(report, summarize(rows, rules)["paper"])


if __name__ == "__main__":
    unittest.main()
