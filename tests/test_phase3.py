from pathlib import Path
from tempfile import TemporaryDirectory
import tomllib
import unittest

from rag_eval.chunking import Chunk
from rag_eval.corpus import paragraphs
from rag_eval.evaluation import score
from rag_eval.evidence_audit import audited_evidence
from rag_eval.fusion import reciprocal_rank_fusion, rerank
from rag_eval.phase3 import Checkpoints, validate_config
from rag_eval.phase3_analysis import paper_bootstrap
from test_phase2 import annotation, paper


class FusionTests(unittest.TestCase):
    def test_fusion_formula_and_tie(self):
        a = [("b", 99), ("c", -2)]
        b = [("a", -9), ("c", 5)]
        expected = [("c", 2 / 62), ("a", 1 / 61), ("b", 1 / 61)]
        self.assertEqual(reciprocal_rank_fusion([a, b], 60, 2), expected)
        self.assertEqual(reciprocal_rank_fusion([b, a], 60, 2), expected)
        self.assertEqual(reciprocal_rank_fusion([a, b], 60, 1), expected[1:])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            reciprocal_rank_fusion([a + a], 60, 4)

    def test_rerank_preserves_candidates_prior_scores_and_tail(self):
        class Model:
            def score(self, query, passages):
                self.inputs = (query, passages)
                return [-3.0, 2.0, 2.0]

        fused = [("a", .5), ("b", .4), ("c", .3), ("d", .2)]
        chunks = {cid: Chunk(cid, "p", cid, 0, 1, cid) for cid, _ in fused}
        model = Model()
        original, reordered = rerank("query", fused, chunks, model, 3)
        self.assertEqual(model.inputs, ("query", ["a", "b", "c"]))
        self.assertEqual(reordered, [("b", 2.0), ("c", 2.0), ("a", -3.0), ("d", .2)])
        self.assertEqual([(r["chunk_id"], r["fusion_score"]) for r in original], fused[:3])
        self.assertEqual([r["fusion_rank"] for r in original], [1, 2, 3])


class EvidenceAuditTests(unittest.TestCase):
    def test_unique_section_expands_conservatively(self):
        record = paper()
        record["full_text"] = [{"section_name": "Some heading", "paragraphs": ["alpha", "beta"]}]
        question = {"answers": [annotation("ann", ["Some  heading"])]}
        chunks = [Chunk("x", "p", "p:0:0", 0, 5, "alpha"), Chunk("y", "p", "p:0:1", 0, 4, "beta")]
        prior, current = audited_evidence("p", record, question, list(paragraphs("p", record)), chunks)
        self.assertEqual(prior.status, "no_complete_text_set")
        self.assertEqual(current.status, "eligible")
        self.assertEqual(current.annotations[0]["labels"][0]["status"], "mapped_section")
        self.assertEqual(current.sets, [{"p:0:0", "p:0:1"}])
        self.assertEqual(score(chunks, current, [1, 2])["annotated_set_coverage@1"], 0)
        self.assertEqual(score(chunks, current, [1, 2])["annotated_set_coverage@2"], 1)
        self.assertEqual(question["answers"][0]["answer"]["evidence"], ["Some  heading"])

    def test_duplicate_empty_and_mixed_sections_remain_unresolved(self):
        record = paper()
        record["full_text"] = [{"section_name": "same", "paragraphs": ["alpha"]},
                               {"section_name": "same", "paragraphs": ["beta"]},
                               {"section_name": "empty", "paragraphs": [" "]},
                               {"section_name": "valid", "paragraphs": ["body"]}]
        question = {"answers": [annotation("ann", ["same", "empty", "valid", "FLOAT SELECTED: x"])]}
        _, gold = audited_evidence("p", record, question, list(paragraphs("p", record)),
                                   [Chunk("x", "p", "p:3:0", 0, 4, "body")])
        self.assertEqual(gold.status, "no_complete_text_set")
        self.assertEqual([r["status"] for r in gold.annotations[0]["labels"]],
                         ["ambiguous_section", "section_without_indexable_body", "mapped_section", "non_text"])

    def test_paragraph_metrics_stay_identical_without_new_labels(self):
        record = paper()
        chunk = Chunk("x", "p", "p:0:0", 0, 10, "alpha beta")
        prior, gold = audited_evidence("p", record, record["qas"][0], list(paragraphs("p", record)), [chunk])
        self.assertEqual(score([chunk], prior, [1, 5]), score([chunk], gold, [1, 5]))


class ExperimentTests(unittest.TestCase):
    def test_config_validates_primary_depth_batch_and_split(self):
        with open("configs/phase3.toml", "rb") as stream:
            config = tomllib.load(stream)
        pilot = {"split": "dev", "cutoffs": [1, 5, 20]}
        validate_config(config, pilot)
        for change in ({"primary_fusion_depth": 30}, {"rerank_depth": 51}, {"rerank_batch": 0}):
            with self.assertRaises(ValueError):
                validate_config(config | change, pilot)
        with self.assertRaises(ValueError):
            validate_config(config, pilot | {"split": "test"})

    def test_checkpoint_replay_and_corruption(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "work.sqlite"
            cache = Checkpoints(path, "fingerprint")
            cache.put("base", "paper", "id", {"ranking": [["x", 1.0]]})
            cache.db.close()
            other = Checkpoints(path, "different")
            self.assertIsNone(other.get("base", "paper", "id"))
            other.db.close()
            cache = Checkpoints(path, "fingerprint")
            self.assertEqual(cache.get("base", "paper", "id"), {"ranking": [["x", 1.0]]})
            with cache.db:
                cache.db.execute("UPDATE work SET payload='{}'")
            with self.assertRaisesRegex(ValueError, "Corrupt checkpoint"):
                cache.get("base", "paper", "id")
            cache.db.close()

    def test_cluster_bootstrap_is_reproducible_and_question_weighted(self):
        pairs = [("p", 1.0), ("p", 1.0), ("q", 0.0)]
        first = paper_bootstrap(pairs, 42, 2000)
        self.assertEqual(first, paper_bootstrap(pairs[::-1], 42, 2000))
        self.assertEqual(first, {"delta": 2 / 3, "ci95": [0.0, 1.0], "papers": 2, "questions": 3})


if __name__ == "__main__":
    unittest.main()
