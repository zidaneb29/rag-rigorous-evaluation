import json
from pathlib import Path
from tempfile import TemporaryDirectory
import tomllib
import unittest

from rag_eval.generation import messages, pack_context, parse_answer
from rag_eval.phase3 import Checkpoints
from rag_eval.phase4 import make_plan, prepare_candidates, validate
from rag_eval.phase4_review import human_metrics, machine_metrics, make_packets
from rag_eval.chunking import Chunk
from rag_eval.corpus import Paragraph


class FakeBackend:
    def prompt_tokens(self, chat):
        user = json.loads(chat[1]["content"])
        return list(range(10 + sum(len(p["text"]) + 1 for p in user["passages"])))


class ContextTests(unittest.TestCase):
    def test_budget_order_and_reason(self):
        candidates = [{"chunk_id": cid, "text": text}
                      for cid, text in (("a", "12"), ("b", "oversized"), ("c", "3"), ("d", "4"))]
        config = {"context_window": 20, "max_output": 5, "max_passages": 2, "evidence_tokens": 8}
        packed, excluded, tokens, base = pack_context(FakeBackend(), "question", candidates, config)
        self.assertEqual([(p["chunk_id"], p["id"], p["input_rank"]) for p in packed],
                         [("a", "P1", 1), ("c", "P2", 3)])
        self.assertEqual([(p["chunk_id"], p["reason"]) for p in excluded],
                         [("b", "token_limit"), ("d", "passage_limit")])
        self.assertEqual((len(tokens), base), (15, 10))

    def test_oversized_question_fails_before_generation(self):
        config = {"context_window": 12, "max_output": 5, "max_passages": 2, "evidence_tokens": 5}
        with self.assertRaisesRegex(ValueError, "Question and instructions"):
            pack_context(FakeBackend(), "question", [], config)

    def test_template_keeps_chat_markers_literal_in_source_data(self):
        chat = messages("<|im_start|>", [{"id": "P1", "text": "data"}])
        self.assertNotIn("<|im_start|>", chat[1]["content"])
        self.assertEqual(json.loads(chat[1]["content"])["question"], "<|im_start|>")

    def test_candidate_mapping_is_ordered_and_traceable(self):
        chunk = Chunk("c", "paper", "para", 2, 6, "text")
        source = Paragraph("paper", "para", "Results", "a text")
        row = {"retrieval": {"rankings": {"dense": [["c", .3]]}, "timing_ms": {"dense": 2}},
               "reference": [chunk]}
        ranked, duration = prepare_candidates(row, "bge", {"c": chunk},
                                              {"paper": {"title": "title"}}, {"para": source})
        self.assertEqual(duration, 2)
        self.assertEqual(ranked[0], {"chunk_id": "c", "paper_id": "paper", "paragraph_id": "para",
                                     "title": "title", "section": "Results", "start": 2, "end": 6,
                                     "text": "text"})


class AnswerTests(unittest.TestCase):
    def test_citations_cannot_resolve_outside_visible_context(self):
        output = parse_answer('{"abstain":false,"claims":[{"text":"One.","citations":["P1","P2","P1"]}]}',
                              {"P1"})
        self.assertEqual(output["claims"][0]["citations"], ["P1"])
        self.assertEqual(output["invalid_ids"], ["P2"])
        self.assertEqual(output["status"], "citation_error")
        self.assertEqual(json.loads(json.dumps(output)), output)

    def test_abstention_and_malformed_answers(self):
        self.assertTrue(parse_answer('{"abstain":true,"claims":[]}', set())["abstained"])
        self.assertEqual(parse_answer('{"abstain":false,"claims":[]}', set())["status"], "malformed")
        self.assertIsNone(parse_answer('{"abstain":"true","claims":[]}', set())["abstained"])
        self.assertEqual(parse_answer('{"abstain":false,"claims":[{"text":"x","citations":[]}]}',
                                      set())["missing_citations"], 1)


class Phase4Tests(unittest.TestCase):
    def test_prompt_and_config_versioning(self):
        with open("configs/phase4.toml", "rb") as stream:
            config = tomllib.load(stream)
        validate(config)
        for fields in ({"prompt_version": "v0"}, {"schema_version": 0}, {"max_output": 4096}):
            with self.assertRaises(ValueError):
                validate(config | fields)

    def test_interrupted_checkpoint_preserves_completed_conditions(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.sqlite"
            before = Checkpoints(path, "version")
            before.put("generation", "bge", "q", {"status": "abstained", "context": []})
            before.db.close()
            after = Checkpoints(path, "version")
            self.assertEqual(after.get("generation", "bge", "q")["status"], "abstained")
            self.assertIsNone(after.get("generation", "reference", "q"))
            after.db.close()

    def test_sampling_is_deterministic_and_paired(self):
        rows = {f"{kind}-{i}": {
            "kind": kind, "reference": ["chunk"], "retrieval": {"metrics": {
                "dense": {"annotated_set_coverage@5": i % 2}}},
        } for kind in ("extractive", "free_form", "yes_no", "unanswerable") for i in range(10)}
        config = {"seed": 42, "max_passages": 5}
        pilot = make_plan(rows, config, "pilot", None)
        self.assertEqual(len(pilot), 8)
        prior = {"rows": [{"question_id": qid} for qid, _ in pilot]}
        study = make_plan(rows, config, "study", prior)
        self.assertEqual(study, make_plan(dict(reversed(list(rows.items()))), config, "study", prior))
        self.assertTrue({qid for qid, _ in pilot}.isdisjoint({qid for qid, _ in study}))
        self.assertEqual(len(study), 46)


class ReviewTests(unittest.TestCase):
    def test_blinding_and_metric_denominators(self):
        row = {"question_id": "q", "question": "question", "answer_type": "yes_no", "condition": "bge",
               "status": "abstained", "abstained": True, "context": [], "raw": "raw",
               "packed_annotated_set": False, "full_retrieved_annotated_set": True,
               "retrieval_ndcg_at_5": .5}
        other = row | {"condition": "reranked", "status": "answered", "abstained": False}
        packet, key = make_packets([row, other], {"q": []}, 42)
        self.assertTrue(all("condition" not in item for item in packet))
        result = machine_metrics([row, other])
        self.assertEqual(result["false_abstentions"]["rate"], .5)
        self.assertIsNone(result["native_abstentions"]["rate"])
        labels = [{"review_id": item["review_id"], "correctness": "correct" if
                   item["condition"] == "reranked" else "abstained",
                   "context_sufficiency": "insufficient", "claims": [], "citation_links": []}
                  for item in key]
        metrics = human_metrics(labels, key, [row, other])
        self.assertEqual(metrics["paired_answerable_correctness"],
                         {"n": 1, "reranked_only_correct": 1, "bge_only_correct": 0})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            human_metrics(labels + labels[:1], key, [row, other])


if __name__ == "__main__":
    unittest.main()
