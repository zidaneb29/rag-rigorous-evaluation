import hashlib
import io
import json
from math import log2
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from tokenizers import BertWordPieceTokenizer

from rag_eval.chunking import Chunk, build_chunks
from rag_eval.corpus import ARCHIVES, Paragraph, load_split, validate
from rag_eval.evaluation import map_evidence, score
from rag_eval.retrieval import Retrieval, cached_vectors


def annotation(key, evidence, unanswerable=False, yes_no=None):
    return {"annotation_id": key, "answer": {
        "unanswerable": unanswerable, "extractive_spans": [], "yes_no": yes_no,
        "free_form_answer": "", "evidence": evidence, "highlighted_evidence": evidence[:1]}}


def paper(question_id="q"):
    return {"title": "title", "abstract": "abstract",
            "full_text": [{"section_name": None, "paragraphs": ["alpha beta"]}],
            "figures_and_tables": [{"caption": "figure evidence"}],
            "qas": [{"question_id": question_id, "question": "alpha?",
                     "answers": [annotation("a", ["alpha beta"], yes_no=False),
                                 annotation("b", ["FLOAT SELECTED: figure evidence"])]}]}


class Encoder:
    calls = 0

    def encode(self, texts, *, query=False):
        self.calls += 1
        output = np.zeros((len(texts), 384), dtype=np.float32)
        output[:, 0] = 1
        return output


class CorpusTests(unittest.TestCase):
    def test_archive_preserves_annotations_and_checks_checksum(self):
        payload = {"pid": paper()}
        data = json.dumps(payload).encode()
        with TemporaryDirectory() as temp:
            name = ARCHIVES["dev"][0]
            path = Path(temp) / name
            with tarfile.open(path, "w:gz") as tar:
                item = tarfile.TarInfo("qasper-dev-v0.3.json")
                item.size = len(data)
                tar.addfile(item, io.BytesIO(data))
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.dict(ARCHIVES, {"dev": (name, checksum)}):
                actual, provenance = load_split(temp, "dev")
            self.assertEqual(actual, payload)
            self.assertEqual(provenance["sha256"], checksum)
            self.assertEqual(provenance["split"], "dev")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_split(temp, "dev")

    def test_duplicate_question_id(self):
        with self.assertRaisesRegex(ValueError, "duplicate question"):
            validate({"p": paper(), "other": paper()})

    def test_invalid_schema(self):
        bad = paper()
        bad["qas"][0]["answers"][0]["answer"]["evidence"] = [None]
        with self.assertRaisesRegex(ValueError, "strings in evidence"):
            validate({"p": bad})


class ChunkingTests(unittest.TestCase):
    def test_stable_window_order_exact_offsets_and_special_tokens(self):
        tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "alpha", "beta", "title", "abstract"]
        tokenizer = BertWordPieceTokenizer({token: i for i, token in enumerate(tokens)})
        raw = "  " + "alpha beta " * 12 + " "
        first, second = paper("q1"), paper("q2")
        first["full_text"][0]["paragraphs"] = [raw, ""]
        chunks = build_chunks({"z": second, "a": first}, tokenizer, 9, 2)
        self.assertEqual(chunks, build_chunks({"a": first, "z": second}, tokenizer, 9, 2))
        self.assertEqual(len(chunks), len({chunk.id for chunk in chunks}))
        windows = [c for c in chunks if c.paragraph_id == "a:0:0"]
        self.assertEqual(windows[0].start, 0)
        self.assertEqual(windows[-1].end, len(raw))
        self.assertTrue(all(c.end > n.start for c, n in zip(windows, windows[1:])))
        for chunk in windows:
            self.assertEqual(chunk.text, "title\n" + raw[chunk.start:chunk.end])
            self.assertLessEqual(len(tokenizer.encode(chunk.text).ids), 9)
        self.assertTrue(any(c.paragraph_id == "a:abstract" for c in chunks))
        self.assertFalse(any("figure evidence" in c.text for c in chunks))


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.paragraphs = [Paragraph("p", "x", "", "alpha beta"),
                           Paragraph("p", "y", "", "gamma"),
                           Paragraph("p", "z", "", "other")]
        self.chunks = [Chunk("x1", "p", "x", 0, 6, "alpha "),
                       Chunk("x2", "p", "x", 6, 10, "beta"),
                       Chunk("y1", "p", "y", 0, 5, "gamma"),
                       Chunk("z1", "p", "z", 0, 5, "other")]

    def test_multi_gold_partial_windows_and_ranking_metrics(self):
        gold = map_evidence({"answers": [annotation("1", ["alpha   beta", "gamma"])]},
                            self.paragraphs, self.chunks)
        metrics = score(self.chunks, gold, [1, 2, 3, 5])
        self.assertEqual(metrics["recall@1"], 0)
        self.assertEqual(metrics["recall@2"], 0.5)
        self.assertEqual(metrics["recall@3"], 1)
        self.assertEqual(metrics["mrr@5"], 0.5)
        self.assertEqual(metrics["annotated_set_coverage@2"], 0)
        self.assertEqual(metrics["annotated_set_coverage@3"], 1)
        self.assertAlmostEqual(metrics["ndcg@3"], (1 / log2(3) + 0.5) / (1 + 1 / log2(3)))

    def test_alternative_sets_and_repeated_windows(self):
        gold = map_evidence({"answers": [annotation("1", ["gamma"]), annotation("2", ["alpha beta"])]},
                            self.paragraphs, self.chunks)
        x1, x2, y1, _ = self.chunks
        metrics = score([y1, x1, x1, x2], gold, [1, 3, 4])
        self.assertEqual(metrics["annotated_set_coverage@1"], 1)
        self.assertEqual(metrics["recall@3"], 0.5)
        self.assertEqual(metrics["recall@4"], 1)

    def test_ineligible_and_non_text_labels(self):
        for answers, status in [
            ([], "no_annotations"),
            ([annotation("1", [], True)], "unanswerable"),
            ([annotation("1", [], True), annotation("2", ["gamma"])], "answerability_tie"),
            ([annotation("1", ["gamma", "FLOAT SELECTED: table"])], "no_complete_text_set"),
        ]:
            with self.subTest(status=status):
                gold = map_evidence({"answers": answers}, self.paragraphs, self.chunks)
                self.assertEqual(gold.status, status)
                self.assertEqual(score(self.chunks, gold, [5]), {})

    def test_duplicate_passage_is_ambiguous_and_yes_no_false_is_preserved(self):
        duplicate = Paragraph("p", "duplicate", "", "gamma")
        gold = map_evidence({"answers": [annotation("1", ["gamma"], yes_no=False)]},
                            self.paragraphs + [duplicate], self.chunks)
        self.assertEqual(gold.status, "no_complete_text_set")
        self.assertEqual(gold.annotations[0]["answer_type"], "yes_no")
        self.assertEqual(gold.annotations[0]["labels"][0]["status"], "ambiguous")

    def test_miss_and_short_ranking(self):
        gold = map_evidence({"answers": [annotation("1", ["gamma"])]}, self.paragraphs, self.chunks)
        self.assertEqual(score([], gold, [5]), {"recall@5": 0, "ndcg@5": 0,
                                               "annotated_set_coverage@5": 0, "mrr@5": 0})


class RetrievalTests(unittest.TestCase):
    chunks = [Chunk("b", "p", "x", 0, 4, "beta"), Chunk("a", "q", "y", 0, 5, "alpha"),
              Chunk("c", "p", "z", 0, 5, "gamma")]

    def test_bm25_matches_and_ties_in_stable_order(self):
        engine = Retrieval(self.chunks, method="bm25")
        self.assertEqual(engine.search("alpha", 1)[0][0].id, "a")
        self.assertEqual([chunk.id for chunk, _ in engine.search("absent", 5)], ["a", "b", "c"])
        self.assertEqual([chunk.id for chunk, _ in engine.search("alpha", 5, "p")], ["b", "c"])
        self.assertEqual(engine.search("alpha", 5, "missing"), [])

    def test_dense_exact_and_scope(self):
        vectors = Encoder().encode([chunk.text for chunk in self.chunks])
        vectors[0, :2] = [0, 1]
        engine = Retrieval(self.chunks, method="dense", vectors=vectors, encoder=Encoder())
        self.assertEqual([(chunk.id, value) for chunk, value in engine.search("query", 3)],
                         [("a", 1.0), ("c", 1.0), ("b", 0.0)])
        self.assertEqual(engine.search("query", 1, "p")[0][0].id, "c")

    def test_cache_reuse_invalidation_and_corruption(self):
        encoder = Encoder()
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            _, path, hit = cached_vectors(directory, self.chunks, "key", encoder)
            self.assertFalse(hit)
            _, _, hit = cached_vectors(directory, self.chunks, "key", encoder)
            self.assertTrue(hit)
            self.assertEqual(encoder.calls, 1)
            reordered, other, hit = cached_vectors(directory, self.chunks[::-1], "key", encoder)
            self.assertFalse(hit)
            self.assertNotEqual(path, other)
            del reordered
            path.write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "checksum"):
                cached_vectors(directory, self.chunks, "key", encoder)


if __name__ == "__main__":
    unittest.main()
