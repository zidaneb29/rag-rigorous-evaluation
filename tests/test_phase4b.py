import json
import unittest

from rag_eval.generation import SCHEMA, messages, parse_answer
from rag_eval.phase4b import EXAMPLES, SYNTHETIC, chat_for, request_payload, token_hash


class RecoveryTests(unittest.TestCase):
    context = [{"id": "P1", "text": "The calibration object is square."}]

    def test_original_condition_preserves_exact_messages(self):
        self.assertEqual(chat_for("A", "What shape?", self.context), messages("What shape?", self.context))

    def test_ablation_keeps_evidence_and_schema_fixed(self):
        chats = [chat_for(condition, "What shape?", self.context) for condition in "ABC"]
        self.assertTrue(all(chat[-1] == chats[0][-1] for chat in chats))
        config = {"max_output": 176, "seed": 42}
        payloads = [request_payload([1, 2], config, condition) for condition in "ABC"]
        self.assertEqual(payloads[0], payloads[1])
        self.assertEqual(payloads[0], payloads[2])
        self.assertEqual(payloads[0]["json_schema"], SCHEMA)
        plain = request_payload([1, 2], config, "plain")
        self.assertEqual(plain, {k: v for k, v in payloads[0].items() if k != "json_schema"})

    def test_balanced_examples_are_synthetic_and_distinct_from_checks(self):
        chat = chat_for("C", "What shape?", self.context)
        answers = [json.loads(turn["content"]) for turn in chat if turn["role"] == "assistant"]
        self.assertEqual([answer["abstain"] for answer in answers], [False, True])
        self.assertEqual(answers[0]["claims"][0]["citations"], ["P1"])
        self.assertTrue({item[0] for item in EXAMPLES}.isdisjoint(q for _, q, _, _ in SYNTHETIC))

    def test_parse_failures_do_not_create_abstentions(self):
        self.assertIsNone(parse_answer("Insufficient evidence.", {"P1"})["abstained"])
        self.assertIsNone(parse_answer('{"abstain":true,"claims":[{}]}', {"P1"})["abstained"])
        self.assertTrue(parse_answer('{"abstain":true,"claims":[]}', {"P1"})["abstained"])
        self.assertNotEqual(token_hash([1, 2]), token_hash([2, 1]))


if __name__ == "__main__":
    unittest.main()
