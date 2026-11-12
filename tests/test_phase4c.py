import copy
import inspect as source_inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import tomllib
import unittest

from rag_eval.generation import parse_answer
from rag_eval.phase4c import case, inspect, validate_config
from rag_eval.phase4c_runtime import (
    RUNTIME_ID, append_result, digest, render, select_wheel, validate_checkpoint,
)


class Phase4CTests(unittest.TestCase):
    def setUp(self):
        self.config = tomllib.loads(Path("configs/phase4c.toml").read_text())
        cases = [case(stage, f"{stage}-{i}", "What color?", [{"id": "P1", "text": "Blue."}],
                      True, "reference")
                 for stage, size in [("sanity", 3), ("original", 8), ("confirmation", 15)]
                 for i in range(size)]
        payload = {"config": self.config, "cases": cases}
        self.packet = {"payload": payload, "sha256": digest(payload)}
        wheel, checksum, size = select_wheel("Linux", "x86_64", (3, 12))
        self.checkpoint = {"packet_sha256": self.packet["sha256"], "approvals": {}, "rows": [],
                           "sessions": [{
                               "runtime_id": RUNTIME_ID,
                               "model_revision": self.config["revision"],
                               "weight_shards": self.config["shards"],
                               "versions": {"llama-cpp-python": self.config["backend_version"]},
                               "wheel": {"name": wheel, "sha256": checksum, "bytes": size},
                               "gpu_before": {"free_mib": 15000, "name": "Tesla T4",
                                              "compute_capability": "7.5", "driver": "550.54.15"},
                               "cuda_driver_api": 12040,
                               "gpu_probe": {"process_used_mib": 5000, "tokens_evaluated": 4},
                               "settings": {"n_gpu_layers": -1, "n_batch": 256, "n_ctx": 2048},
                           }]}
        first = cases[0]
        prompt = render(first["chat"])
        self.row = {"id": first["id"], "stage": first["stage"], "session": 0,
                    "prompt": prompt, "prompt_sha256": digest(prompt), "prompt_tokens": 256,
                    "output_tokens": 32, "latency_s": 1.0,
                    "raw": '{"abstain":false,"claims":[{"text":"Blue.","citations":["P1","P2"]}]}'}

    def test_frozen_settings_and_phase4_budget(self):
        phase4 = tomllib.loads(Path("configs/phase4.toml").read_text())
        validate_config(self.config, phase4)
        for patch in ({"temperature": .1}, {"max_output": 128}, {"prompt": "B"}):
            with self.assertRaises(ValueError):
                validate_config(self.config | patch, phase4)

    def test_checkpoint_recovery_and_citation_resolution(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint.json"
            append_result(self.packet, self.checkpoint, path, self.row)
            reloaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(validate_checkpoint(self.packet, reloaded)[0]["record"], self.row)
            summary = inspect(self.packet, reloaded)["automatic_counts"]["sanity"]
            self.assertEqual(summary["invalid_citation_ids"], 1)
            self.assertEqual(summary["resolving_citation_ids"], 1)
            self.assertEqual(summary["citation_error"], 1)
            with self.assertRaises(ValueError):
                append_result(self.packet, self.checkpoint, path, self.row)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), reloaded)

    def test_reject_packet_and_session_tampering(self):
        packet = copy.deepcopy(self.packet)
        packet["payload"]["cases"][0]["chat"][0]["content"] += " change"
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_checkpoint(packet, self.checkpoint)
        self.checkpoint["sessions"][0]["gpu_before"]["free_mib"] = 1024
        with self.assertRaisesRegex(ValueError, "Runtime session"):
            validate_checkpoint(self.packet, self.checkpoint)

    def test_notebook_embeds_exact_protocol_and_parser(self):
        notebook = json.loads(Path("notebooks/phase4c.ipynb").read_text(encoding="utf-8"))
        embedded = "".join(notebook["cells"][1]["source"])
        self.assertEqual(embedded, Path("rag_eval/phase4c_runtime.py").read_text(encoding="utf-8")
                         + "\n" + source_inspect.getsource(parse_answer))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "phase4c cell", "exec")
        setup = next("".join(cell["source"]) for cell in notebook["cells"]
                     if "def process_gpu_mib" in "".join(cell["source"]))
        self.assertIn("--only-binary=:all:", setup)
        self.assertNotIn("CMAKE_ARGS", setup)
        self.assertNotIn("nvcc", setup)

    def test_wheel_selection_and_runtime_isolation(self):
        name, _, _ = select_wheel("Linux", "x86_64", (3, 11))
        self.assertIn("cp311", name)
        for system, arch, version in [("Windows", "x86_64", (3, 12)),
                                      ("Linux", "aarch64", (3, 12)),
                                      ("Linux", "x86_64", (3, 13))]:
            with self.assertRaises(ValueError):
                select_wheel(system, arch, version)
        newer = copy.deepcopy(self.checkpoint["sessions"][0])
        newer["gpu_before"]["driver"] = "570.00"
        self.checkpoint["sessions"].append(newer)
        with self.assertRaisesRegex(ValueError, "different runtime environments"):
            validate_checkpoint(self.packet, self.checkpoint)


if __name__ == "__main__":
    unittest.main()
