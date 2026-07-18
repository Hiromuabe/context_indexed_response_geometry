from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from prefix_displacement.cache_io import (
    CacheValidationError,
    TransitionCacheWriter,
    validate_transition_cache,
)
from prefix_displacement.config import CacheConfig
from prefix_displacement.schema import TransitionSchemaError
from prefix_displacement.split_registry import (
    SplitRatios,
    create_split_registry,
    write_split_registry,
)


def make_record(problem_id: str, split: str, transition_index: int) -> dict:
    base = torch.arange(6, dtype=torch.float32) + transition_index
    step = torch.linspace(0.1, 0.6, 6)
    arrival = base + step
    return {
        "problem_id": problem_id,
        "trajectory_id": f"{problem_id}-trajectory",
        "transition_id": f"transition-{transition_index}",
        "token_index": transition_index,
        "current_token_id": 10 + transition_index,
        "current_token_text": f" token-{transition_index}",
        "next_token_id": 11 + transition_index,
        "next_token_text": "." if transition_index % 2 else " 2",
        "absolute_position": 20 + transition_index,
        "relative_generated_position": transition_index,
        "boundary_class": "sentence_punctuation" if transition_index % 2 else "number_token",
        "surprisal": 0.25 + transition_index,
        "correctness": True,
        "split": split,
        "h_departure": base,
        "h_arrival": arrival,
        "delta": arrival - base,
        "baseline_margin": 1.5 - transition_index,
        "g_arrival": torch.ones(6) * 0.2,
    }


class CacheIoTest(unittest.TestCase):
    def _build(self, root: Path):
        problem_ids = [f"gsm8k-{index}" for index in range(6)]
        registry = create_split_registry(
            problem_ids, ratios=SplitRatios(0.5, 1.0 / 6.0, 1.0 / 3.0), seed=5
        )
        registry_path = root / "split_registry.json"
        write_split_registry(registry, registry_path)
        cache_config = CacheConfig(
            output_dir=root / "cache",
            max_records_per_shard=3,
            max_bytes_per_shard=4096,
            storage_dtype="float32",
            zero_norm_epsilon=1e-12,
            delta_atol=1e-6,
            delta_rtol=1e-6,
        )
        writer = TransitionCacheWriter(
            cache_config=cache_config,
            split_registry=registry,
            split_registry_path=registry_path,
            scientific_metadata={
                "model": {"checkpoint": "synthetic", "layer": 1, "hook": "toy"},
                "margin": {"definition": "synthetic", "gradient_location": "arrival"},
            },
            source_metadata={"kind": "synthetic", "purpose": "unit_test"},
        )
        for problem_id in problem_ids:
            split = registry["problem_to_split"][problem_id]
            writer.add(make_record(problem_id, split, 0))
            writer.add(make_record(problem_id, split, 1))
        return writer.finalize(), registry

    def test_sharded_round_trip_and_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, registry = self._build(Path(directory))
            summary = validate_transition_cache(manifest_path, split_registry=registry)
        self.assertEqual(summary["num_records"], 12)
        self.assertEqual(summary["num_problems"], 6)
        self.assertEqual(summary["num_shards"], 4)
        self.assertEqual(summary["split_leakage_check"], "PASS")
        self.assertEqual(summary["delta_identity_check"], "PASS")
        self.assertTrue(all(count == 0 for count in summary["missing_counts"].values()))
        self.assertEqual(summary["metadata_types"]["problem_id"], ["str"])

    def test_delta_mismatch_fails_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = create_split_registry(
                ["p0", "p1", "p2"], ratios=SplitRatios(1 / 3, 1 / 3, 1 / 3), seed=1
            )
            registry_path = root / "registry.json"
            write_split_registry(registry, registry_path)
            writer = TransitionCacheWriter(
                cache_config=CacheConfig(
                    root / "cache", 10, None, "float32", 1e-12, 1e-6, 1e-6
                ),
                split_registry=registry,
                split_registry_path=registry_path,
                scientific_metadata={"model": {}, "margin": {}},
            )
            record = make_record("p0", registry["problem_to_split"]["p0"], 0)
            record["delta"] = torch.zeros(6)
            with self.assertRaisesRegex(TransitionSchemaError, "delta must equal"):
                writer.add(record)

    def test_checksum_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, registry = self._build(root)
            metadata_path = root / "cache" / "shard-00000.metadata.jsonl"
            with metadata_path.open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(CacheValidationError, "checksum mismatch"):
                validate_transition_cache(manifest_path, split_registry=registry)


if __name__ == "__main__":
    unittest.main()
