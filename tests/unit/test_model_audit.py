from pathlib import Path

import yaml

from ts_coder.model import ModelConfig, model_audit, parameter_breakdown


def test_analytic_parameter_breakdown_matches_instantiated_tiers():
    for path, expected in (
        ("configs/model/smoke-25m.yaml", 25_172_352),
        ("configs/model/pilot-100m.yaml", 100_682_496),
    ):
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        config = ModelConfig(
            **{key: value for key, value in raw.items() if key in ModelConfig.__dataclass_fields__}
        )
        assert parameter_breakdown(config)["total"] == expected


def test_model_audit_reports_scale_and_memory_without_instantiation():
    report = model_audit(
        ModelConfig(
            vocab_size=32768,
            context_length=8192,
            layers=24,
            hidden_size=2048,
            attention_heads=16,
            kv_heads=4,
            ffn_size=5504,
        )
    )
    assert report["parameter_count"] == 1_130_465_280
    assert report["head_dimension"] == 128
    assert report["query_to_kv_group_ratio"] == 4
    assert report["memory_estimate_fp32"]["total_estimate"] > report["parameter_count"] * 16
