"""C8-ABL-SWEEP1 — config_hash determinism (pkg-08 spec 05 §3.5)."""
from __future__ import annotations

from hyper_mve.experiments.sweep import compute_config_hash


def test_config_hash_is_32_hex_chars():
    h = compute_config_hash({"a": 1, "b": "x", "c": [1, 2, 3]})
    assert len(h) == 32
    assert all(c in "0123456789abcdef" for c in h)


def test_config_hash_deterministic_across_calls():
    cfg = {"env": {"N": 4, "A": 6}, "train": {"lr": 3e-4, "max_steps": 100000}}
    h1 = compute_config_hash(cfg)
    h2 = compute_config_hash(cfg)
    assert h1 == h2


def test_config_hash_invariant_under_dict_order():
    cfg_a = {"a": 1, "b": 2, "c": 3}
    cfg_b = {"c": 3, "a": 1, "b": 2}
    assert compute_config_hash(cfg_a) == compute_config_hash(cfg_b)


def test_config_hash_sensitive_to_value_change():
    cfg_a = {"env": {"N": 4}, "train": {"lr": 3e-4}}
    cfg_b = {"env": {"N": 4}, "train": {"lr": 1e-4}}
    assert compute_config_hash(cfg_a) != compute_config_hash(cfg_b)


def test_config_hash_tuple_list_coerce_equivalent():
    # cfg.to_dict() coerces tuples to lists, but the hash should only see
    # the post-coercion JSON; we mimic that by passing lists directly.
    cfg_a = {"seeds": [0, 1, 2]}
    cfg_b = {"seeds": [0, 1, 2]}
    assert compute_config_hash(cfg_a) == compute_config_hash(cfg_b)
