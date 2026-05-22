from pathlib import Path

from utils.patterns import load_rl_patterns, load_sft_patterns, rl_row, sft_row


def test_sft_patterns_only_use_harmful_and_general_sources():
    rows = load_sft_patterns(Path("datasets/sft/source"), limit_per_pattern=2)
    assert {row.pattern_type for row in rows} == {"harmful", "general"}
    first = sft_row(rows[0], system_prompt="sys")
    assert first["pattern_type"] in {"harmful", "general"}
    assert first["question"]
    assert first["response"]


def test_rl_patterns_emit_verl_schema():
    rows = load_rl_patterns(Path("datasets/rl"), limit_per_pattern=2)
    assert {row.pattern_type for row in rows} == {"harmful", "general"}
    item = rl_row(rows[0])
    assert item["data_source"] in {"reflector_harmful", "reflector_general"}
    assert item["prompt"][0]["role"] == "user"
    assert item["extra_info"]["prompt_type"] in {"harmful", "general"}
