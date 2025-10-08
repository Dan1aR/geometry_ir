from pathlib import Path

from geoscript_ir.reference import BNF, LLM_PROMPT, get_llm_prompt


def test_bnf_matches_docs():
    docs_path = Path(__file__).resolve().parents[1] / "docs" / "bnf.txt"
    assert docs_path.exists(), "expected docs/bnf.txt to exist"
    assert BNF == docs_path.read_text(encoding="utf-8").strip()


def test_prompt_includes_bnf_by_default():
    prompt = LLM_PROMPT
    assert "SYNTAX REFERENCE (BNF)" in prompt
    assert BNF in prompt


def test_prompt_can_skip_bnf():
    prompt = get_llm_prompt(include_bnf=False)
    assert "SYNTAX REFERENCE" not in prompt
    assert BNF not in prompt
