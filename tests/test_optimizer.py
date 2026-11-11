"""Tests for prompt optimization module."""
import pytest
from prompt_engineering.optimizer import (
    PromptCompressor, ABTester, PromptRefiner, PromptVariant,
)


class TestPromptCompressor:
    def test_remove_fillers(self):
        c = PromptCompressor()
        result = c.compress("Please just simply explain this")
        assert "please" not in result["compressed"].lower()
        assert "just" not in result["compressed"].lower()
        assert "simply" not in result["compressed"].lower()

    def test_simplify_verbose(self):
        c = PromptCompressor()
        result = c.compress("In order to understand this, due to the fact that it matters")
        assert "in order to" not in result["compressed"].lower()
        assert "to" in result["compressed"].lower()

    def test_reduction_percentage(self):
        c = PromptCompressor()
        result = c.compress("Please kindly just basically explain this very simple thing")
        assert result["reduction_pct"] > 0

    def test_collapse_whitespace(self):
        c = PromptCompressor()
        result = c.compress("hello    world   test")
        assert "  " not in result["compressed"]

    def test_empty_prompt(self):
        c = PromptCompressor()
        result = c.compress("")
        assert result["compressed"] == ""


class TestABTester:
    def test_add_variant(self):
        t = ABTester()
        t.add_variant("v1", "template 1")
        assert "v1" in t.variants

    def test_record_result(self):
        t = ABTester()
        t.add_variant("v1", "template")
        t.record_result("v1", 0.8)
        assert t.variants["v1"].n_trials == 1
        assert t.variants["v1"].avg_score == 0.8

    def test_unknown_variant_raises(self):
        t = ABTester()
        with pytest.raises(ValueError):
            t.record_result("unknown", 0.5)

    def test_get_winner(self):
        t = ABTester()
        t.add_variant("good", "good template")
        t.add_variant("bad", "bad template")
        for _ in range(5):
            t.record_result("good", 0.9)
            t.record_result("bad", 0.3)
        assert t.get_winner() == "good"

    def test_winner_needs_min_trials(self):
        t = ABTester()
        t.add_variant("v1", "template")
        t.record_result("v1", 1.0)
        t.record_result("v1", 1.0)
        assert t.get_winner() is None  # only 2 trials, need 3


class TestPromptRefiner:
    def test_suggest_refinements(self):
        r = PromptRefiner()
        suggestions = r.suggest_refinements("Explain quantum computing")
        assert len(suggestions) > 0

    def test_no_duplicate_role(self):
        r = PromptRefiner()
        suggestions = r.suggest_refinements("You are an expert. Explain this.")
        rule_names = [s["rule"] for s in suggestions]
        assert "add_role" not in rule_names

    def test_auto_refine(self):
        r = PromptRefiner()
        original = "Explain quantum computing"
        refined = r.auto_refine(original)
        assert len(refined) > len(original)

    def test_auto_refine_max_rules(self):
        r = PromptRefiner()
        refined1 = r.auto_refine("test prompt", max_rules=1)
        refined2 = r.auto_refine("test prompt", max_rules=4)
        assert len(refined2) >= len(refined1)


class TestPromptVariant:
    def test_avg_score(self):
        v = PromptVariant(name="test", template="hello")
        v.scores = [0.5, 0.7, 0.9]
        assert abs(v.avg_score - 0.7) < 0.01

    def test_empty_scores(self):
        v = PromptVariant(name="test", template="hello")
        assert v.avg_score == 0.0

    def test_fingerprint(self):
        v = PromptVariant(name="test", template="hello")
        assert len(v.fingerprint()) == 8
