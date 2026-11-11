"""Tests for the prompt evaluation framework."""

from prompt_lib.evaluation import (
    PromptEvaluator,
    exact_match,
    contains_match,
    word_overlap,
    length_ratio,
)


def test_exact_match():
    assert exact_match("hello", "hello") == 1.0
    assert exact_match("hello", "world") == 0.0
    assert exact_match("  hello  ", "hello") == 1.0  # strips whitespace


def test_contains_match():
    assert contains_match("I love Python programming", "Python") == 1.0
    assert contains_match("I love Java", "Python") == 0.0


def test_word_overlap():
    score = word_overlap("the cat sat", "the cat ran")
    assert 0.0 < score < 1.0
    assert word_overlap("hello world", "hello world") == 1.0
    assert word_overlap("a b c", "x y z") == 0.0


def test_length_ratio():
    assert length_ratio("one two three", "one two three") == 1.0
    score = length_ratio("one", "one two three four five")
    assert score < 1.0
    assert score > 0.0


def test_evaluator_all_metrics():
    evaluator = PromptEvaluator()
    result = evaluator.evaluate(
        prompt="test prompt",
        output="hello world",
        expected="hello world",
    )
    assert result.scores["exact_match"] == 1.0
    assert result.passed()


def test_evaluator_specific_metrics():
    evaluator = PromptEvaluator(metrics=["exact_match", "word_overlap"])
    result = evaluator.evaluate(
        prompt="",
        output="the quick brown fox",
        expected="the slow brown fox",
    )
    assert "exact_match" in result.scores
    assert "word_overlap" in result.scores
    assert "contains" not in result.scores


def test_custom_metric():
    evaluator = PromptEvaluator(metrics=[])

    def starts_with_score(output, expected):
        return 1.0 if output.startswith(expected[:3]) else 0.0

    evaluator.add_custom_metric("starts_with", starts_with_score)
    result = evaluator.evaluate(prompt="", output="hello", expected="hel")
    assert result.scores["starts_with"] == 1.0


def test_batch_evaluation():
    evaluator = PromptEvaluator(metrics=["exact_match"])
    cases = [
        {"prompt": "p1", "output": "a", "expected": "a"},
        {"prompt": "p2", "output": "b", "expected": "b"},
        {"prompt": "p3", "output": "c", "expected": "x"},
    ]
    summary = evaluator.evaluate_batch(cases)
    assert summary.num_examples == 3
    assert summary.pass_rate == pytest.approx(2 / 3, abs=0.01)


# Need to import pytest for approx
import pytest
