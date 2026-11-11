"""Prompt evaluation framework for measuring quality."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class EvalResult:
    """Result from evaluating a single prompt-output pair."""

    scores: dict[str, float]
    prompt: str
    output: str
    expected: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def average_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)

    def passed(self, threshold: float = 0.5) -> bool:
        return self.average_score >= threshold


@dataclass
class EvalSummary:
    """Aggregate results across multiple evaluations."""

    results: list[EvalResult]
    metric_names: list[str]

    @property
    def num_examples(self) -> int:
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        passed = sum(1 for r in self.results if r.passed())
        return passed / len(self.results)

    def mean_scores(self) -> dict[str, float]:
        if not self.results:
            return {}
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for r in self.results:
            for name, score in r.scores.items():
                totals[name] = totals.get(name, 0.0) + score
                counts[name] = counts.get(name, 0) + 1
        return {name: totals[name] / counts[name] for name in totals}

    def summary_table(self) -> str:
        means = self.mean_scores()
        lines = [
            f"Evaluation Summary ({self.num_examples} examples)",
            "-" * 40,
        ]
        for name, score in means.items():
            lines.append(f"  {name:20s}: {score:.4f}")
        lines.append(f"  {'Pass Rate':20s}: {self.pass_rate:.2%}")
        return "\n".join(lines)


# Built-in metric functions

def exact_match(output: str, expected: str) -> float:
    """1.0 if output matches expected exactly, else 0.0."""
    return 1.0 if output.strip() == expected.strip() else 0.0


def contains_match(output: str, expected: str) -> float:
    """1.0 if expected text appears anywhere in output."""
    return 1.0 if expected.strip() in output else 0.0


def length_ratio(output: str, expected: str) -> float:
    """Score based on how close the output length is to the expected length.

    Returns a value between 0 and 1, where 1 means equal length.
    """
    len_out = len(output.split())
    len_exp = len(expected.split())
    if len_exp == 0:
        return 1.0 if len_out == 0 else 0.0
    ratio = min(len_out, len_exp) / max(len_out, len_exp)
    return ratio


def word_overlap(output: str, expected: str) -> float:
    """Jaccard similarity of word sets."""
    words_out = set(output.lower().split())
    words_exp = set(expected.lower().split())
    if not words_out and not words_exp:
        return 1.0
    if not words_out or not words_exp:
        return 0.0
    intersection = words_out & words_exp
    union = words_out | words_exp
    return len(intersection) / len(union)


BUILTIN_METRICS = {
    "exact_match": exact_match,
    "contains": contains_match,
    "length_ratio": length_ratio,
    "word_overlap": word_overlap,
}


class PromptEvaluator:
    """Evaluate prompt quality across test cases.

    Usage:
        evaluator = PromptEvaluator(metrics=["exact_match", "word_overlap"])
        evaluator.add_custom_metric("my_metric", my_scoring_fn)

        result = evaluator.evaluate(
            prompt="Translate: {text}",
            output="Bonjour le monde",
            expected="Bonjour le monde",
        )

        # Batch evaluation
        summary = evaluator.evaluate_batch(test_cases)
    """

    def __init__(self, metrics: Optional[list[str]] = None):
        self._metrics: dict[str, Callable[[str, str], float]] = {}
        if metrics:
            for name in metrics:
                if name in BUILTIN_METRICS:
                    self._metrics[name] = BUILTIN_METRICS[name]
                else:
                    raise ValueError(
                        f"Unknown metric: {name}. "
                        f"Available: {list(BUILTIN_METRICS.keys())}"
                    )
        else:
            self._metrics = dict(BUILTIN_METRICS)

    def add_custom_metric(
        self,
        name: str,
        fn: Callable[[str, str], float],
    ) -> None:
        """Register a custom metric function.

        Args:
            name: Metric name.
            fn: Callable taking (output, expected) and returning a float score.
        """
        self._metrics[name] = fn

    def evaluate(
        self,
        prompt: str,
        output: str,
        expected: str,
        metadata: Optional[dict] = None,
    ) -> EvalResult:
        """Evaluate a single prompt-output pair.

        Args:
            prompt: The prompt that was sent.
            output: The LLM's response.
            expected: The expected/reference response.
            metadata: Optional extra info to attach.

        Returns:
            EvalResult with scores for each metric.
        """
        scores = {}
        for name, fn in self._metrics.items():
            scores[name] = fn(output, expected)

        return EvalResult(
            scores=scores,
            prompt=prompt,
            output=output,
            expected=expected,
            metadata=metadata or {},
        )

    def evaluate_batch(
        self,
        test_cases: list[dict[str, str]],
    ) -> EvalSummary:
        """Evaluate multiple test cases.

        Args:
            test_cases: List of dicts with 'prompt', 'output', and 'expected'.

        Returns:
            EvalSummary with aggregate statistics.
        """
        results = []
        for case in test_cases:
            result = self.evaluate(
                prompt=case.get("prompt", ""),
                output=case["output"],
                expected=case["expected"],
                metadata=case.get("metadata"),
            )
            results.append(result)

        return EvalSummary(
            results=results,
            metric_names=list(self._metrics.keys()),
        )
