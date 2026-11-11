"""
Prompt Optimization Module.
Provides automated prompt optimization techniques including:
- Prompt compression for token efficiency
- A/B testing framework for prompt variants
- Automatic prompt refinement based on scoring criteria
"""
import re
import hashlib
import json
from typing import Callable, Optional
from dataclasses import dataclass, field


# Common filler words that can be removed without losing meaning
FILLER_WORDS = {
    "please", "kindly", "just", "simply", "basically", "actually",
    "really", "very", "quite", "rather", "somewhat", "perhaps",
    "maybe", "possibly", "certainly", "definitely", "absolutely",
    "obviously", "clearly", "essentially", "literally",
}

# Verbose phrases and their concise replacements
VERBOSE_TO_CONCISE = {
    "in order to": "to",
    "due to the fact that": "because",
    "at this point in time": "now",
    "in the event that": "if",
    "for the purpose of": "to",
    "with regard to": "regarding",
    "in spite of the fact that": "although",
    "on the other hand": "however",
    "as a matter of fact": "actually",
    "in addition to": "besides",
    "prior to": "before",
    "subsequent to": "after",
    "in the vicinity of": "near",
    "a large number of": "many",
    "a small number of": "few",
    "is able to": "can",
    "has the ability to": "can",
    "make use of": "use",
    "take into consideration": "consider",
    "come to the conclusion": "conclude",
}


@dataclass
class PromptVariant:
    """A prompt variant for A/B testing."""
    name: str
    template: str
    scores: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def avg_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def n_trials(self) -> int:
        return len(self.scores)

    def fingerprint(self) -> str:
        return hashlib.md5(self.template.encode()).hexdigest()[:8]


class PromptCompressor:
    """Compress prompts to reduce token count while preserving meaning."""

    def __init__(
        self,
        remove_fillers: bool = True,
        simplify_verbose: bool = True,
        collapse_whitespace: bool = True,
    ):
        self.remove_fillers = remove_fillers
        self.simplify_verbose = simplify_verbose
        self.collapse_whitespace = collapse_whitespace

    def compress(self, prompt: str) -> dict:
        """Compress a prompt and return stats."""
        original = prompt
        original_words = len(prompt.split())

        if self.simplify_verbose:
            for verbose, concise in VERBOSE_TO_CONCISE.items():
                prompt = re.sub(
                    re.escape(verbose), concise, prompt, flags=re.IGNORECASE
                )

        if self.remove_fillers:
            words = prompt.split()
            words = [w for w in words if w.lower().strip(".,!?;:") not in FILLER_WORDS]
            prompt = " ".join(words)

        if self.collapse_whitespace:
            prompt = re.sub(r'\s+', ' ', prompt).strip()

        compressed_words = len(prompt.split())
        reduction = 1 - (compressed_words / max(original_words, 1))

        return {
            "original": original,
            "compressed": prompt,
            "original_words": original_words,
            "compressed_words": compressed_words,
            "reduction_pct": round(reduction * 100, 1),
        }


class ABTester:
    """A/B testing framework for prompt variants."""

    def __init__(self):
        self.variants: dict[str, PromptVariant] = {}
        self.history: list[dict] = []

    def add_variant(self, name: str, template: str, **metadata):
        """Register a prompt variant for testing."""
        self.variants[name] = PromptVariant(
            name=name, template=template, metadata=metadata
        )

    def record_result(self, variant_name: str, score: float, context: dict = None):
        """Record a test result for a variant."""
        if variant_name not in self.variants:
            raise ValueError(f"Unknown variant: {variant_name}")
        self.variants[variant_name].scores.append(score)
        self.history.append({
            "variant": variant_name,
            "score": score,
            "context": context or {},
        })

    def get_results(self) -> list[dict]:
        """Get summary results for all variants, sorted by average score."""
        results = []
        for name, variant in self.variants.items():
            results.append({
                "name": name,
                "avg_score": round(variant.avg_score, 4),
                "n_trials": variant.n_trials,
                "min_score": min(variant.scores) if variant.scores else None,
                "max_score": max(variant.scores) if variant.scores else None,
                "fingerprint": variant.fingerprint(),
            })
        results.sort(key=lambda x: x["avg_score"], reverse=True)
        return results

    def get_winner(self) -> Optional[str]:
        """Get the best-performing variant (minimum 3 trials)."""
        eligible = [
            v for v in self.variants.values()
            if v.n_trials >= 3
        ]
        if not eligible:
            return None
        return max(eligible, key=lambda v: v.avg_score).name


class PromptRefiner:
    """Refine prompts using rule-based transformations."""

    REFINEMENT_RULES = [
        {
            "name": "add_role",
            "check": lambda p: not any(
                w in p.lower() for w in ["you are", "act as", "role:", "persona:"]
            ),
            "transform": lambda p: f"You are a helpful expert assistant.\n\n{p}",
            "description": "Add a role/persona prefix",
        },
        {
            "name": "add_format_spec",
            "check": lambda p: not any(
                w in p.lower() for w in [
                    "format:", "output:", "respond in", "return as",
                    "json", "markdown", "bullet",
                ]
            ),
            "transform": lambda p: f"{p}\n\nProvide your response in a clear, structured format.",
            "description": "Add output format specification",
        },
        {
            "name": "add_step_by_step",
            "check": lambda p: not any(
                w in p.lower() for w in [
                    "step by step", "step-by-step", "think through",
                    "chain of thought", "reasoning",
                ]
            ),
            "transform": lambda p: f"{p}\n\nThink through this step by step.",
            "description": "Add chain-of-thought instruction",
        },
        {
            "name": "add_constraints",
            "check": lambda p: not any(
                w in p.lower() for w in [
                    "do not", "don't", "avoid", "never",
                    "must not", "should not",
                ]
            ),
            "transform": lambda p: f"{p}\n\nAvoid unnecessary verbosity and stay focused on the task.",
            "description": "Add behavioral constraints",
        },
    ]

    def suggest_refinements(self, prompt: str) -> list[dict]:
        """Suggest refinements for a prompt based on rules."""
        suggestions = []
        for rule in self.REFINEMENT_RULES:
            if rule["check"](prompt):
                refined = rule["transform"](prompt)
                suggestions.append({
                    "rule": rule["name"],
                    "description": rule["description"],
                    "original": prompt,
                    "refined": refined,
                })
        return suggestions

    def auto_refine(self, prompt: str, max_rules: int = 2) -> str:
        """Automatically apply top refinement rules."""
        applied = 0
        for rule in self.REFINEMENT_RULES:
            if applied >= max_rules:
                break
            if rule["check"](prompt):
                prompt = rule["transform"](prompt)
                applied += 1
        return prompt


if __name__ == "__main__":
    # Demo: Compression
    print("=== Prompt Compression ===")
    compressor = PromptCompressor()
    verbose_prompt = (
        "Please kindly take into consideration the following information "
        "and basically just simply provide a response that is quite clear. "
        "In order to do this, you should definitely make use of your knowledge "
        "due to the fact that the user really needs help."
    )
    result = compressor.compress(verbose_prompt)
    print(f"  Original ({result['original_words']} words): {result['original'][:80]}...")
    print(f"  Compressed ({result['compressed_words']} words): {result['compressed'][:80]}...")
    print(f"  Reduction: {result['reduction_pct']}%")

    # Demo: A/B Testing
    print("\n=== A/B Testing ===")
    tester = ABTester()
    tester.add_variant("direct", "Summarize this text: {text}")
    tester.add_variant("detailed", "You are an expert summarizer. Create a concise summary of: {text}")
    tester.add_variant("structured", "Summarize in 3 bullet points: {text}")

    import random
    random.seed(42)
    for _ in range(10):
        tester.record_result("direct", random.uniform(0.5, 0.8))
        tester.record_result("detailed", random.uniform(0.6, 0.9))
        tester.record_result("structured", random.uniform(0.7, 0.95))

    for r in tester.get_results():
        print(f"  {r['name']}: avg={r['avg_score']:.3f} (n={r['n_trials']})")
    print(f"  Winner: {tester.get_winner()}")

    # Demo: Refinement
    print("\n=== Prompt Refinement ===")
    refiner = PromptRefiner()
    simple_prompt = "Explain quantum computing"
    suggestions = refiner.suggest_refinements(simple_prompt)
    print(f"  Original: {simple_prompt}")
    print(f"  Suggestions ({len(suggestions)}):")
    for s in suggestions:
        print(f"    - {s['description']}")
    refined = refiner.auto_refine(simple_prompt)
    print(f"  Auto-refined: {refined}")
"""
