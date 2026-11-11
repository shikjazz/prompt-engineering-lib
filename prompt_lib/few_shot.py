"""Few-shot example manager for in-context learning."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Example:
    """A single few-shot example."""

    input_text: str
    output_text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None

    def format(self, input_prefix: str = "Input", output_prefix: str = "Output") -> str:
        return f"{input_prefix}: {self.input_text}\n{output_prefix}: {self.output_text}"


class FewShotManager:
    """Manage and select few-shot examples for prompts.

    Supports random selection, similarity-based retrieval,
    and stratified sampling across categories.
    """

    def __init__(self):
        self._examples: list[Example] = []
        self._categories: dict[str, list[Example]] = {}

    def add(
        self,
        input_text: str,
        output_text: str,
        category: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Example:
        """Add a few-shot example.

        Args:
            input_text: The example input.
            output_text: The expected output.
            category: Optional category for stratified sampling.
            metadata: Optional metadata dict.

        Returns:
            The created Example.
        """
        example = Example(
            input_text=input_text,
            output_text=output_text,
            metadata=metadata or {},
        )
        self._examples.append(example)
        if category:
            if category not in self._categories:
                self._categories[category] = []
            self._categories[category].append(example)
        return example

    def add_example(self, example: Example, category: Optional[str] = None) -> None:
        """Add a pre-built Example object."""
        self._examples.append(example)
        if category:
            if category not in self._categories:
                self._categories[category] = []
            self._categories[category].append(example)

    def select_random(self, n: int = 3, seed: Optional[int] = None) -> list[Example]:
        """Select n random examples.

        Args:
            n: Number of examples to select.
            seed: Optional random seed for reproducibility.

        Returns:
            List of selected examples.
        """
        if seed is not None:
            random.seed(seed)
        n = min(n, len(self._examples))
        return random.sample(self._examples, n)

    def select_stratified(self, n_per_category: int = 1) -> list[Example]:
        """Select examples evenly across categories.

        Args:
            n_per_category: Number of examples per category.

        Returns:
            List of selected examples, balanced across categories.
        """
        selected = []
        for cat_examples in self._categories.values():
            n = min(n_per_category, len(cat_examples))
            selected.extend(random.sample(cat_examples, n))
        return selected

    def select_similar(
        self,
        query_embedding: np.ndarray,
        n: int = 3,
    ) -> list[Example]:
        """Select the most similar examples by cosine similarity.

        Args:
            query_embedding: The query vector.
            n: Number of examples to return.

        Returns:
            List of most similar examples, sorted by similarity.
        """
        scored = []
        for ex in self._examples:
            if ex.embedding is not None:
                sim = _cosine_similarity(query_embedding, ex.embedding)
                scored.append((sim, ex))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored[:n]]

    def format(
        self,
        examples: Optional[list[Example]] = None,
        input_prefix: str = "Input",
        output_prefix: str = "Output",
        separator: str = "\n\n",
    ) -> str:
        """Format examples into a prompt-ready string.

        Args:
            examples: Specific examples to format. If None, uses all.
            input_prefix: Prefix before each input.
            output_prefix: Prefix before each output.
            separator: String between examples.

        Returns:
            Formatted string of all examples.
        """
        exs = examples or self._examples
        return separator.join(
            ex.format(input_prefix=input_prefix, output_prefix=output_prefix)
            for ex in exs
        )

    @property
    def categories(self) -> list[str]:
        return list(self._categories.keys())

    def __len__(self) -> int:
        return len(self._examples)

    def __iter__(self):
        return iter(self._examples)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
