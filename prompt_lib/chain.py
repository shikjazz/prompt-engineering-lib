"""Prompt chaining for multi-step reasoning pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from prompt_lib.template import PromptTemplate


@dataclass
class ChainStep:
    """A single step in a prompt chain."""

    template: PromptTemplate
    name: Optional[str] = None
    output_key: Optional[str] = None
    transform: Optional[Callable[[str], Any]] = None
    condition: Optional[Callable[[dict], bool]] = None

    def __post_init__(self):
        if self.name is None:
            self.name = self.template.name
        if self.output_key is None:
            self.output_key = self.name

    def should_run(self, context: dict) -> bool:
        """Check whether this step should execute given the current context."""
        if self.condition is None:
            return True
        return self.condition(context)


@dataclass
class ChainResult:
    """Result from executing a prompt chain."""

    final_output: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    def get_step_output(self, step_name: str) -> Optional[str]:
        """Get the output of a specific step by name."""
        for step in self.steps:
            if step.get("name") == step_name:
                return step.get("output")
        return None


class PromptChain:
    """Compose multi-step prompt pipelines.

    Each step receives the accumulated context from previous steps
    and produces output that feeds into subsequent steps.

    Usage:
        chain = PromptChain([
            ChainStep(extract_template, output_key="entities"),
            ChainStep(classify_template, output_key="domain"),
            ChainStep(summarize_template, output_key="summary"),
        ])

        # Build all prompts (without calling an LLM)
        prompts = chain.build(text="Some input text")

        # Or simulate with a runner function
        result = chain.run(
            runner=my_llm_call,
            text="Some input text",
        )
    """

    def __init__(self, steps: list[ChainStep]):
        self.steps = steps

    def build(self, **initial_vars) -> list[dict[str, str]]:
        """Build all prompts in the chain without executing them.

        Renders each step's template using the accumulated context.
        Later steps can reference earlier steps' output_keys as variables,
        but since no LLM is called, those will use placeholder values.

        Args:
            **initial_vars: Initial variables available to all steps.

        Returns:
            List of dicts with 'name', 'prompt', and 'output_key' for each step.
        """
        context = dict(initial_vars)
        prompts = []

        for step in self.steps:
            if not step.should_run(context):
                continue
            prompt = step.template.render(**context)
            prompts.append({
                "name": step.name,
                "prompt": prompt,
                "output_key": step.output_key,
            })
            # Placeholder for the output
            context[step.output_key] = f"[{step.output_key}]"

        return prompts

    def run(
        self,
        runner: Callable[[str, Optional[str]], str],
        **initial_vars,
    ) -> ChainResult:
        """Execute the chain with an LLM runner function.

        Args:
            runner: A callable that takes (prompt, system_prompt) and returns
                    the LLM's response string.
            **initial_vars: Initial variables available to all steps.

        Returns:
            ChainResult with all step outputs and the final result.
        """
        context = dict(initial_vars)
        step_results = []

        for step in self.steps:
            if not step.should_run(context):
                step_results.append({
                    "name": step.name,
                    "skipped": True,
                })
                continue

            prompt = step.template.render(**context)
            system = step.template.system_prompt

            output = runner(prompt, system)

            if step.transform:
                output = step.transform(output)

            context[step.output_key] = output
            step_results.append({
                "name": step.name,
                "prompt": prompt,
                "output": output,
                "output_key": step.output_key,
            })

        final_output = step_results[-1].get("output", "") if step_results else ""

        return ChainResult(
            final_output=str(final_output),
            steps=step_results,
            context=context,
        )

    def add_step(self, step: ChainStep) -> "PromptChain":
        """Add a step and return self for chaining."""
        self.steps.append(step)
        return self

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)
