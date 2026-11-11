"""Prompt template engine with variable validation and token counting."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import tiktoken
except ImportError:
    tiktoken = None


@dataclass
class TemplateVariable:
    """A variable slot in a prompt template."""

    name: str
    description: str = ""
    required: bool = True
    default: Any = None
    validator: Optional[callable] = None

    def validate(self, value: Any) -> bool:
        if self.required and value is None:
            raise ValueError(f"Variable '{self.name}' is required")
        if self.validator and value is not None:
            if not self.validator(value):
                raise ValueError(
                    f"Validation failed for variable '{self.name}': {value}"
                )
        return True


@dataclass
class PromptTemplate:
    """A reusable, type-safe prompt template.

    Supports {variable} placeholders, optional sections with
    {% if var %}...{% endif %}, and token counting.
    """

    name: str
    template: str
    description: str = ""
    variables: list[str] = field(default_factory=list)
    variable_defs: dict[str, TemplateVariable] = field(default_factory=dict)
    system_prompt: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.variables:
            self.variables = self._extract_variables()
        for var_name in self.variables:
            if var_name not in self.variable_defs:
                self.variable_defs[var_name] = TemplateVariable(name=var_name)

    def _extract_variables(self) -> list[str]:
        """Extract {variable} placeholders from the template."""
        pattern = r"\{(\w+)\}"
        found = re.findall(pattern, self.template)
        seen = set()
        unique = []
        for v in found:
            if v not in seen:
                seen.add(v)
                unique.append(v)
        return unique

    def render(self, **kwargs) -> str:
        """Render the template with provided variables.

        Args:
            **kwargs: Variable values to fill into the template.

        Returns:
            The rendered prompt string.

        Raises:
            ValueError: If a required variable is missing or validation fails.
        """
        for var_name, var_def in self.variable_defs.items():
            value = kwargs.get(var_name, var_def.default)
            var_def.validate(value)

        result = self.template

        # Handle conditional blocks: {% if var %}...{% endif %}
        conditional_pattern = r"\{%\s*if\s+(\w+)\s*%\}(.*?)\{%\s*endif\s*%\}"
        for match in re.finditer(conditional_pattern, result, re.DOTALL):
            var_name = match.group(1)
            block_content = match.group(2)
            if kwargs.get(var_name):
                result = result.replace(match.group(0), block_content.strip())
            else:
                result = result.replace(match.group(0), "")

        # Fill in variables
        for var_name in self.variables:
            value = kwargs.get(var_name, self.variable_defs[var_name].default)
            if value is not None:
                result = result.replace(f"{{{var_name}}}", str(value))

        # Clean up extra whitespace from removed conditionals
        result = re.sub(r"\n{3,}", "\n\n", result).strip()

        return result

    def count_tokens(self, rendered_text: Optional[str] = None, model: str = "gpt-4") -> int:
        """Count tokens in the rendered (or raw) template.

        Args:
            rendered_text: Pre-rendered text to count. If None, counts the raw template.
            model: Tokenizer model name for tiktoken.

        Returns:
            Token count, or word-based estimate if tiktoken is unavailable.
        """
        text = rendered_text or self.template
        if tiktoken:
            try:
                enc = tiktoken.encoding_for_model(model)
                return len(enc.encode(text))
            except Exception:
                pass
        # Fallback: rough word-based estimate
        return int(len(text.split()) * 1.3)

    def with_system(self, system_prompt: str) -> "PromptTemplate":
        """Return a copy of this template with a system prompt attached."""
        return PromptTemplate(
            name=self.name,
            template=self.template,
            description=self.description,
            variables=self.variables.copy(),
            variable_defs=dict(self.variable_defs),
            system_prompt=system_prompt,
            metadata=dict(self.metadata),
        )

    def extend(self, suffix: str, name: Optional[str] = None) -> "PromptTemplate":
        """Create a new template by appending text to this one."""
        new_template = self.template + "\n" + suffix
        return PromptTemplate(
            name=name or f"{self.name}_extended",
            template=new_template,
            description=self.description,
            system_prompt=self.system_prompt,
            metadata=dict(self.metadata),
        )

    def __str__(self) -> str:
        return f"PromptTemplate(name='{self.name}', vars={self.variables})"
