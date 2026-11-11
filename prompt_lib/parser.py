"""Output parsers for extracting structured data from LLM responses."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ParseResult:
    """Result of parsing an LLM output."""

    data: Any
    raw_text: str
    success: bool = True
    error: Optional[str] = None


class OutputParser(ABC):
    """Base class for output parsers."""

    @abstractmethod
    def parse(self, text: str) -> ParseResult:
        """Parse LLM output into structured data."""
        ...

    @abstractmethod
    def get_format_instructions(self) -> str:
        """Return instructions to include in the prompt so the LLM
        produces output in the expected format."""
        ...


class JsonParser(OutputParser):
    """Parse JSON from LLM output.

    Handles common issues: markdown code fences, trailing commas,
    extra text around the JSON block.
    """

    def __init__(self, schema: Optional[dict] = None):
        self.schema = schema

    def parse(self, text: str) -> ParseResult:
        cleaned = self._extract_json(text)
        try:
            data = json.loads(cleaned)
            if self.schema:
                self._validate_schema(data)
            return ParseResult(data=data, raw_text=text)
        except (json.JSONDecodeError, ValueError) as e:
            return ParseResult(
                data=None, raw_text=text, success=False, error=str(e)
            )

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text, handling code fences and surrounding prose."""
        # Try to find JSON in code fences
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if fence_match:
            return fence_match.group(1).strip()

        # Try to find a JSON object or array
        for pattern in [r"(\{.*\})", r"(\[.*\])"]:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1)

        return text.strip()

    def _validate_schema(self, data: Any) -> None:
        """Basic schema validation: check required keys exist."""
        if not isinstance(self.schema, dict):
            return
        required = self.schema.get("required", [])
        if isinstance(data, dict):
            for key in required:
                if key not in data:
                    raise ValueError(f"Missing required key: {key}")

    def get_format_instructions(self) -> str:
        instructions = "Respond with valid JSON only. No additional text."
        if self.schema:
            instructions += f"\n\nExpected schema:\n{json.dumps(self.schema, indent=2)}"
        return instructions


class ListParser(OutputParser):
    """Parse a numbered or bulleted list from LLM output."""

    def __init__(self, min_items: int = 0, max_items: Optional[int] = None):
        self.min_items = min_items
        self.max_items = max_items

    def parse(self, text: str) -> ParseResult:
        items = []
        for line in text.strip().split("\n"):
            line = line.strip()
            # Remove bullet markers or numbering
            cleaned = re.sub(r"^(?:\d+[.)]\s*|[-*+]\s*)", "", line).strip()
            if cleaned:
                items.append(cleaned)

        if self.min_items and len(items) < self.min_items:
            return ParseResult(
                data=items,
                raw_text=text,
                success=False,
                error=f"Expected at least {self.min_items} items, got {len(items)}",
            )
        if self.max_items and len(items) > self.max_items:
            items = items[: self.max_items]

        return ParseResult(data=items, raw_text=text)

    def get_format_instructions(self) -> str:
        msg = "Respond with a numbered list, one item per line."
        if self.min_items:
            msg += f" Include at least {self.min_items} items."
        if self.max_items:
            msg += f" Include at most {self.max_items} items."
        return msg


class RegexParser(OutputParser):
    """Parse output using a regex pattern with named groups."""

    def __init__(self, pattern: str, flags: int = 0):
        self.pattern = pattern
        self.flags = flags

    def parse(self, text: str) -> ParseResult:
        match = re.search(self.pattern, text, self.flags)
        if match:
            data = match.groupdict() if match.groupdict() else match.groups()
            return ParseResult(data=data, raw_text=text)
        return ParseResult(
            data=None,
            raw_text=text,
            success=False,
            error=f"Pattern not found: {self.pattern}",
        )

    def get_format_instructions(self) -> str:
        return f"Format your response to match: {self.pattern}"


class KeyValueParser(OutputParser):
    """Parse key-value pairs from LLM output.

    Expects lines like:
        Key: Value
        Another Key: Another Value
    """

    def __init__(self, expected_keys: Optional[list[str]] = None, delimiter: str = ":"):
        self.expected_keys = expected_keys
        self.delimiter = delimiter

    def parse(self, text: str) -> ParseResult:
        data = {}
        for line in text.strip().split("\n"):
            if self.delimiter in line:
                key, _, value = line.partition(self.delimiter)
                data[key.strip()] = value.strip()

        if self.expected_keys:
            missing = [k for k in self.expected_keys if k not in data]
            if missing:
                return ParseResult(
                    data=data,
                    raw_text=text,
                    success=False,
                    error=f"Missing keys: {missing}",
                )

        return ParseResult(data=data, raw_text=text)

    def get_format_instructions(self) -> str:
        msg = f"Respond with key{self.delimiter} value pairs, one per line."
        if self.expected_keys:
            msg += f"\n\nRequired keys: {', '.join(self.expected_keys)}"
        return msg
