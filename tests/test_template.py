"""Tests for the prompt template engine."""

import pytest
from prompt_lib.template import PromptTemplate, TemplateVariable


def test_basic_render():
    t = PromptTemplate(
        name="test",
        template="Hello {name}, you are {age} years old.",
    )
    result = t.render(name="Alice", age=30)
    assert result == "Hello Alice, you are 30 years old."


def test_auto_extract_variables():
    t = PromptTemplate(
        name="test",
        template="Translate {text} from {src} to {tgt}",
    )
    assert t.variables == ["text", "src", "tgt"]


def test_conditional_block():
    t = PromptTemplate(
        name="test",
        template="Do this task.\n{% if context %}Context: {context}{% endif %}\nAnswer:",
    )
    with_ctx = t.render(context="some context")
    assert "Context: some context" in with_ctx

    without_ctx = t.render(context="")
    assert "Context:" not in without_ctx


def test_required_variable_raises():
    t = PromptTemplate(
        name="test",
        template="{name} is required",
        variable_defs={"name": TemplateVariable(name="name", required=True)},
    )
    with pytest.raises(ValueError, match="required"):
        t.render()


def test_default_value():
    t = PromptTemplate(
        name="test",
        template="Length: {length}",
        variable_defs={
            "length": TemplateVariable(name="length", default="3 sentences")
        },
    )
    result = t.render()
    assert result == "Length: 3 sentences"


def test_validator():
    t = PromptTemplate(
        name="test",
        template="Count: {n}",
        variable_defs={
            "n": TemplateVariable(
                name="n",
                validator=lambda x: isinstance(x, int) and x > 0,
            )
        },
    )
    result = t.render(n=5)
    assert result == "Count: 5"

    with pytest.raises(ValueError, match="Validation failed"):
        t.render(n=-1)


def test_with_system():
    t = PromptTemplate(name="test", template="Hello {name}")
    t2 = t.with_system("You are a helpful assistant.")
    assert t2.system_prompt == "You are a helpful assistant."
    assert t.system_prompt is None  # original unchanged


def test_extend():
    t = PromptTemplate(name="test", template="Part 1")
    t2 = t.extend("Part 2", name="extended")
    assert "Part 1" in t2.template
    assert "Part 2" in t2.template
    assert t2.name == "extended"


def test_count_tokens_fallback():
    t = PromptTemplate(name="test", template="Hello world this is a test")
    count = t.count_tokens()
    assert count > 0  # At least a rough estimate


def test_duplicate_variables():
    t = PromptTemplate(
        name="test",
        template="{x} and {y} and {x} again",
    )
    assert t.variables == ["x", "y"]  # No duplicates
