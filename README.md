# Prompt Engineering Library

A Python library for structured prompt engineering with LLMs. Build, compose, validate, and evaluate prompts programmatically.

## Features

- **Prompt Templates** -- type-safe, reusable prompt templates with variables
- **Prompt Chains** -- compose multi-step reasoning pipelines
- **Few-shot Manager** -- organize and select in-context examples
- **Output Parsing** -- extract structured data from LLM responses
- **Evaluation** -- score prompt quality against ground truth
- **Template Library** -- pre-built templates for common tasks

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from prompt_lib import PromptTemplate, FewShotManager, PromptChain

# Create a template
template = PromptTemplate(
    name="sentiment",
    template="Analyze the sentiment of this text: {text}\nSentiment:",
    variables=["text"],
)

# Render with variables
prompt = template.render(text="I love this product!")
# "Analyze the sentiment of this text: I love this product!\nSentiment:"

# Add few-shot examples
few_shot = FewShotManager()
few_shot.add("Great quality!", "POSITIVE")
few_shot.add("Terrible experience.", "NEGATIVE")

prompt = template.render(
    text="Pretty good overall.",
    examples=few_shot.format(),
)
```

## Prompt Chains

```python
# Multi-step reasoning
chain = PromptChain([
    PromptTemplate(
        name="extract",
        template="Extract the key entities from: {text}\nEntities:",
    ),
    PromptTemplate(
        name="classify",
        template="Given these entities: {entities}\nClassify the domain:",
    ),
    PromptTemplate(
        name="summarize",
        template="Summarize this {domain} text: {text}\nSummary:",
    ),
])
```

## Built-in Templates

```python
from prompt_lib.templates import TEMPLATES

# Available templates
for name, desc in TEMPLATES.items():
    print(f"{name}: {desc}")

# Use a built-in template
template = TEMPLATES["summarize"]
prompt = template.render(text="Long article here...", length="3 sentences")
```

## License

MIT
