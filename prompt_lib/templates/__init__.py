"""Pre-built prompt templates for common NLP tasks."""

from prompt_lib.template import PromptTemplate

TEMPLATES = {
    "summarize": PromptTemplate(
        name="summarize",
        template=(
            "Summarize the following text in {length}.\n\n"
            "Text: {text}\n\n"
            "Summary:"
        ),
        description="Text summarization with configurable length.",
        variables=["text", "length"],
    ),
    "translate": PromptTemplate(
        name="translate",
        template=(
            "Translate the following text from {source_lang} to {target_lang}.\n\n"
            "Text: {text}\n\n"
            "Translation:"
        ),
        description="Translation between language pairs.",
        variables=["text", "source_lang", "target_lang"],
    ),
    "sentiment": PromptTemplate(
        name="sentiment",
        template=(
            "Analyze the sentiment of the following text. "
            "Classify it as POSITIVE, NEGATIVE, or NEUTRAL "
            "and provide a confidence score (0-1).\n\n"
            "Text: {text}\n\n"
            "Sentiment:"
        ),
        description="Sentiment analysis with confidence.",
    ),
    "ner": PromptTemplate(
        name="ner",
        template=(
            "Extract all named entities from the following text. "
            "For each entity, provide the text span, entity type "
            "(PERSON, ORG, LOC, DATE, etc.), and position.\n\n"
            "Text: {text}\n\n"
            "Entities (as JSON array):"
        ),
        description="Named entity recognition.",
    ),
    "classify": PromptTemplate(
        name="classify",
        template=(
            "Classify the following text into one of these categories: "
            "{categories}\n\n"
            "Text: {text}\n\n"
            "Category:"
        ),
        description="Text classification with custom categories.",
    ),
    "qa": PromptTemplate(
        name="qa",
        template=(
            "Answer the question based on the given context.\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
        description="Question answering with context.",
    ),
    "chain_of_thought": PromptTemplate(
        name="chain_of_thought",
        template=(
            "Solve the following problem step by step. "
            "Show your reasoning before giving the final answer.\n\n"
            "Problem: {problem}\n\n"
            "Let's think step by step:\n"
        ),
        description="Chain-of-thought reasoning.",
    ),
    "code_review": PromptTemplate(
        name="code_review",
        template=(
            "Review the following {language} code. "
            "Check for bugs, performance issues, and style. "
            "Suggest improvements.\n\n"
            "```{language}\n{code}\n```\n\n"
            "Review:"
        ),
        description="Code review with suggestions.",
    ),
    "rewrite": PromptTemplate(
        name="rewrite",
        template=(
            "Rewrite the following text to be more {style}. "
            "Keep the core meaning but adjust the tone and language.\n\n"
            "Original: {text}\n\n"
            "Rewritten:"
        ),
        description="Text rewriting with target style.",
    ),
    "extract_keywords": PromptTemplate(
        name="extract_keywords",
        template=(
            "Extract the {n} most important keywords from the text.\n\n"
            "Text: {text}\n\n"
            "Keywords:"
        ),
        description="Keyword extraction.",
    ),
}
