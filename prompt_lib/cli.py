"""CLI interface for the prompt engineering library."""

import json

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from prompt_lib.template import PromptTemplate
from prompt_lib.templates import TEMPLATES
from prompt_lib.evaluation import PromptEvaluator

console = Console()


@click.group()
def main():
    """Prompt Engineering Library -- build and evaluate prompts."""
    pass


@main.command()
def list_templates():
    """List all built-in prompt templates."""
    table = Table(title="Built-in Prompt Templates")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Variables", style="green")

    for name, template in TEMPLATES.items():
        table.add_row(
            name,
            template.description,
            ", ".join(template.variables),
        )

    console.print(table)


@main.command()
@click.argument("template_name")
@click.option("--var", "-v", multiple=True, help="Variable in key=value format")
def render(template_name: str, var: tuple):
    """Render a built-in template with variables."""
    if template_name not in TEMPLATES:
        console.print(f"[red]Unknown template: {template_name}[/red]")
        console.print(f"Available: {', '.join(TEMPLATES.keys())}")
        return

    variables = {}
    for v in var:
        key, _, value = v.partition("=")
        variables[key.strip()] = value.strip()

    template = TEMPLATES[template_name]
    try:
        result = template.render(**variables)
        console.print(Panel(result, title=f"Rendered: {template_name}"))
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")


@main.command()
@click.option("--prompt", "-p", required=True, help="The prompt used")
@click.option("--output", "-o", required=True, help="LLM output")
@click.option("--expected", "-e", required=True, help="Expected output")
@click.option(
    "--metrics",
    "-m",
    multiple=True,
    help="Metrics to use (exact_match, contains, word_overlap, length_ratio)",
)
def evaluate(prompt: str, output: str, expected: str, metrics: tuple):
    """Evaluate a prompt-output pair against expected output."""
    metric_list = list(metrics) if metrics else None
    evaluator = PromptEvaluator(metrics=metric_list)
    result = evaluator.evaluate(prompt=prompt, output=output, expected=expected)

    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="green")

    for name, score in result.scores.items():
        color = "green" if score >= 0.5 else "red"
        table.add_row(name, f"[{color}]{score:.4f}[/{color}]")

    table.add_row("Average", f"[bold]{result.average_score:.4f}[/bold]")
    table.add_row("Passed", "Yes" if result.passed() else "[red]No[/red]")

    console.print(table)


@main.command()
@click.argument("template_name")
def info(template_name: str):
    """Show details about a template."""
    if template_name not in TEMPLATES:
        console.print(f"[red]Unknown template: {template_name}[/red]")
        return

    t = TEMPLATES[template_name]
    console.print(Panel(
        f"[bold]{t.name}[/bold]\n\n"
        f"Description: {t.description}\n"
        f"Variables: {', '.join(t.variables)}\n\n"
        f"Template:\n{t.template}",
        title=f"Template: {template_name}",
    ))


if __name__ == "__main__":
    main()
