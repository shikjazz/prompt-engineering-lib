from setuptools import setup, find_packages

setup(
    name="prompt-engineering-lib",
    version="0.1.0",
    description="A Python library for structured prompt engineering with LLMs",
    author="Shikhar Sharma",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "jinja2>=3.1.0",
        "pydantic>=2.0.0",
        "tiktoken>=0.5.0",
        "numpy>=1.24.0",
        "click>=8.1.0",
        "rich>=13.0.0",
    ],
    entry_points={
        "console_scripts": [
            "promptlib=prompt_lib.cli:main",
        ],
    },
)
