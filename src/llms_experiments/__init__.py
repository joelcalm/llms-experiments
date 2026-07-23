"""Installable, configuration-driven LLM inference tools."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("llms-experiments")
except PackageNotFoundError:  # source checkout without an editable install
    __version__ = "0.2.0"

__all__ = ["__version__"]
