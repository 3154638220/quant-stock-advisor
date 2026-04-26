"""Compat shim for environments that only provide stdlib ``tomllib``."""

from __future__ import annotations

from tomllib import TOMLDecodeError, load, loads

__all__ = ["TOMLDecodeError", "load", "loads"]
