"""Problem-level split and transition-cache infrastructure."""

from .split_registry import (
    SplitLeakageError,
    SplitRatios,
    assert_rows_respect_registry,
    create_split_registry,
    load_split_registry,
    validate_split_registry,
    write_split_registry,
)

__all__ = [
    "SplitLeakageError",
    "SplitRatios",
    "assert_rows_respect_registry",
    "create_split_registry",
    "load_split_registry",
    "validate_split_registry",
    "write_split_registry",
]
