from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolicNumber:
    """Numeric value with a symbolic text representation."""

    text: str
    value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", float(self.value))

    def __float__(self) -> float:
        return float(self.value)

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"SymbolicNumber(text={self.text!r}, value={self.value!r})"
