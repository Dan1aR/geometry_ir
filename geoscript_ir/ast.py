from dataclasses import dataclass, field, replace
from typing import Dict, Any, List

@dataclass
class Span:
    line: int
    col: int


@dataclass
class Stmt:
    kind: str
    span: Span
    data: Dict[str, Any] = field(default_factory=dict)
    opts: Dict[str, Any] = field(default_factory=dict)
    origin: str = 'source'  # 'source' or 'desugar(<kind>)'


@dataclass
class Program:
    stmts: List[Stmt] = field(default_factory=list)

    @property
    def source_program(self) -> "Program":
        """Return a shallow copy of self, but with stmts filtered to 'source' only."""
        return replace(
            self,
            stmts=[stmt for stmt in self.stmts if stmt.origin == "source"]
        )
