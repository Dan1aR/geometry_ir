from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple

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
    def source_stmts(self) -> List[Stmt]:
        """Return only statements that originate from the source program."""

        return [stmt for stmt in self.stmts if stmt.origin == 'source']