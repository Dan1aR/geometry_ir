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