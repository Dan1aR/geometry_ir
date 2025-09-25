import re
import unicodedata
from typing import List

_MATH_DELIM_RE = re.compile(r'(?<!\\)(\$\$|\$)')  # matches unescaped $ or $$

def _strip_combining(text: str) -> str:
    # Normalize, then drop all Unicode combining marks (Mn)
    text = unicodedata.normalize('NFC', text)
    return ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')

def _escape_text_segment(text: str) -> str:
    # Remove combining marks (e.g., U+0301) and escape LaTeX specials, but DO NOT touch $
    text = _strip_combining(text)
    # Minimal, safe escapes for text mode:
    repl = {
        '\\': r'\textbackslash{}',
        '&':  r'\&',
        '%':  r'\%',
        '#':  r'\#',
        '_':  r'\_',
        '{':  r'\{',
        '}':  r'\}',
        '~':  r'\textasciitilde{}',
        '^':  r'\textasciicircum{}',
    }
    return ''.join(repl.get(c, c) for c in text)

def latex_escape_keep_math(s: str) -> str:
    """
    Escape for LaTeX text while preserving math.
    - Splits on unescaped $ or $$.
    - Text segments: strip combining marks (e.g. U+0301) and escape specials.
    - Math segments: returned verbatim between the same delimiters.
    """
    parts: List[str] = []
    pos = 0
    in_math = False
    current_delim = None  # '$' or '$$'

    for m in _MATH_DELIM_RE.finditer(s):
        delim = m.group(1)
        start, end = m.span()

        # Text before this delimiter
        chunk = s[pos:start]
        if in_math:
            # inside math -> keep verbatim
            parts.append(chunk)
        else:
            # outside math -> escape
            parts.append(_escape_text_segment(chunk))

        # Emit the delimiter and flip state
        parts.append(delim)
        if not in_math:
            in_math = True
            current_delim = delim
        else:
            # Only close if the same delimiter repeats (i.e., $$ closes $$)
            if delim == current_delim:
                in_math = False
                current_delim = None
            else:
                # Different delimiter inside math: treat as literal content
                # (rare; we keep it and remain in math)
                pass

        pos = end

    # Tail after the last delimiter
    tail = s[pos:]
    if in_math:
        parts.append(tail)          # still inside math: verbatim
    else:
        parts.append(_escape_text_segment(tail))  # outside math: escape

    return ''.join(parts)
