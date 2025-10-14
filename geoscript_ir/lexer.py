import re
from typing import List, Tuple

Token = Tuple[str, str, int, int]  # (type, value, line, col)

SYMBOLS = {
    '[': 'LBRACK',
    ']': 'RBRACK',
    '(': 'LPAREN',
    ')': 'RPAREN',
    '{': 'LBRACE',
    '}': 'RBRACE',
    ',': 'COMMA',
    '-': 'DASH',
    ';': 'SEMI',
    '=': 'EQUAL',
    '*': 'STAR',
    ':': 'COLON',
}

WS = ' \t\r'

_id_re = re.compile(r'\\?[A-Za-z][A-Za-z0-9_]*')
_num_re = re.compile(r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?')
_str_re = re.compile(r'"([^"\\]|\\.)*"')  # double-quoted with escapes

def tokenize_line(s: str, line_no: int) -> List[Token]:
    tokens: List[Token] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        col = i + 1
        if ch == '#':
            break
        if ch in WS:
            i += 1
            continue
        if ch == '"':
            m = _str_re.match(s, i)
            if not m:
                raise SyntaxError(f'[line {line_no}, col {col}] unterminated string literal')
            raw = m.group(0)
            val = bytes(raw[1:-1], 'utf-8').decode('unicode_escape')
            tokens.append(('STRING', val, line_no, col))
            i = m.end()
            continue
        m = _num_re.match(s, i)
        if m:
            val = m.group(0)
            tokens.append(('NUMBER', val, line_no, col))
            i = m.end()
            continue
        m = _id_re.match(s, i)
        if m:
            val = m.group(0)
            if val.startswith('\\'):
                val = val[1:]
            tokens.append(('ID', val, line_no, col))
            i = m.end()
            continue
        if ch in SYMBOLS:
            tokens.append((SYMBOLS[ch], ch, line_no, col))
            i += 1
            continue
        raise SyntaxError(f'[line {line_no}, col {col}] unexpected character: {ch!r}')
    return tokens
