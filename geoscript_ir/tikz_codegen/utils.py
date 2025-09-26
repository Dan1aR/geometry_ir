import re
import unicodedata
from typing import List

_MATH_DELIM_RE = re.compile(r'(?<!\\)(\$\$|\$)')  # matches unescaped $ or $$
_CYRILLIC_RUN_RE = re.compile(r'[А-Яа-яЁё]+')

# 1) Геометрический приоритет (семантика имён точек/вершин)
_GEOM_PRIORITY = str.maketrans({
    # Прописные
    'А':'A','В':'B','С':'C','Д':'D','Е':'E','К':'K','М':'M','Н':'N','О':'O','Р':'R','Т':'T','У':'U','Х':'X',
    # Строчные
    'а':'a','в':'b','с':'c','д':'d','е':'e','к':'k','м':'m','н':'n','о':'o','р':'r','т':'t','у':'u','х':'x',
})

# 2) Общий гомоглиф-мэппинг (визуальная близость), применяется ПОСЛЕ приоритета
_GENERIC_HOMOGLYPHS = str.maketrans({
    # Прописные (оставили только то, что не конфликтует с приоритетами)
    'Ё':'E','І':'I','Ї':'I','Й':'I',
    # Строчные
    'ё':'e','і':'i','ї':'i','й':'i',
})

def _strip_combining(text: str) -> str:
    text = unicodedata.normalize('NFC', text)
    return ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')

def _escape_text_segment(text: str) -> str:
    text = _strip_combining(text)
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

def _convert_cyrillic_in_math(s: str) -> str:
    """
    Внутри math:
    1) геометрический приоритет (семантика имён точек/вершин),
    2) общий гомоглиф-мэппинг,
    3) остатки кириллицы -> \text{…}
    """
    # 1) приоритетная транслитерация
    step1 = s.translate(_GEOM_PRIORITY)
    # 2) общий гомоглиф-мэппинг
    step2 = step1.translate(_GENERIC_HOMOGLYPHS)

    # 3) завернуть оставшиеся кириллические *раны* в \text{…}
    def _wrap(m: re.Match) -> str:
        return r'\text{' + m.group(0) + r'}'
    return _CYRILLIC_RUN_RE.sub(_wrap, step2)

def latex_escape_keep_math(s: str) -> str:
    """
    Escape для LaTeX-текста с сохранением math и корректной обработкой кириллицы.
    """
    parts: List[str] = []
    pos = 0
    in_math = False
    current_delim = None  # '$' or '$$'

    for m in _MATH_DELIM_RE.finditer(s):
        delim = m.group(1)
        start, end = m.span()

        chunk = s[pos:start]
        parts.append(_convert_cyrillic_in_math(chunk) if in_math else _escape_text_segment(chunk))

        parts.append(delim)
        if not in_math:
            in_math = True
            current_delim = delim
        else:
            if delim == current_delim:
                in_math = False
                current_delim = None
            # иначе: другой делимитер — оставляем как содержимое math
        pos = end

    tail = s[pos:]
    parts.append(_convert_cyrillic_in_math(tail) if in_math else _escape_text_segment(tail))
    return ''.join(parts)
