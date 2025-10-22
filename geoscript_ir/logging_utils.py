from __future__ import annotations

import inspect
import logging
import reprlib
from functools import wraps
from typing import Any, Callable, Iterable, MutableMapping, Optional, Sequence, Set, Tuple, TypeVar, cast

try:  # pragma: no cover - optional dependency handling
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - numpy may be unavailable
    _np = None

F = TypeVar("F", bound=Callable[..., Any])

_repr = reprlib.Repr()
_repr.maxother = 160
_repr.maxdict = 10
_repr.maxlist = 10
_repr.maxtuple = 10
_repr.maxset = 10


def _sequence_brackets(value: Sequence[Any]) -> Tuple[str, str]:
    if isinstance(value, tuple):
        return "(", ")"
    if isinstance(value, set):
        return "{", "}"
    if isinstance(value, frozenset):
        return "frozenset({", "})"
    return "[", "]"


def _safe_repr(value: Any, *, max_items: int = 5, max_length: int = 400) -> str:
    if _np is not None and isinstance(value, _np.ndarray):  # type: ignore[arg-type]
        try:
            size = int(value.size)
        except Exception:  # pragma: no cover - defensive
            size = -1
        summary_parts = [
            f"ndarray(shape={tuple(value.shape)}, dtype={value.dtype})",
            f"size={size if size >= 0 else 'unknown'}",
        ]
        if size == 0:
            summary = ", ".join(summary_parts)
        elif 0 < size <= max_items:
            try:
                summary = ", ".join(summary_parts + [f"values={_repr.repr(value.tolist())}"])
            except Exception:  # pragma: no cover - defensive
                summary = ", ".join(summary_parts)
        else:
            try:
                min_val = float(value.min())
                max_val = float(value.max())
                summary = ", ".join(summary_parts + [f"min={min_val:.6g}", f"max={max_val:.6g}"])
            except Exception:  # pragma: no cover - defensive
                summary = ", ".join(summary_parts)
        return summary

    if isinstance(value, dict):
        items = []
        for idx, (key, val) in enumerate(value.items()):
            if idx >= max_items:
                items.append("...")
                break
            items.append(f"{_safe_repr(key)}: {_safe_repr(val)}")
        return "{" + ", ".join(items) + "}"

    if isinstance(value, (list, tuple, set, frozenset)):
        open_br, close_br = _sequence_brackets(value)  # type: ignore[arg-type]
        items = []
        for idx, item in enumerate(value):
            if idx >= max_items:
                items.append("...")
                break
            items.append(_safe_repr(item))
        return f"{open_br}{', '.join(items)}{close_br}"

    try:
        rendered = _repr.repr(value)
    except Exception as exc:  # pragma: no cover - defensive
        rendered = f"<repr-error {exc!r}>"
    if len(rendered) > max_length:
        return rendered[:max_length] + "... (truncated)"
    return rendered


def _format_arguments(args: Sequence[Any], kwargs: MutableMapping[str, Any]) -> str:
    parts = []
    if args:
        parts.append("args=[" + ", ".join(_safe_repr(arg) for arg in args) + "]")
    if kwargs:
        parts.append(
            "kwargs={"
            + ", ".join(f"{key}={_safe_repr(value)}" for key, value in kwargs.items())
            + "}"
        )
    if not parts:
        return "no-args"
    return ", ".join(parts)


def debug_log_call(
    logger: logging.Logger, *, name: Optional[str] = None, log_result: bool = True
) -> Callable[[F], F]:
    """Return a decorator that emits detailed DEBUG logs for function calls."""

    def decorator(func: F) -> F:
        if getattr(func, "_debug_logging_wrapped", False):
            return func

        qualname = name or getattr(func, "__qualname__", getattr(func, "__name__", "<callable>"))

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Entering %s (%s)", qualname, _format_arguments(args, kwargs))
            try:
                result = func(*args, **kwargs)
            except Exception:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.exception("Exception in %s", qualname)
                raise
            if logger.isEnabledFor(logging.DEBUG):
                if log_result:
                    logger.debug("Exiting %s -> %s", qualname, _safe_repr(result))
                else:
                    logger.debug("Exiting %s", qualname)
            return result

        setattr(wrapper, "_debug_logging_wrapped", True)
        return cast(F, wrapper)

    return decorator


def _apply_debug_logging_to_class(
    cls: type,
    logger: logging.Logger,
    skip: Set[str],
) -> None:
    for attr_name, attr_value in list(cls.__dict__.items()):
        if attr_name.startswith("__") and attr_name.endswith("__"):
            continue
        qualified = f"{cls.__name__}.{attr_name}"
        if attr_name in skip or qualified in skip:
            continue
        if isinstance(attr_value, staticmethod):
            func = attr_value.__func__
            if getattr(func, "__module__", None) != cls.__module__:
                continue
            wrapped = debug_log_call(logger, name=qualified)(func)
            setattr(cls, attr_name, staticmethod(wrapped))
        elif isinstance(attr_value, classmethod):
            func = attr_value.__func__
            if getattr(func, "__module__", None) != cls.__module__:
                continue
            wrapped = debug_log_call(logger, name=qualified)(func)
            setattr(cls, attr_name, classmethod(wrapped))
        elif inspect.isfunction(attr_value) and getattr(attr_value, "__module__", None) == cls.__module__:
            wrapped = debug_log_call(logger, name=qualified)(attr_value)
            setattr(cls, attr_name, wrapped)


def apply_debug_logging(
    namespace: MutableMapping[str, Any],
    *,
    logger: Optional[logging.Logger] = None,
    skip: Optional[Iterable[str]] = None,
    wrap_methods: bool = True,
) -> None:
    """Wrap module-level callables with verbose DEBUG logging."""

    module_name = namespace.get("__name__")
    if not isinstance(module_name, str):  # pragma: no cover - defensive
        module_name = None
    logger = logger or logging.getLogger(module_name or __name__)
    skip_set: Set[str] = set(skip or [])

    for name, value in list(namespace.items()):
        if name in skip_set:
            continue
        if inspect.isfunction(value) and getattr(value, "__module__", None) == module_name:
            namespace[name] = debug_log_call(logger, name=name)(value)
        elif wrap_methods and inspect.isclass(value) and getattr(value, "__module__", None) == module_name:
            _apply_debug_logging_to_class(value, logger, skip_set)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Verbose debug logging enabled for %s", module_name or "<unknown module>")
