from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from paibox.types import Shape

"""Handful utilities."""


def check_elem_unique(obj: Any) -> bool:
    """Check whether a object consists of unique elements"""
    if hasattr(obj, "__iter__"):
        return len(obj) == len(set(obj))

    if isinstance(obj, dict):
        return len(obj) == len(set(obj.values()))

    if hasattr(obj, "__contains__"):
        seen = set()
        for elem in obj:
            if elem in seen:
                return False
            seen.add(elem)

        return True

    raise TypeError(f"unsupported type: {type(obj)}.")


def count_unique_elem(obj: Iterable[Any]) -> int:
    seen = set()
    for item in obj:
        seen.add(item)

    return len(seen)


def merge_unique_ordered(list1: List[Any], list2: List[Any]) -> List[Any]:
    seen = set()
    result = []

    for item in list1 + list2:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


def check_attr_same(obj: Sequence[Any], attr: str) -> bool:
    return all(getattr(obj[0], attr) == getattr(item, attr) for item in obj)


def check_elem_same(obj: Any) -> bool:
    if hasattr(obj, "__iter__") or hasattr(obj, "__contains__"):
        return len(set(obj)) == 1

    if isinstance(obj, dict):
        return len(set(obj.values())) == 1

    raise TypeError(f"unsupported type: {type(obj)}.")


def is_nested_obj(obj_on_top: Any) -> bool:
    """Check whether a object is nested"""
    return any(
        isinstance(item, Iterable) and not isinstance(item, str) for item in obj_on_top
    )


def shape2num(shape: Shape) -> int:
    """Convert a shape to a number"""
    if isinstance(shape, int):
        return shape
    elif isinstance(shape, np.ndarray):
        return int(np.prod(shape))
    else:
        a = 1
        for b in shape:
            a *= b

        return a


def as_shape(x, min_dim: int = 0) -> Tuple[int, ...]:
    """Return a tuple if `x` is iterable or `(x,)` if `x` is integer."""
    if is_integer(x):
        _shape = (int(x),)
    elif is_iterable(x):
        _shape = tuple(int(e) for e in x)
    else:
        raise ValueError(f"{x} cannot be safely converted to a shape.")

    if len(_shape) < min_dim:
        _shape = (1,) * (min_dim - len(_shape)) + _shape

    return _shape


def is_shape(x, shape: Shape) -> bool:
    if not is_array_like(x):
        raise TypeError(f"only support an array-like type: {x}.")

    _x = np.asarray(x)
    return _x.shape == as_shape(shape)


def is_integer(obj: Any) -> bool:
    return isinstance(obj, (int, np.integer))


def is_number(obj: Any) -> bool:
    return is_integer(obj) or isinstance(obj, (float, np.number))


def is_array_like(obj: Any) -> bool:
    return (
        isinstance(obj, np.ndarray) or is_number(obj) or isinstance(obj, (list, tuple))
    )


def is_iterable(obj: Any) -> bool:
    """Check whether obj is an iterable."""
    if isinstance(obj, np.ndarray):
        return obj.ndim > 0

    return isinstance(obj, Iterable)


def fn_sgn(a, b) -> int:
    """Signal function."""
    return 1 if a > b else -1 if a < b else 0


def arg_check_pos(arg: int, desc: Optional[str] = None) -> None:
    _desc = "value" if desc is None else f"{desc}"
    if arg < 1:
        raise ValueError(f"{_desc} must be positive, but got {arg}.")


def arg_check_non_pos(arg: int, desc: Optional[str] = None) -> None:
    _desc = "value" if desc is None else f"{desc}"
    if arg > 0:
        raise ValueError(f"{_desc} must be non-positive, but got {arg}.")


def arg_check_neg(arg: int, desc: Optional[str] = None) -> None:
    _desc = "value" if desc is None else f"{desc}"
    if arg > -1:
        raise ValueError(f"{_desc} must be negative, but got {arg}.")


def arg_check_non_neg(arg: int, desc: Optional[str] = None) -> None:
    _desc = "value" if desc is None else f"{desc}"
    if arg < 0:
        raise ValueError(f"{_desc} must be non-negative, but got {arg}.")
