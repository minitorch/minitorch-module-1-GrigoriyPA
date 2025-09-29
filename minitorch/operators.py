"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, List

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    return x * y


def id(x: float) -> float:
    return x


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return -x


def lt(x: float, y: float) -> bool:
    return x < y


def eq(x: float, y: float) -> bool:
    return x == y


def max(x: float, y: float) -> float:
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))

    ex = math.exp(x)
    return ex / (1 + ex)


def relu(x: float) -> float:
    return max(x, 0)


def log(x: float) -> float:
    return math.log(x)


def exp(x: float) -> float:
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    return d / x


def inv(x: float) -> float:
    return 1 / x


def inv_back(x: float, d: float) -> float:
    return -d / (x ** 2)


def relu_back(x: float, d: float) -> float:
    return d if x > 0 else 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float], it: Iterable[float]) -> Iterable[float]:
    """Applies fn to every element in Iterable."""
    for x in it:
        yield fn(x)


def zipWith(fn: Callable[[float, float], float], it_left: Iterable[float], it_right: Iterable[float]) -> Iterable[float]:
    """Applies fn to pairs from left Iterable and right Iterable."""
    for x, y, in zip(it_left, it_right):
        yield fn(x, y)


def reduce(fn: Callable[[float, float], float], it: Iterable[float], start: float) -> float:
    """Reduces lst with fn starting from start."""
    result = start
    for x in it:
        result = fn(result, x)
    return result


def negList(lst: List[float]) -> List[float]:
    return list(map(neg, lst))


def addLists(left_lst: List[float], right_lst: List[float]) -> List[float]:
    return list(zipWith(add, left_lst, right_lst))


def sum(lst: List[float]) -> float:
    return reduce(add, lst, 0)


def prod(lst: List[float]) -> float:
    return reduce(mul, lst, 1)
