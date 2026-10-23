from collections.abc import Callable
import dataclasses
import functools
import inspect
import re
from typing import Any, ParamSpec, TypeVar
import flax.nnx as nnx
import jax

P = ParamSpec("P")
R = TypeVar("R")


def module_jit(meth: Callable[P, R], *jit_args, **jit_kwargs) -> Callable[P, R]:
    if not (inspect.ismethod(meth) and isinstance(meth.__self__, nnx.Module)):
        raise ValueError(
            "module_jit must only be used on bound methods of nnx.Modules."
        )
    graphdef, state = nnx.split(meth.__self__)

    def fun(state: nnx.State, *args: P.args, **kwargs: P.kwargs) -> R:
        module = nnx.merge(graphdef, state)
        return meth.__func__(module, *args, **kwargs)

    jitted_fn = jax.jit(fun, *jit_args, **jit_kwargs)

    @functools.wraps(meth)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return jitted_fn(state, *args, **kwargs)

    return wrapper


@dataclasses.dataclass(frozen=True)
class PathRegex:
    pattern: str | re.Pattern
    sep: str = "/"

    def __post_init__(self):
        if not isinstance(self.pattern, re.Pattern):
            object.__setattr__(self, "pattern", re.compile(self.pattern))

    def __call__(self, path: nnx.filterlib.PathParts, x: Any) -> bool:
        joined_path = self.sep.join((str(x) for x in path))
        assert isinstance(self.pattern, re.Pattern)
        return self.pattern.fullmatch(joined_path) is not None


def state_map(
    state: nnx.State, filter: nnx.filterlib.Filter, fn: Callable[[Any], Any]
) -> nnx.State:
    filtered_keys = set(state.filter(filter).flat_state())
    return state.map(lambda k, v: fn(v) if k in filtered_keys else v)
