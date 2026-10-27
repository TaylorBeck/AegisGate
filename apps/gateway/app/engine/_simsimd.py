"""Typed facade over the untyped `simsimd` package (consumed in milestone 2)."""

from typing import Any

import simsimd as _simsimd

cosine: Any = _simsimd.cosine
dot: Any = getattr(_simsimd, "dot", None)
