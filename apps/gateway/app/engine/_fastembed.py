"""Typed facade over the untyped `fastembed` package.

`TextEmbedding` has no `quantize=` kwarg. `BAAI/bge-small-en-v1.5` already
ships as a quantized ONNX artifact (`qdrant/bge-small-en-v1.5-onnx-q`).
Call `embed()` (symmetric) rather than `query_embed()`, which prepends
bge's retrieval prefix and would desynchronize cache and routing vectors.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
from fastembed import TextEmbedding as _TextEmbedding
from numpy.typing import NDArray


class TextEmbedding:
    """Narrow, typed wrapper around `fastembed.TextEmbedding`."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: str | None = None,
        threads: int | None = None,
    ) -> None:
        self._inner = _TextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=threads,
        )

    def embed(self, documents: Iterable[str]) -> Iterator[NDArray[np.float32]]:
        for item in self._inner.embed(documents):
            array: NDArray[np.float32] = np.asarray(item, dtype=np.float32)
            yield array
