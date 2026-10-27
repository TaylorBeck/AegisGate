"""In-process quantized ONNX embedder (`BAAI/bge-small-en-v1.5`, 384-d)."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.typing import NDArray

from app.engine._fastembed import TextEmbedding

_DIM = 384
_EPS = np.float32(1e-12)


class LocalVectorEngine:
    """Process-wide ONNX embedding singleton.

    `embed()` runs on a dedicated thread so the event loop can keep streaming
    tokens. The session is only invoked from that thread after startup.
    """

    dim: int = _DIM

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: str | None = None,
        threads: int = 4,
    ) -> None:
        self._model = TextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=threads,
        )
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="aegis-onnx"
        )
        self._closed = False
        self.warm: bool = False
        self.warmup_ms: float = 0.0

    @property
    def executor(self) -> ThreadPoolExecutor:
        return self._executor

    def embed_batch_sync(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed `texts` with `embed()` and L2-normalize to unit `float32`."""
        if not texts:
            return np.empty((0, _DIM), dtype=np.float32)
        stacked = np.stack(list(self._model.embed(texts)), axis=0)
        vecs: NDArray[np.float32] = np.asarray(stacked, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, _EPS)
        normalized = vecs / norms
        result: NDArray[np.float32] = np.asarray(normalized, dtype=np.float32)
        return result

    def warmup(self) -> float:
        """Run a dummy embedding so the first user request skips cold start."""
        started = time.perf_counter()
        _ = self.embed_batch_sync(["ok"])
        self.warmup_ms = (time.perf_counter() - started) * 1000.0
        self.warm = True
        return self.warmup_ms

    async def warmup_async(self) -> float:
        """Warm the model on the ONNX thread."""
        loop = asyncio.get_running_loop()
        elapsed: float = await loop.run_in_executor(self._executor, self.warmup)
        return elapsed

    async def embed(self, text: str) -> NDArray[np.float32]:
        """Embed one string off the event loop and return a unit float32 vector."""
        loop = asyncio.get_running_loop()
        vecs: NDArray[np.float32] = await loop.run_in_executor(
            self._executor,
            self.embed_batch_sync,
            [text],
        )
        row: NDArray[np.float32] = np.ascontiguousarray(vecs[0], dtype=np.float32)
        return row

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)
