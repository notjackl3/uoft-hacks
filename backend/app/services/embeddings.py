from __future__ import annotations

import logging
from typing import List

import anyio

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingsError(RuntimeError):
    pass


async def embed_text(text: str) -> List[float]:
    """
    Generate embedding using Voyage AI.

    Returns a list of floats (voyage-2 is typically 1024 dims).
    """

    def _embed_sync() -> List[float]:
        try:
            import voyageai  # type: ignore

            vo = voyageai.Client(api_key=settings.voyage_api_key)
            result = vo.embed([text], model="voyage-2")
            return result.embeddings[0]
        except Exception as e:  # pragma: no cover (exact SDK exceptions vary)
            raise EmbeddingsError(str(e)) from e

    try:
        return await anyio.to_thread.run_sync(_embed_sync)
    except EmbeddingsError:
        raise
    except Exception as e:  # pragma: no cover
        logger.exception("Voyage embedding failed")
        raise EmbeddingsError(str(e)) from e

