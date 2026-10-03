import json
import logging
import os
from typing import Any


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger("iRAG_logger")


def log_event(event: str, **fields: Any) -> None:
    """Write one machine-readable event without losing the normal log envelope."""
    logger.info(json.dumps({"event": event, **fields}, sort_keys=True, default=str))
