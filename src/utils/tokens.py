import tiktoken

from src.config import settings
from src.telemetry import prometheus_metrics
from src.telemetry.prometheus.metrics import (
    DeriverComponents,
    DeriverTaskTypes,
    TokenTypes,
)

_tokenizer: tiktoken.Encoding | None = None


def _get_tokenizer() -> tiktoken.Encoding | None:
    """Lazily load the tiktoken encoding to avoid network I/O at import time."""
    global _tokenizer
    if _tokenizer is None:
        try:
            _tokenizer = tiktoken.get_encoding("o200k_base")
        except Exception:
            return None
    return _tokenizer


def estimate_tokens(text: str | list[str] | None) -> int:
    """Estimate token count using tiktoken for text or list of strings."""
    if not text:
        return 0
    if isinstance(text, list):
        text = "\n".join(text)
    tokenizer = _get_tokenizer()
    if tokenizer is None:
        return len(text) // 4
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return len(text) // 4


def track_deriver_input_tokens(
    task_type: DeriverTaskTypes,
    components: dict[DeriverComponents, int],
) -> None:
    """
    Helper method to track input token components for a given task type.

    Args:
        task_type: The type of task
        components: Dict mapping component names to token counts
    """
    for component, token_count in components.items():
        # Prometheus metrics
        if settings.METRICS.ENABLED:
            prometheus_metrics.record_deriver_tokens(
                count=token_count,
                task_type=task_type.value,
                token_type=TokenTypes.INPUT.value,
                component=component.value,
            )
