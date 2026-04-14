"""Shared application state initialised during FastAPI lifespan.

Stores the OpenAI client instance so that route modules can access it
without circular imports or passing it through every function call.
"""

from openai import OpenAI

_client: OpenAI | None = None


def init_client() -> None:
    """Create the OpenAI client.  Called once during app startup."""
    global _client
    _client = OpenAI()


def get_openai_client() -> OpenAI:
    """Return the initialised OpenAI client.

    Raises:
        RuntimeError: If called before ``init_client()``.
    """
    if _client is None:
        raise RuntimeError("OpenAI client not initialised. Was the app lifespan started?")
    return _client
