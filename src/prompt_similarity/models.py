"""Pydantic request and response schemas for the API."""

from typing import Optional

from pydantic import BaseModel


class PromptInput(BaseModel):
    """Schema for a single prompt submitted to the embedding generation endpoint."""
    prompt_id: str
    category:  str
    layer:     str
    name:      Optional[str] = None
    content:   str


class SemanticSearchRequest(BaseModel):
    """Schema for semantic search requests."""
    query:     str
    limit:     int   = 10
    threshold: float = 0.0
