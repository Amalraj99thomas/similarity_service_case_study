"""Text normalisation and OpenAI embedding generation.

Handles template variable expansion, content normalisation, and batched
calls to the OpenAI embeddings API.  All vectors are L2-normalised at
embed time so that dot products equal cosine similarities.
"""

import re

import numpy as np
from openai import OpenAI

from prompt_similarity.config import VARIABLE_DESCRIPTIONS, MODEL_NAME, EMBED_BATCH_SIZE


def expand_variable(var: str) -> str:
    """Expand a single template variable name to a natural-language phrase.

    Looks up the variable in VARIABLE_DESCRIPTIONS first; falls back to
    converting snake_case to 'the <words>'.

    Examples:
        >>> expand_variable("patient_name")
        'the name of the patient'
        >>> expand_variable("appointment_date")
        'the appointment date'
    """
    key = var.strip().lower()
    if key in VARIABLE_DESCRIPTIONS:
        return VARIABLE_DESCRIPTIONS[key]
    return "the " + key.replace("_", " ")


def normalize(text: str) -> str:
    """Expand all ``{{variable}}`` placeholders into semantic phrases.

    This ensures that two prompts with different variable names but identical
    intent produce similar embeddings.

    Example:
        >>> normalize("Greet {{patient_name}} from {{org_name}}")
        'Greet the name of the patient from the name of the organization'
    """
    return re.sub(
        r"\{\{([^}]+)\}\}",
        lambda m: expand_variable(m.group(1)),
        text,
    ).strip()


def extract_vars(text: str) -> list[str]:
    """Return a sorted, deduplicated list of ``{{variable}}`` names in *text*."""
    return sorted(set(re.findall(r"\{\{([^}]+)\}\}", text)))


def embed(texts: list[str], client: OpenAI, batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
    """Encode *texts* into L2-normalised float32 vectors via the OpenAI API.

    Processes in chunks of *batch_size* to stay within the API's 2048-input
    limit.  Returns an (n, DIM) array where every row has unit L2 norm.
    """
    all_vecs: list[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        response = client.embeddings.create(input=chunk, model=MODEL_NAME)
        vecs = np.array([item.embedding for item in response.data], dtype="float32")
        all_vecs.append(vecs)

    vecs = np.concatenate(all_vecs, axis=0)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1, norms)
