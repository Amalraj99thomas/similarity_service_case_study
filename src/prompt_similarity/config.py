"""Application configuration and constants."""

# ── Database ───────────────────────────────────────────────────────────────────
DB_PATH = "prompts_database.db"

# ── Embedding model ────────────────────────────────────────────────────────────
MODEL_NAME = "text-embedding-3-small"
DIM = 1536
EMBED_BATCH_SIZE = 512

# ── Variable semantic expansion ────────────────────────────────────────────────
# Maps template variable names to natural-language descriptions used during
# normalisation.  When a variable isn't in this dict the fallback heuristic
# converts snake_case to "the <words>" (e.g. appointment_date → the appointment date).
VARIABLE_DESCRIPTIONS: dict[str, str] = {
    "agent_name":    "the name of the agent",
    "org_name":      "the name of the organization",
    "organization":  "the organization",
    "question_text": "the question being asked",
    "options":       "the list of valid answer options",
    "field_name":    "the name of the form field being collected",
    "next_step":     "the next step in the process",
    "caller_name":   "the name of the caller",
    "patient_name":  "the name of the patient",
    "date":          "the relevant date",
    "time":          "the relevant time",
}
