"""Tests for prompt_similarity.embeddings — text normalisation and variable handling."""

import pytest

from prompt_similarity.embeddings import expand_variable, normalize, extract_vars


class TestExpandVariable:
    """Tests for single-variable expansion."""

    def test_known_variable(self):
        assert expand_variable("patient_name") == "the name of the patient"

    def test_known_variable_case_insensitive(self):
        assert expand_variable("Patient_Name") == "the name of the patient"

    def test_known_variable_with_whitespace(self):
        assert expand_variable("  agent_name  ") == "the name of the agent"

    def test_unknown_variable_snake_case_fallback(self):
        assert expand_variable("appointment_date") == "the appointment date"

    def test_unknown_variable_single_word(self):
        assert expand_variable("email") == "the email"


class TestNormalize:
    """Tests for content normalisation (variable placeholder expansion)."""

    def test_single_variable(self):
        result = normalize("Hello {{patient_name}}")
        assert result == "Hello the name of the patient"

    def test_multiple_variables(self):
        result = normalize("Greet {{patient_name}} from {{org_name}}")
        assert "the name of the patient" in result
        assert "the name of the organization" in result

    def test_no_variables(self):
        text = "Show empathy and understanding in every response."
        assert normalize(text) == text

    def test_unknown_variable(self):
        result = normalize("Book on {{appointment_date}}")
        assert result == "Book on the appointment date"

    def test_whitespace_stripping(self):
        result = normalize("  Hello {{agent_name}}  ")
        assert result == "Hello the name of the agent"

    def test_adjacent_variables(self):
        result = normalize("{{agent_name}} {{org_name}}")
        assert result == "the name of the agent the name of the organization"


class TestExtractVars:
    """Tests for template variable extraction."""

    def test_no_variables(self):
        assert extract_vars("No variables here.") == []

    def test_single_variable(self):
        assert extract_vars("Hello {{patient_name}}") == ["patient_name"]

    def test_multiple_variables_sorted_and_deduped(self):
        text = "{{org_name}} greets {{patient_name}} from {{org_name}}"
        result = extract_vars(text)
        assert result == ["org_name", "patient_name"]

    def test_complex_template(self):
        text = "Schedule {{appointment_type}} with {{provider_name}} at {{location}}"
        result = extract_vars(text)
        assert result == ["appointment_type", "location", "provider_name"]
