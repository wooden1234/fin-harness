from dataclasses import replace
from types import SimpleNamespace

import pytest

from harness.memory_specs import list_memory_agent_specs
from app.services.memory.memory_catalog import (
    MEMORY_KEYS,
    MEMORY_TYPES,
    validate_agent_memory_keys,
    validate_memory_catalog,
    validate_memory_configuration,
)


def test_default_catalog_and_agent_whitelists_are_valid():
    validate_memory_configuration(list_memory_agent_specs())


def test_unknown_agent_memory_key_fails_fast():
    invalid_agent = SimpleNamespace(
        agent_id="invalid_agent",
        memory_keys=("missing_memory_key",),
    )

    with pytest.raises(ValueError, match="unknown_agent_memory_key"):
        validate_agent_memory_keys([invalid_agent])


def test_forbidden_key_cannot_enable_cache_or_vector():
    invalid_keys = dict(MEMORY_KEYS)
    invalid_keys["credential_secret"] = replace(
        MEMORY_KEYS["credential_secret"],
        cacheable=True,
        vectorizable=True,
    )

    with pytest.raises(ValueError, match="forbidden_memory_key_storage_enabled"):
        validate_memory_catalog(memory_keys=invalid_keys)


def test_turn_memory_type_cannot_write_sql():
    invalid_types = dict(MEMORY_TYPES)
    invalid_types["turn_context"] = replace(
        MEMORY_TYPES["turn_context"],
        sql_writable=True,
    )

    with pytest.raises(ValueError, match="turn_memory_cannot_write_sql"):
        validate_memory_catalog(memory_types=invalid_types)


def test_structured_enum_cannot_enable_vectorization():
    invalid_keys = dict(MEMORY_KEYS)
    invalid_keys["response_language"] = replace(
        MEMORY_KEYS["response_language"],
        vectorizable=True,
    )

    with pytest.raises(ValueError, match="structured_enum_cannot_vectorize"):
        validate_memory_catalog(memory_keys=invalid_keys)
