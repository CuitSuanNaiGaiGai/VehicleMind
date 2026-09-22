# Versioned Context Contract Design

## Scope

This increment covers the first, fourth, and fifth tasks in section 3 of `todolist.md`: a stable Driver/Road/Vehicle field contract, type and range validation, and atomic domain updates. Observation envelopes, freshness, debouncing, queues, and runtime metrics follow in later increments.

## Contract

`modules/vehicle_ai/context/contract.py` owns `CONTEXT_SCHEMA_VERSION = 1` and the field rules for all three domains. The rules cover enum membership, booleans, nullable fields, finite numeric ranges, nonnegative counts, and strings. Metadata fields (`updated_at`, `source`) are managed by the runtime and cannot be changed through partial updates. `VehicleContext.to_dict()` includes `schema_version`; the Agent's compact semantic context keeps its existing shape. Replay imports the same schema version constant and keeps the current semantic digest for unchanged scenario behavior.

## Update semantics

`ContextManager` validates every incoming field while holding its lock. An invalid field raises `ValueError`, `TypeError`, or `AttributeError` before changing the context, timestamp, or change history. A valid multi-field update applies to a copy of one domain, then replaces that domain and publishes field changes together. Identical values still refresh the timestamp, matching current behavior.

## Verification

Tests cover contract completeness against dataclass fields, serialized schema version, valid boundary values, invalid enum/type/range values, and rollback of data, timestamp, and change history when a later field fails. Existing replay and tool tests must preserve their outputs and semantic trace digest.
