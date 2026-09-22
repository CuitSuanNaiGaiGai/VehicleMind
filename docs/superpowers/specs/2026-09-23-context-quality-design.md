# Runtime Context Quality Design

The default values in `DriverContext` and `RoadContext` are not evidence of a successful perception result. Runtime quality therefore lives alongside semantic values rather than overloading `False`, `None`, or `UNKNOWN`.

For each domain, `ContextManager` tracks whether any update was received, whether the latest observation was valid, and when it was received on a monotonic clock. The status order is `MISSING` before any update, `INVALID` after an explicitly invalid observation, `STALE` when the last valid update exceeds the domain TTL, `UNKNOWN` for a fresh `None`/`UNKNOWN` field, and `KNOWN` otherwise. A fresh `False` is `KNOWN`. Driver and vehicle TTL are 2 seconds; road TTL is 1 second, matching the existing context model's `is_fresh` defaults. Source-frame timestamps are retained as evidence but are not used to compare freshness across domains.

Normal context updates mark the domain as observed only after field validation and atomic replacement succeed. Invalid perception metadata does not mutate semantic fields or publish events; it marks the domain invalid. The new report API returns domain and field quality separately. Agent presentation and replay rendering will consume it in a later increment. Tests use injected monotonic times for deterministic boundary cases.
