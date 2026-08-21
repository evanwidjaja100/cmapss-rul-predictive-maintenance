"""Marker taxonomy enforcement (Phase 8).

Every test must have exactly one primary tier:
  unit | static_contract | tracked_artifacts | integration | app_smoke
needs_artifacts is supplemental.

Enforced via collection hook (fail fast) and mirrored by test_marker_audit.
"""
from __future__ import annotations

PRIMARY = {"unit", "static_contract", "tracked_artifacts", "integration", "app_smoke"}
SUPPLEMENTAL = {"needs_artifacts"}


def pytest_collection_modifyitems(session, config, items):  # noqa: ARG001
    errors: list[str] = []
    for item in items:
        declared = {m.name for m in item.iter_markers()}
        primary = declared & PRIMARY
        # count exactly one primary
        if len(primary) != 1:
            errors.append(f"{item.nodeid}: primary={sorted(primary)} (need exactly 1 of {sorted(PRIMARY)}) declared={sorted(declared)}")
        # unknown markers only primary/supplemental? allow other markers like parametrize, skipif etc. but ensure no unknown tier
        # supplemental is allowed 0 or 1; no check needed beyond primary.
        # Also check that needs_artifacts is not used as primary confusion – it's supplemental, ok.
        # Ensure no test uses needs_artifacts without also having primary (already covered)
    if errors:
        # fail collection: make visible
        from _pytest.outcomes import Failed
        msg = "Marker taxonomy violation: every test must have exactly one primary tier (unit|static_contract|tracked_artifacts|integration|app_smoke), needs_artifacts is supplemental.\n" + "\n".join(errors[:80])
        # Use pytest.fail to abort collection with clear message
        # We raise Failed so pytest reports it
        raise Failed(msg, pytrace=False)
