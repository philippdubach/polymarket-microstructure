from __future__ import annotations

import pytest


def test_lob_best_bid_ask_matches_live_api_tolerance():
    """For a currently-active market, the last replayed best bid/ask should be
    within a tolerance of Polymarket's public CLOB API snapshot."""
    pytest.skip(
        "Manual validation: run scripts/snapshot_validation_report.py once and commit result"
    )
