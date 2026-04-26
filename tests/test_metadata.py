from polydata.metadata import MarketMetadata, parse_gamma_response

SAMPLE = {
    "conditionId": "0xabc",
    "question": "Will X happen by Y?",
    "category": "Politics",
    "endDate": "2026-11-05T00:00:00Z",
    "closedTime": "2026-11-05T01:23:00Z",
    "outcomePrices": ["0.6", "0.4"],
    "active": False,
    "closed": True,
}


def test_parse_gamma_response():
    md = parse_gamma_response(SAMPLE)
    assert isinstance(md, MarketMetadata)
    assert md.market_id == "0xabc"
    assert md.category == "Politics"
    assert md.question.startswith("Will")
    assert md.resolved_outcome == 0  # YES prevailed (0.6 > 0.4)
