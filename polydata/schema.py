import pyarrow as pa

EXPECTED_SCHEMA = pa.schema([
    pa.field("timestamp_received", pa.timestamp("ms", tz="UTC"), nullable=False),
    pa.field("timestamp_created_at", pa.timestamp("ms", tz="UTC"), nullable=False),
    pa.field("market_id", pa.string(), nullable=False),
    pa.field("update_type", pa.string(), nullable=False),
    pa.field("data", pa.string(), nullable=False),
])


def validate_schema(schema: pa.Schema) -> None:
    names = set(schema.names)
    expected = set(EXPECTED_SCHEMA.names)
    assert names == expected, f"column mismatch: {names} vs {expected}"
    for field in EXPECTED_SCHEMA:
        got = schema.field(field.name)
        assert got.type == field.type, f"{field.name}: {got.type} != {field.type}"
        assert got.nullable == field.nullable, f"{field.name} nullability mismatch"
