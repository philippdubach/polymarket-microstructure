import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from polydata.paths import parquet_files
from polydata.schema import validate_schema


def test_schema_matches_first_file():
    f = parquet_files()[0]
    actual = pq.read_schema(f)
    validate_schema(actual)  # raises on mismatch


def test_schema_rejects_empty():
    bad = pa.schema([("foo", pa.int64())])
    with pytest.raises(AssertionError):
        validate_schema(bad)
