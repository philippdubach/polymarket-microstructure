import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow.parquet as pq

from polydata.paths import parquet_files
from polydata.schema import validate_schema

_FNAME_RE = re.compile(r"polymarket_orderbook_(\d{4}-\d{2}-\d{2})T(\d{2})\.parquet")


@dataclass
class InventoryReport:
    n_files: int
    total_rows: int
    total_bytes: int
    first_hour: datetime
    last_hour: datetime
    gaps: list[datetime]
    schema_errors: list[tuple[Path, str]]


def _parse_hour(path: Path) -> datetime:
    m = _FNAME_RE.match(path.name)
    if not m:
        raise ValueError(f"unrecognised filename: {path.name}")
    d, h = m.group(1), int(m.group(2))
    return datetime.fromisoformat(d).replace(hour=h, tzinfo=UTC)


def scan_inventory(limit: int | None = None) -> InventoryReport:
    files = parquet_files()
    if limit is not None:
        files = files[:limit]
    hours = [_parse_hour(f) for f in files]
    total_rows = 0
    total_bytes = 0
    schema_errors: list[tuple[Path, str]] = []
    for f in files:
        try:
            md = pq.read_metadata(f)
            total_rows += md.num_rows
            total_bytes += f.stat().st_size
            validate_schema(md.schema.to_arrow_schema())
        except Exception as exc:  # noqa: BLE001
            schema_errors.append((f, str(exc)))
    gaps: list[datetime] = []
    if len(hours) >= 2:
        hours_sorted = sorted(hours)
        expected = hours_sorted[0]
        step = timedelta(hours=1)
        present = set(hours_sorted)
        while expected <= hours_sorted[-1]:
            if expected not in present:
                gaps.append(expected)
            expected += step
    return InventoryReport(
        n_files=len(files),
        total_rows=total_rows,
        total_bytes=total_bytes,
        first_hour=min(hours),
        last_hour=max(hours),
        gaps=gaps,
        schema_errors=schema_errors,
    )
