from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal

from polydata.lob import LOBReplay
from polydata.stream import StreamRecord

DEPTH_KS: tuple[int, ...] = (1, 2, 3, 5, 10)


@dataclass(frozen=True)
class LOBSample:
    t: datetime
    best_bid: Decimal | None
    best_ask: Decimal | None
    depth_by_k: dict[int, tuple[Decimal, Decimal]] = field(default_factory=dict)

    @property
    def top1_bid_depth(self) -> Decimal:
        return self.depth_by_k.get(1, (Decimal("0"), Decimal("0")))[0]

    @property
    def top1_ask_depth(self) -> Decimal:
        return self.depth_by_k.get(1, (Decimal("0"), Decimal("0")))[1]

    @property
    def top5_bid_depth(self) -> Decimal:
        return self.depth_by_k.get(5, (Decimal("0"), Decimal("0")))[0]

    @property
    def top5_ask_depth(self) -> Decimal:
        return self.depth_by_k.get(5, (Decimal("0"), Decimal("0")))[1]


def resample_lob(
    records: Iterable[StreamRecord],
    start: datetime,
    end: datetime,
    step: timedelta,
) -> list[LOBSample]:
    lob = LOBReplay()
    it = iter(records)
    current: StreamRecord | None = next(it, None)
    samples: list[LOBSample] = []
    t = start
    while t < end:
        while current is not None and current.ts_received <= t:
            lob.apply(current.event)
            current = next(it, None)
        depth_by_k = {k: (lob.top_k_bid_depth(k), lob.top_k_ask_depth(k)) for k in DEPTH_KS}
        samples.append(LOBSample(
            t=t, best_bid=lob.best_bid(), best_ask=lob.best_ask(),
            depth_by_k=depth_by_k,
        ))
        t += step
    return samples
