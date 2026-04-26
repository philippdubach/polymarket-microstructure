from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from polydata.resample import LOBSample, resample_lob
from polydata.stream import StreamRecord


@dataclass
class MeasurementWindow:
    market_id: str
    t_start: datetime
    t_end: datetime
    events: Sequence[StreamRecord]
    sample_step: timedelta = field(default_factory=lambda: timedelta(seconds=1))
    samples: list[LOBSample] = field(init=False)

    def __post_init__(self) -> None:
        if self.t_end <= self.t_start:
            raise ValueError(f"t_end {self.t_end} must be after t_start {self.t_start}")
        self.samples = resample_lob(self.events, self.t_start, self.t_end, self.sample_step)

    def n_samples(self) -> int:
        return len(self.samples)

    def duration(self) -> timedelta:
        return self.t_end - self.t_start
