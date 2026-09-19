from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

RISK_LABELS = {"risk", "collision", "near_miss", "near-miss"}
NORMAL_LABELS = {"normal", "safe", "no_event"}

REQUIRED_COLUMNS = (
    "clip_id",
    "path",
    "label",
)

OPTIONAL_COLUMNS = (
    "trip_id",
    "event_type",
    "weather",
    "time_of_day",
    "scene",
    "duration_s",
    "fps",
    "event_time_s",
    "alert_time_s",
    "license",
)


@dataclass(frozen=True)
class ClipRecord:
    clip_id: str
    path: str
    label: str
    trip_id: Optional[str] = None
    event_type: Optional[str] = None
    weather: Optional[str] = None
    time_of_day: Optional[str] = None
    scene: Optional[str] = None
    duration_s: Optional[float] = None
    fps: Optional[float] = None
    event_time_s: Optional[float] = None
    alert_time_s: Optional[float] = None
    license: Optional[str] = None

    @property
    def is_risk(self) -> bool:
        return str(self.label).lower() in RISK_LABELS

    def cutoff_s(self) -> Optional[float]:
        """Last timestamp allowed in the model input."""
        if self.alert_time_s is not None:
            return float(self.alert_time_s)
        if self.is_risk and self.event_time_s is not None:
            return float(self.event_time_s)
        return self.duration_s
