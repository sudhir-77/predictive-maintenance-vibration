"""CWRU 12 kHz drive-end recordings used for the 10-class fault task."""

from __future__ import annotations

from dataclasses import dataclass

from vibration.config import CLASS_ORDER, RPM_BY_LOAD

# file ids are the Bearing Data Center names, ordered by motor load 0, 1, 2, 3 hp.
# Outer-race files are the load-zone-centered position (6:00), one geometry per class.
_FILE_IDS: dict[str, tuple[int, ...]] = {
    "normal": (97, 98, 99, 100),
    "ir_007": (105, 106, 107, 108),
    "ir_014": (169, 170, 171, 172),
    "ir_021": (209, 210, 211, 212),
    "or_007": (130, 131, 132, 133),
    "or_014": (197, 198, 199, 200),
    "or_021": (234, 235, 236, 237),
    "ball_007": (118, 119, 120, 121),
    "ball_014": (185, 186, 187, 188),
    "ball_021": (222, 223, 224, 225),
}

_DIAMETER_IN: dict[str, float | None] = {
    "normal": None,
    "ir_007": 0.007,
    "ir_014": 0.014,
    "ir_021": 0.021,
    "or_007": 0.007,
    "or_014": 0.014,
    "or_021": 0.021,
    "ball_007": 0.007,
    "ball_014": 0.014,
    "ball_021": 0.021,
}


@dataclass(frozen=True)
class Recording:
    file_id: str
    label: str
    load_hp: int
    nominal_rpm: int
    diameter_in: float | None

    @property
    def family(self) -> str:
        if self.label == "normal":
            return "normal"
        if self.label.startswith("ir_"):
            return "inner"
        if self.label.startswith("or_"):
            return "outer"
        return "ball"


def _build() -> tuple[Recording, ...]:
    rows: list[Recording] = []
    for label in CLASS_ORDER:
        ids = _FILE_IDS[label]
        if len(ids) != 4:
            raise ValueError(f"{label} needs four load files")
        for load_hp, file_id in enumerate(ids):
            rows.append(
                Recording(
                    file_id=str(file_id),
                    label=label,
                    load_hp=load_hp,
                    nominal_rpm=RPM_BY_LOAD[load_hp],
                    diameter_in=_DIAMETER_IN[label],
                )
            )
    return tuple(rows)


RECORDINGS: tuple[Recording, ...] = _build()


def recording_by_id(file_id: str) -> Recording:
    for item in RECORDINGS:
        if item.file_id == file_id:
            return item
    raise KeyError(file_id)
