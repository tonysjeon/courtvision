"""Select the two active players from all person detections."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Literal

from courtvision.geometry.mapper import CourtMapper
from courtvision.players.detection import PersonDetection

PlayerId = Literal["near_player", "far_player"]


@dataclass(frozen=True, slots=True)
class PlayerCandidate:
    player_id: PlayerId
    detection: PersonDetection
    pixel_x: float
    pixel_y: float
    court_x: float
    court_y: float


class ActivePlayerSelector:
    """Use court location and prior position to reject non-player people."""

    def __init__(self, *, court_margin: float = 30.0) -> None:
        self.court_margin = court_margin

    def select(
        self,
        detections: list[PersonDetection],
        mapper: CourtMapper,
        previous_positions: dict[PlayerId, tuple[float, float]] | None = None,
    ) -> dict[PlayerId, PlayerCandidate]:
        previous_positions = previous_positions or {}
        mapped = []
        for detection in detections:
            pixel_x, pixel_y = detection.bbox.ground_position
            court_x, court_y = mapper.transform(pixel_x, pixel_y)
            if self._is_near_court(court_x, court_y):
                mapped.append((detection, pixel_x, pixel_y, court_x, court_y))

        selected: dict[PlayerId, PlayerCandidate] = {}
        used: set[int] = set()
        for player_id in ("near_player", "far_player"):
            ranked = sorted(
                enumerate(mapped),
                key=lambda item: self._score(player_id, item[1], previous_positions.get(player_id)),
            )
            for index, values in ranked:
                if index in used or not self._belongs_to_half(player_id, values[4]):
                    continue
                detection, pixel_x, pixel_y, court_x, court_y = values
                selected[player_id] = PlayerCandidate(
                    player_id=player_id,
                    detection=detection,
                    pixel_x=pixel_x,
                    pixel_y=pixel_y,
                    court_x=court_x,
                    court_y=court_y,
                )
                used.add(index)
                break
        return selected

    def _is_near_court(self, x: float, y: float) -> bool:
        margin = self.court_margin
        return -margin <= x <= 100 + margin and -margin <= y <= 100 + margin

    @staticmethod
    def _belongs_to_half(player_id: PlayerId, court_y: float) -> bool:
        return court_y <= 60 if player_id == "near_player" else court_y >= 40

    @staticmethod
    def _score(
        player_id: PlayerId,
        values: tuple[PersonDetection, float, float, float, float],
        previous: tuple[float, float] | None,
    ) -> float:
        detection, _, _, court_x, court_y = values
        expected_y = 0.0 if player_id == "near_player" else 100.0
        position_score = abs(court_y - expected_y)
        if previous is not None:
            position_score = hypot(court_x - previous[0], court_y - previous[1]) * 2
        return position_score - detection.confidence * 20
