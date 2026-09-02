"""Side-by-side player tracking visualization."""

from __future__ import annotations

import cv2
import numpy as np

from courtvision.players.filtering import PlayerCandidate, PlayerId
from courtvision.tracking.events import PlayerTrackingEvent
from courtvision.visualization.court import court_position_to_canvas, render_normalized_court

PLAYER_COLORS: dict[PlayerId, tuple[int, int, int]] = {
    "near_player": (0, 200, 255),
    "far_player": (255, 100, 180),
}


def render_tracking_preview(
    frame: np.ndarray,
    candidates: dict[PlayerId, PlayerCandidate],
    events: list[PlayerTrackingEvent],
    *,
    timestamp_ms: float,
) -> np.ndarray:
    annotated = frame.copy()
    event_by_player = {event.object_id: event for event in events}

    for player_id, candidate in candidates.items():
        color = PLAYER_COLORS[player_id]
        bbox = candidate.detection.bbox
        cv2.rectangle(
            annotated,
            (round(bbox.x1), round(bbox.y1)),
            (round(bbox.x2), round(bbox.y2)),
            color,
            3,
        )
        event = event_by_player[player_id]
        label = f"{player_id} #{event.track_id} {event.confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (round(bbox.x1), max(24, round(bbox.y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.circle(
            annotated,
            (round(candidate.pixel_x), round(candidate.pixel_y)),
            6,
            color,
            -1,
            cv2.LINE_AA,
        )

    cv2.putText(
        annotated,
        f"{timestamp_ms / 1000:.2f}s",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    court = render_normalized_court(width=max(300, frame.shape[1] // 2), height=frame.shape[0])
    for event in events:
        color = PLAYER_COLORS[event.object_id]
        x, y = court_position_to_canvas(
            event.court_position.x,
            event.court_position.y,
            court.shape[1],
            court.shape[0],
        )
        cv2.circle(court, (x, y), 12, color, -1, cv2.LINE_AA)
        cv2.putText(
            court,
            f"{event.object_id} #{event.track_id}",
            (x + 15, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return np.hstack((annotated, court))
