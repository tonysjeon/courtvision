"""Side-by-side player tracking visualization."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from courtvision.players.filtering import PlayerCandidate, PlayerId
from courtvision.tracking.events import BallTrackingEvent, PlayerTrackingEvent
from courtvision.visualization.court import court_position_to_canvas, render_normalized_court

PLAYER_COLORS: dict[PlayerId, tuple[int, int, int]] = {
    "near_player": (0, 215, 255),
    "far_player": (55, 75, 255),
}
PLAYER_LABELS: dict[PlayerId, str] = {
    "near_player": "Player 1",
    "far_player": "Player 2",
}
OUTLINE_COLOR = (25, 25, 25)
LABEL_TEXT_COLOR = (255, 255, 255)
BALL_COLOR = (35, 255, 190)
BALL_SEAM_COLOR = (225, 255, 245)


def _draw_rounded_rectangle(
    image: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    left, top = top_left
    right, bottom = bottom_right
    radius = max(1, (bottom - top) // 2)
    cv2.rectangle(image, (left + radius, top), (right - radius, bottom), color, -1)
    cv2.rectangle(image, (left, top + radius), (right, bottom - radius), color, -1)
    cv2.circle(image, (left + radius, top + radius), radius, color, -1, cv2.LINE_AA)
    cv2.circle(image, (right - radius, top + radius), radius, color, -1, cv2.LINE_AA)


def _draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    font_scale: float,
    pill_height: int | None = None,
) -> None:
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )
    x, y = origin
    x = max(3, min(x, image.shape[1] - text_width - 15))
    if pill_height is None:
        text_y = max(text_height + 6, min(y, image.shape[0] - baseline - 4))
        top = text_y - text_height - 5
        bottom = text_y + baseline + 3
    else:
        top = max(0, min(y - pill_height // 2, image.shape[0] - pill_height))
        bottom = top + pill_height
        text_y = top + (pill_height + text_height - baseline) // 2
    top_left = (x, top)
    bottom_right = (x + text_width + 12, bottom)
    overlay = image.copy()
    _draw_rounded_rectangle(
        overlay,
        top_left,
        bottom_right,
        OUTLINE_COLOR,
    )
    cv2.addWeighted(overlay, 0.78, image, 0.22, 0, image)
    cv2.putText(
        image,
        text,
        (x + 6, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        LABEL_TEXT_COLOR,
        thickness,
        cv2.LINE_AA,
    )


def _draw_tennis_ball(
    image: np.ndarray,
    center: tuple[int, int],
    radius: int,
) -> None:
    """Draw one consistent tennis-ball marker for every tracked position."""
    cv2.circle(image, center, radius + 1, OUTLINE_COLOR, -1, cv2.LINE_AA)
    cv2.circle(image, center, radius, BALL_COLOR, -1, cv2.LINE_AA)
    seam_radius = max(2, radius)
    seam_offset = max(1, radius // 2)
    seam_thickness = max(1, radius // 5)
    cv2.ellipse(
        image,
        (center[0] - seam_offset, center[1]),
        (seam_radius, seam_radius),
        0,
        -55,
        55,
        BALL_SEAM_COLOR,
        seam_thickness,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        image,
        (center[0] + seam_offset, center[1]),
        (seam_radius, seam_radius),
        180,
        -55,
        55,
        BALL_SEAM_COLOR,
        seam_thickness,
        cv2.LINE_AA,
    )


def render_tracking_preview(
    frame: np.ndarray,
    candidates: dict[PlayerId, PlayerCandidate],
    events: list[PlayerTrackingEvent],
    *,
    timestamp_ms: float,
    ball_event: BallTrackingEvent | None = None,
    ball_trail: Sequence[BallTrackingEvent] = (),
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
            OUTLINE_COLOR,
            7,
        )
        cv2.rectangle(
            annotated,
            (round(bbox.x1), round(bbox.y1)),
            (round(bbox.x2), round(bbox.y2)),
            color,
            3,
        )
        event = event_by_player[player_id]
        label = f"{PLAYER_LABELS[player_id]}  {event.confidence:.2f}"
        _draw_label(
            annotated,
            label,
            (round(bbox.x1), max(24, round(bbox.y1) - 8)),
            font_scale=0.58,
        )
        cv2.circle(
            annotated,
            (round(candidate.pixel_x), round(candidate.pixel_y)),
            8,
            OUTLINE_COLOR,
            -1,
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

    if ball_event is not None:
        ball_x = round(ball_event.pixel_position.x)
        ball_y = round(ball_event.pixel_position.y)
        _draw_tennis_ball(annotated, (ball_x, ball_y), 4)
        _draw_label(
            annotated,
            "Ball",
            (ball_x + 14, ball_y + 5),
            font_scale=0.48,
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
    for trail_event in ball_trail:
        age_ms = timestamp_ms - trail_event.timestamp_ms
        if not 0 <= age_ms <= 450:
            continue
        trail_x, trail_y = court_position_to_canvas(
            trail_event.court_position.x,
            trail_event.court_position.y,
            court.shape[1],
            court.shape[0],
        )
        strength = 1 - age_ms / 450
        radius = max(1, round(3 * strength))
        overlay = court.copy()
        cv2.circle(overlay, (trail_x, trail_y), radius, BALL_COLOR, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.2 + strength * 0.35, court, 0.8 - strength * 0.35, 0, court)

    for event in events:
        color = PLAYER_COLORS[event.object_id]
        x, y = court_position_to_canvas(
            event.court_position.x,
            event.court_position.y,
            court.shape[1],
            court.shape[0],
        )
        cv2.circle(court, (x, y), 14, OUTLINE_COLOR, -1, cv2.LINE_AA)
        cv2.circle(court, (x, y), 12, color, -1, cv2.LINE_AA)
        _draw_label(
            court,
            PLAYER_LABELS[event.object_id],
            (x + 22, y),
            font_scale=0.48,
            pill_height=28,
        )

    if ball_event is not None:
        ball_x, ball_y = court_position_to_canvas(
            ball_event.court_position.x,
            ball_event.court_position.y,
            court.shape[1],
            court.shape[0],
        )
        _draw_tennis_ball(court, (ball_x, ball_y), 7)
        _draw_label(court, "Ball", (ball_x + 20, ball_y), font_scale=0.48, pill_height=24)

    return np.hstack((annotated, court))
