"""Small robot-geometry helpers shared by the two-arm interaction controller."""
from __future__ import annotations

import numpy as np

FINGER_OPEN = 0.035
FINGER_CENTER_Y = 0.0085


def _polyline_at(p0, points, fraction):
    """Interpolate along a polyline from p0 through points."""
    polyline = np.vstack([p0, points])
    lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    total = float(lengths.sum())
    if total < 1e-12:
        return polyline[-1].copy()
    remaining = fraction * total
    for index, length in enumerate(lengths):
        if remaining <= length or index == len(lengths) - 1:
            return polyline[index] + (polyline[index + 1] - polyline[index]) * (
                remaining / max(length, 1e-12)
            )
        remaining -= length
    return polyline[-1].copy()


def _smooth(fraction):
    fraction = min(max(fraction, 0.0), 1.0)
    return fraction * fraction * (3.0 - 2.0 * fraction)


def _slerp_R(start, end, fraction):
    """Interpolate two rotation matrices through their relative axis-angle rotation."""
    relative = start.T @ end
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cosine)
    if angle < 1e-8:
        return end.copy()
    axis = np.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ]
    ) / (2.0 * np.sin(angle))
    angle *= fraction
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return start @ (np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * skew @ skew)


def grip_opening(points, hand_transform, opening, pad_radius, squeeze):
    """Estimate a finger opening that closes around Gaussian points at the hand tip."""
    from robot_arm import TCP_OFFSET

    local = (points - hand_transform[:3, 3]) @ hand_transform[:3, :3]
    inside = (
        (np.abs(local[:, 0]) < 0.012)
        & (np.abs(local[:, 1]) < opening + FINGER_CENTER_Y)
        & (local[:, 2] > TCP_OFFSET - 0.016)
        & (local[:, 2] < TCP_OFFSET - 0.002)
    )
    if inside.sum() < 10:
        return 0.0, int(inside.sum())
    half_width = float(np.percentile(np.abs(local[inside, 1]), 75))
    finger = np.clip(half_width - squeeze - FINGER_CENTER_Y + pad_radius, 0.0, opening)
    return float(finger), int(inside.sum())


def _hand_frame(tcp, rotation):
    """Return a world transform whose finger-tip center is tcp."""
    from robot_arm import TCP_OFFSET

    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(tcp, float) - TCP_OFFSET * rotation[:, 2]
    return transform