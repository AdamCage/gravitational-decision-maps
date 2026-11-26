"""Matched inertial dynamics for the Gaussian prototype field."""

import numpy as np


def inertial_velocity(momentum, mode, speed_limit=None):
    """Convert unit-mass momentum to velocity along the final array axis."""
    if mode == "inertial":
        return momentum
    if mode not in {"relativistic", "clipped_inertial"}:
        raise ValueError(f"Unknown inertial mode: {mode}")
    if speed_limit is None or not np.isfinite(speed_limit) or speed_limit <= 0:
        raise ValueError("Capped inertial modes require a positive finite speed limit")

    magnitude = np.hypot.reduce(momentum, axis=-1, keepdims=True)
    if mode == "relativistic":
        divisor = np.hypot(speed_limit, magnitude)
    else:
        divisor = np.maximum(speed_limit, magnitude)
    return momentum * (speed_limit / divisor)


def rk4_inertial(model, x, time, max_step, *, mode, damping, speed_limit=None):
    """Return endpoints and field evaluations per row, starting at rest.

    All modes solve p' = F(x) - damping*p. They differ in the velocity
    map used for x'. The hard cap acts on velocity while momentum persists.
    """
    if not all(np.isfinite(value) and value > 0 for value in (time, max_step)):
        raise ValueError("Positive finite horizon and step required")
    if not np.isfinite(damping) or damping < 0:
        raise ValueError("Damping must be nonnegative and finite")

    steps = int(np.ceil(time / max_step))
    step = time / steps
    position = x - model.origin
    momentum = np.zeros_like(position)

    def derivatives(at, p):
        return inertial_velocity(p, mode, speed_limit), model._rel_field(at) - damping * p

    for _ in range(steps):
        v1, f1 = derivatives(position, momentum)
        v2, f2 = derivatives(position + (step / 2) * v1, momentum + (step / 2) * f1)
        v3, f3 = derivatives(position + (step / 2) * v2, momentum + (step / 2) * f2)
        v4, f4 = derivatives(position + step * v3, momentum + step * f3)
        position += (step / 6) * (v1 + 2 * v2 + 2 * v3 + v4)
        momentum += (step / 6) * (f1 + 2 * f2 + 2 * f3 + f4)
    return position + model.origin, 4 * steps
