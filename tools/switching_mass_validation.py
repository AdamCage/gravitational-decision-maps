"""Deterministic checks of the short-time switch law and finite-time bounds.

Run from the repository root with ``python tools/switching_mass_validation.py``.
The three-class reference uses numerical endpoint integration and quadrature.
The one-dimensional fixture has a closed-form endpoint law.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import roots_legendre


SITES = np.array([[0.65, 0.25], [-0.35, 0.55], [0.25, -0.65]])
WEIGHTS = np.array([1.0, 0.8, 1.2])
VARIANCE = 0.30
INTERCEPTS = VARIANCE * np.log(WEIGHTS) - np.sum(SITES**2, axis=1) / 2
HORIZONS = (0.08, 0.04, 0.02, 0.01, 0.005)
MAX_SPEED = np.sqrt(2) + np.max(np.linalg.norm(SITES, axis=1))
DIAMETER = np.max(np.linalg.norm(SITES[:, None] - SITES[None, :], axis=-1))
LIPSCHITZ = max(1.0, DIAMETER**2 / (4 * VARIANCE) - 1)
NORMAL_NODES, NORMAL_WEIGHTS = roots_legendre(3)


def density(points):
    return (9 / 16) * np.prod(np.maximum(0.0, 1.0 - points**2), axis=-1)


def scores(points):
    return points @ SITES.T + INTERCEPTS


def velocity(points):
    logits = scores(points) / VARIANCE
    likelihoods = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities = likelihoods / likelihoods.sum(axis=-1, keepdims=True)
    return probabilities @ SITES - points


def endpoint(points, horizon, steps):
    step = horizon / steps
    for _ in range(steps):
        k1 = velocity(points)
        k2 = velocity(points + step * k1 / 2)
        k3 = velocity(points + step * k2 / 2)
        k4 = velocity(points + step * k3)
        points = points + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return points


def tangential_quadrature(origin, tangent, third, start, horizon, order):
    extent = np.abs(tangent).sum()
    third_difference = SITES[start] - SITES[third]
    junction = -(origin @ third_difference + INTERCEPTS[start] - INTERCEPTS[third])
    junction /= tangent @ third_difference
    boundaries = [junction]
    for axis in range(2):
        boundaries.extend((sign - origin[axis]) / tangent[axis] for sign in (-1, 1))
    padding = MAX_SPEED * horizon / np.min(np.abs(tangent))
    cuts = [-extent, extent]
    for boundary in boundaries:
        cuts.extend(
            cut
            for cut in (boundary - padding, boundary, boundary + padding)
            if -extent < cut < extent
        )
    cuts = np.array(sorted(set(cuts)))
    centers = (cuts[1:] + cuts[:-1]) / 2
    radii = (cuts[1:] - cuts[:-1]) / 2
    nodes, weights = roots_legendre(order)
    tangential_coordinates = (centers[:, None] + radii[:, None] * nodes).ravel()
    quadrature_weights = (radii[:, None] * weights).ravel()
    base = origin + tangential_coordinates[:, None] * tangent
    return tangential_coordinates, base, quadrature_weights


def normal_mass(base, normal, lower, upper):
    midpoint = (upper + lower) / 2
    halfwidth = np.maximum(0.0, (upper - lower) / 2)
    coordinates = midpoint[:, None] + halfwidth[:, None] * NORMAL_NODES
    points = base[:, None, :] + coordinates[..., None] * normal
    # On the clipped support the density is a polynomial of degree four in s.
    return halfwidth * (density(points) @ NORMAL_WEIGHTS)


def monotone_threshold(lower, upper, predicate):
    # If the condition holds at both (neither) endpoints, the result is lower
    # (upper). Otherwise the result is the unique crossing to this tolerance.
    for _ in range(38):
        middle = (lower + upper) / 2
        positive = predicate(middle)
        lower = np.where(positive, lower, middle)
        upper = np.where(positive, middle, upper)
    return upper


def plane(start, target):
    difference = SITES[target] - SITES[start]
    length = np.linalg.norm(difference)
    normal = difference / length
    tangent = np.array([-normal[1], normal[0]])
    origin = (INTERCEPTS[start] - INTERCEPTS[target]) / length * normal
    return origin, normal, tangent


def projected_bounds(start, geometry, coordinate, horizon):
    origin, normal, tangent = geometry
    offsets = (SITES - origin) @ normal
    fraction = -np.expm1(-horizon)
    projected_vertices = coordinate[:, None] + fraction * (
        (SITES - origin) @ tangent - coordinate[:, None]
    )
    projected_vertices = np.column_stack((coordinate, projected_vertices))
    projected_scores = scores(origin + projected_vertices[..., None] * tangent)
    relative_logits = (projected_scores - projected_scores[..., start, None]) / VARIANCE
    min_weights = np.exp(relative_logits.min(axis=1))
    max_weights = np.exp(relative_logits.max(axis=1))
    rate = np.ptp(offsets) ** 2 / (4 * VARIANCE) - 1
    psi = horizon if rate == 0 else -np.expm1(-rate * horizon) / rate
    return offsets, min_weights, max_weights, psi


def projected_tests(bounds, signed_distance, horizon):
    offsets, min_weights, max_weights, psi = bounds
    lower_coefficients = offsets + signed_distance[:, None] / psi
    lower_terms = lower_coefficients * np.where(lower_coefficients >= 0, min_weights, max_weights)
    upper_coefficients = offsets + signed_distance[:, None] / np.expm1(horizon)
    upper_terms = upper_coefficients * np.where(upper_coefficients >= 0, max_weights, min_weights)
    return lower_terms.sum(axis=1) > 0, upper_terms.sum(axis=1) >= 0


def directed_masses(start, target, horizon, order, steps):
    third = next(index for index in range(len(SITES)) if index not in (start, target))
    geometry = plane(start, target)
    origin, normal, tangent = geometry
    coordinate, base, weights = tangential_quadrature(
        origin, tangent, third, start, horizon, order
    )
    base_scores = scores(base)
    top_facet = base_scores[:, start] >= base_scores[:, third]
    signed_flow = velocity(base) @ normal
    directed_flux = weights @ (top_facet * density(base) * np.maximum(0, signed_flow))

    lower = np.full_like(coordinate, -MAX_SPEED * horizon)
    upper = np.zeros_like(coordinate)
    for axis in range(2):
        support_ends = (np.array([-1, 1]) - base[:, axis, None]) / normal[axis]
        lower = np.maximum(lower, support_ends.min(axis=1))
        upper = np.minimum(upper, support_ends.max(axis=1))
    third_slope = (SITES[start] - SITES[third]) @ normal
    assert third_slope < 0
    upper = np.minimum(upper, (base_scores[:, third] - base_scores[:, start]) / third_slope)
    keep = lower < upper
    lower, upper, base, coordinate, weights = [
        array[keep] for array in (lower, upper, base, coordinate, weights)
    ]

    # Qualifying points first belong to start and then to target. The two
    # target margins strictly increase with the initial normal coordinate.
    target_slopes = SITES[target] - SITES[[start, third]]
    derivative_error = np.expm1(LIPSCHITZ * horizon)
    assert np.all(target_slopes @ normal > derivative_error * np.linalg.norm(target_slopes, axis=1))

    def reaches_target(signed_distance):
        finish = endpoint(base + signed_distance[:, None] * normal, horizon, steps)
        finish_scores = scores(finish)
        return np.all(finish_scores[:, target, None] > finish_scores[:, [start, third]], axis=1)

    numerical_cut = monotone_threshold(lower, upper, reaches_target)
    reference = weights @ normal_mass(base, normal, numerical_cut, upper)

    bounds = projected_bounds(start, geometry, coordinate, horizon)
    lower_cut = monotone_threshold(lower, upper, lambda s: projected_tests(bounds, s, horizon)[0])
    lower_mass = weights @ normal_mass(base, normal, lower_cut, upper)
    upper_cut = monotone_threshold(lower, upper, lambda s: projected_tests(bounds, s, horizon)[1])
    upper_mass = weights @ normal_mass(base, normal, upper_cut, upper)
    return directed_flux, reference, lower_mass, upper_mass


def overlap_masses(start, horizon, order):
    geometries = [plane(start, target) for target in range(len(SITES)) if target != start]
    normals = np.stack([normal for _, normal, _ in geometries])
    intercepts = np.array([origin @ normal for origin, normal, _ in geometries])
    inverse = np.linalg.inv(normals)
    nodes, weights = roots_legendre(order)
    radius = MAX_SPEED * horizon / 2
    coordinates = radius * (nodes - 1)
    signed_distances = np.stack(np.meshgrid(coordinates, coordinates, indexing="ij"), axis=-1)
    signed_distances = signed_distances.reshape(-1, 2)
    points = (signed_distances + intercepts) @ inverse.T
    tests = []
    for axis, geometry in enumerate(geometries):
        origin, _, tangent = geometry
        bounds = projected_bounds(start, geometry, (points - origin) @ tangent, horizon)
        tests.append(projected_tests(bounds, signed_distances[:, axis], horizon))
    area = abs(np.linalg.det(inverse)) * radius**2
    mass = area * np.outer(weights, weights).ravel() * density(points)
    return [mass @ (tests[0][index] & tests[1][index]) for index in range(2)]


def multiclass_row(horizon, order, steps):
    by_direction = np.zeros((len(SITES), len(SITES), 4))
    for start in range(len(SITES)):
        for target in range(len(SITES)):
            if start != target:
                by_direction[start, target] = directed_masses(start, target, horizon, order, steps)
    flux, reference, lower, upper = by_direction.sum(axis=(0, 1))
    overlap = np.sum([overlap_masses(start, horizon, order) for start in range(len(SITES))], axis=0)
    lower -= overlap[0]
    upper -= overlap[1]
    return {
        "horizon": horizon,
        "endpoint_reference": float(reference),
        "first_order_prediction": float(horizon * flux),
        "maximum_directed_error_over_time_squared": float(
            np.max(np.abs(by_direction[:, :, 1] - horizon * by_direction[:, :, 0])) / horizon**2
        ),
        "lower_certificate_mass": float(lower),
        "upper_certificate_mass": float(upper),
        "uncertainty_mass_over_time_squared": float((upper - lower) / horizon**2),
        "directed_endpoint_reference": by_direction[:, :, 1].tolist(),
        "directed_first_order_prediction": (horizon * by_direction[:, :, 0]).tolist(),
    }


def binary_row(horizon):
    # F(x) = 0.4 - x and a fixed affine readout with boundary x = 0.
    # The initial density is 3(1-x^2)/4 on [-1,1]. The two finite-horizon
    # certificate widths and the exact backward transport width coincide.
    width = min(1.0, 0.4 * np.expm1(horizon))
    probability = 0.75 * (width - width**3 / 3)
    return {
        "horizon": horizon,
        "exact_endpoint_mass": float(probability),
        "exact_lower_and_upper_mass": float(probability),
        "first_order_prediction": 0.3 * horizon,
    }


def run_validation():
    rows = []
    for horizon in HORIZONS:
        coarse = multiclass_row(horizon, order=128, steps=12)
        refined = multiclass_row(horizon, order=256, steps=24)
        pairwise_difference = np.abs(
            np.asarray(refined["directed_endpoint_reference"]) - coarse["directed_endpoint_reference"]
        )
        refined["maximum_directed_reference_refinement_difference"] = float(pairwise_difference.max())
        refined["maximum_certificate_mass_refinement_difference"] = max(
            abs(refined[key] - coarse[key])
            for key in ("lower_certificate_mass", "upper_certificate_mass")
        )
        assert refined["lower_certificate_mass"] <= refined["endpoint_reference"]
        assert refined["endpoint_reference"] <= refined["upper_certificate_mass"]
        rows.append(refined)
    return {
        "binary_closed_form": [binary_row(horizon) for horizon in HORIZONS],
        "matched_gaussian_three_class": {
            "sites": SITES.tolist(),
            "weights": WEIGHTS.tolist(),
            "variance": VARIANCE,
            "density": "9/16 (1-x^2)_+ (1-y^2)_+",
            "reference": "Normal fiber bisection with RK4, using 12 to 24 steps and 128 to 256 quadrature nodes per tangent subinterval",
            "certificates": "Floating-point estimates of the two query-set masses. Pairwise overlap within each start cell is subtracted by quadrature in the two plane distances. Both upper tests also use the speed-bound cutoff.",
            "rows": rows,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = json.dumps(run_validation(), indent=2) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8", newline="\n")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
