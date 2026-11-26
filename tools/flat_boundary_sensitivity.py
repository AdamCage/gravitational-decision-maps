"""Calibrate a fixed classifier's sensitivity to a corner-mass perturbation."""

from argparse import ArgumentParser
import json
import math
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import ndtr


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "crossing_results" / "flat_boundary_sensitivity.json"

# This exploratory grid was specified before running the calibration.
A, B = 1.0, 0.75
BETA = 2.0
SITES = np.array([[-A, -B], [-A, B], [A, -B], [A, B]])
X_LIMIT = 0.3
Y_RANGE = (0.3, 1.2)
HORIZONS = (0.25, 0.75, 1.5)
MAGNITUDES = (0.08, 0.04, 0.02, 0.01, 0.005)
NODES = 16
RTOL, ATOL, ROOT_TOL = 2e-11, 2e-13, 1e-13


def predicted_coefficient(initial_y, horizon):
    """Return the predicted limit of minus the boundary abscissa over epsilon."""
    normal_rate = A * A / BETA - 1

    def response(time, state):
        y, _ = state
        tangential_component = math.tanh(B * y / BETA)
        forcing = 0.25 * A * (1 + tangential_component)
        return [B * tangential_component - y, math.exp(-normal_rate * time) * forcing]

    path = solve_ivp(response, (0, horizon), [initial_y, 0.0], method="DOP853", rtol=RTOL, atol=ATOL)
    if not path.success:
        raise RuntimeError(path.message)
    return path.y[1, -1]


def observed_boundary(initial_y, horizon, epsilon, *, rtol, atol, root_tol):
    """Find the initial boundary using the full two-dimensional perturbed flow."""
    log_mass = np.array([0.0, 0.0, 0.0, epsilon])

    def field(_time, state):
        differences = SITES - state
        logits = log_mass - np.sum(differences**2, axis=1) / (2 * BETA)
        masses = np.exp(logits - max(logits))
        return (masses @ differences) / masses.sum()

    def final_abscissa(initial_x):
        path = solve_ivp(
            field, (0, horizon), [initial_x, initial_y], method="DOP853", rtol=rtol, atol=atol
        )
        if not path.success:
            raise RuntimeError(path.message)
        if path.y[1, -1] <= 0:
            raise RuntimeError("The trajectory left the upper half-plane")
        return path.y[0, -1]

    return brentq(final_abscissa, -X_LIMIT, X_LIMIT, xtol=root_tol)


def quadrature(count):
    nodes, weights = leggauss(count)
    midpoint = sum(Y_RANGE) / 2
    half_width = (Y_RANGE[1] - Y_RANGE[0]) / 2
    ordinates = midpoint + half_width * nodes
    gaussian_weights = weights * half_width * np.exp(-ordinates**2 / 2) / np.sqrt(2 * np.pi)
    return ordinates, gaussian_weights


def calibrate(horizon, epsilons, *, count, rtol, atol, root_tol):
    ordinates, weights = quadrature(count)
    predicted = np.array([predicted_coefficient(y, horizon) for y in ordinates])
    mass_limit = float(weights @ predicted / np.sqrt(2 * np.pi))
    rows = []
    for epsilon in epsilons:
        abscissas = np.array([
            observed_boundary(y, horizon, epsilon, rtol=rtol, atol=atol, root_tol=root_tol)
            for y in ordinates
        ])
        mass = float(weights @ np.abs(ndtr(abscissas) - 0.5))
        row = {
            "epsilon": epsilon,
            "unconditional_switch_mass_inside_strip": mass,
            "max_abs_initial_boundary_abscissa": float(np.max(np.abs(abscissas))),
        }
        if epsilon:
            row.update({
                "nodes_with_displacement_opposite_to_prediction": int(np.count_nonzero(epsilon * abscissas > 0.0)),
                "switch_mass_divided_by_abs_epsilon": mass / abs(epsilon),
                "relative_error_in_switch_mass_coefficient": abs(mass / abs(epsilon) - mass_limit) / mass_limit,
                "max_relative_error_in_displacement_coefficient": float(
                    np.max(np.abs(-abscissas / epsilon - predicted) / predicted)
                ),
            })
        rows.append(row)
    return {
        "horizon": horizon,
        "predicted_switch_mass_coefficient": mass_limit,
        "predicted_displacement_coefficient_range_at_nodes": [float(min(predicted)), float(max(predicted))],
        "measurements": rows,
    }


def strip_clearance():
    """Check the trajectory boundary positions needed for the strip integral."""
    time = max(HORIZONS)
    largest_epsilon = max(MAGNITUDES)

    def clearance(distance, site_coordinate):
        contraction = 1 - site_coordinate**2 / BETA
        forcing = site_coordinate * largest_epsilon / 2
        perturbation_bound = forcing * (-math.expm1(-time * contraction)) / contraction
        return distance * math.exp(-time) - perturbation_bound

    horizontal = clearance(X_LIMIT, A)
    vertical = clearance(Y_RANGE[0], B)
    if min(horizontal, vertical) <= 0:
        raise RuntimeError("The strip clearance bound must be positive")
    return {"vertical_side_abscissa": horizontal, "lower_side_ordinate": vertical}


def run():
    epsilons = [0.0, *(sign * value for value in MAGNITUDES for sign in (-1.0, 1.0))]
    clearance = strip_clearance()
    results = [
        calibrate(t, epsilons, count=NODES, rtol=RTOL, atol=ATOL, root_tol=ROOT_TOL) for t in HORIZONS
    ]
    refined = calibrate(
        max(HORIZONS),
        [-min(MAGNITUDES), min(MAGNITUDES)],
        count=2 * NODES,
        rtol=RTOL / 8,
        atol=ATOL / 8,
        root_tol=ROOT_TOL / 8,
    )
    baseline = {row["epsilon"]: row for row in results[-1]["measurements"]}
    refined_differences = [
        abs(row["switch_mass_divided_by_abs_epsilon"] - baseline[row["epsilon"]]["switch_mass_divided_by_abs_epsilon"])
        for row in refined["measurements"]
    ]
    return {
        "study": "Exploratory fixed-horizon perturbation of the driving field with frozen labels",
        "design": {
            "sites": SITES.tolist(),
            "beta": BETA,
            "static_labels": "Nearest site with equal masses",
            "field_perturbation": "Multiply the upper-right site's driving mass by exp(epsilon)",
            "feature_law": "Standard bivariate Gaussian, without class labels",
            "initial_x_range": [-X_LIMIT, X_LIMIT],
            "initial_y_range": list(Y_RANGE),
            "horizons": list(HORIZONS),
            "epsilons": epsilons,
            "gauss_legendre_nodes": NODES,
            "trajectory_solver": "DOP853",
            "trajectory_rtol": RTOL,
            "trajectory_atol": ATOL,
            "brent_xtol": ROOT_TOL,
        },
        "positive_strip_clearance_lower_bounds": clearance,
        "results": results,
        "numerical_refinement": {
            "horizon": max(HORIZONS),
            "epsilons": [-min(MAGNITUDES), min(MAGNITUDES)],
            "gauss_legendre_nodes": 2 * NODES,
            "trajectory_rtol": RTOL / 8,
            "trajectory_atol": ATOL / 8,
            "brent_xtol": ROOT_TOL / 8,
            "max_absolute_change_in_observed_mass_coefficient": max(refined_differences),
        },
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(), indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
