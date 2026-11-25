"""Exact-distribution calibration for the Bayes-risk cost of boundary motion."""

from dataclasses import dataclass
from itertools import combinations
from math import fsum

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq
from scipy.special import ndtr

from .flow import MixtureFlow


HORIZONS = (0.002, 0.005, 0.01, 0.025, 0.05)
_SCAN_SIZES = (513, 4097)


@dataclass(frozen=True)
class _Fixture:
    name: str
    beta: float
    centers: tuple
    labels: tuple
    masses: tuple


_FIXTURES = (
    _Fixture("symmetric_binary", 0.72, (-1.15, 1.15), (0, 1), (0.5, 0.5)),
    _Fixture("asymmetric_binary_priors", 0.63, (-1.25, 1.10), (0, 1), (0.72, 0.28)),
    _Fixture("alternating_binary_sources", 0.43, (-3.00, -1.10, 0.95, 3.05), (0, 1, 0, 1),
             (0.29, 0.22, 0.21, 0.28)),
    _Fixture("three_class_mixtures", 0.36, (-4.10, -2.60, -0.95, 0.75, 2.60, 4.15), (0, 1, 2, 0, 1, 2),
             tuple(np.array([0.55, 0.35, 0.62, 0.45, 0.65, 0.38]) / 3.0)),
)


def _scan_roots(model, pair, window, grid_count):
    a, b = pair
    grid = np.linspace(*window, grid_count)
    scores = model.scores(grid[:, None])
    vals = scores[:, b] - scores[:, a]

    def difference(x):
        point = model.scores(np.array([[x]]))[0]
        return point[b] - point[a]

    roots = [float(x) for x in grid[vals == 0]]
    brackets = np.flatnonzero((vals[:-1] * vals[1:] < 0))
    roots += [float(brentq(difference, grid[i], grid[i + 1], xtol=1e-14)) for i in brackets]
    return sorted(roots)


def _joint_density(mixture, labels, centers, masses, beta, x):
    d = x - centers
    source = masses * np.exp(-(d * d) / (2 * beta)) / np.sqrt(2 * np.pi * beta)
    values = np.bincount(labels, weights=source, minlength=mixture)
    gradients = np.bincount(labels, weights=source * (-d / beta), minlength=mixture)
    return values, gradients


def _boundary_roster(model, centers, labels, masses, beta):
    window = [float(centers.min() - 8 * np.sqrt(beta)), float(centers.max() + 8 * np.sqrt(beta))]
    active, checked = [], []
    for a, b in combinations(range(len(model.classes)), 2):
        members = np.flatnonzero((labels == a) | (labels == b))
        order = members[np.argsort(centers[members])]
        signs = labels[order] == b
        max_roots = int(np.count_nonzero(signs[1:] != signs[:-1]))
        coarse, fine = [_scan_roots(model, (a, b), window, size) for size in _SCAN_SIZES]
        # Pairwise density ties factor into a positive Gaussian term times
        # an exponential polynomial. Ordered coefficient sign variation bounds
        # the total real zeros with multiplicity. All these fixtures saturate it.
        if len(coarse) != max_roots or len(fine) != max_roots or not np.allclose(coarse, fine, atol=2e-12, rtol=0):
            raise RuntimeError(f"Cannot establish a complete root roster for pair {(a, b)}")
        analytical_root = None
        if len(members) == 2:
            i, j = members
            analytical_root = float((centers[i] + centers[j]) / 2 +
                                    beta * np.log(masses[i] / masses[j]) / (centers[j] - centers[i]))
            if not np.isclose(fine[0], analytical_root, atol=2e-12, rtol=0):
                raise RuntimeError("Numerical and analytic single-pair tie differ")
        witnesses = []
        for root in fine:
            bracket = [root - 1e-5, root + 1e-5]
            scores = model.scores(np.array(bracket)[:, None])
            signs = scores[:, b] - scores[:, a]
            if signs[0] * signs[1] >= 0 or np.min(np.abs(signs)) < 1e-9:
                raise RuntimeError("Root lacks a robust numerical sign-change witness")
            if witnesses and bracket[0] <= witnesses[-1]["bracket"][1]:
                raise RuntimeError("Witness brackets overlap")
            witnesses.append({"bracket": bracket, "score_differences_b_minus_a": signs.tolist()})
        checked.append({"pair": [a, b], "sign_variation_bound": max_roots,
                        "coarse": coarse, "fine": fine, "analytical_single_pair_root": analytical_root,
                        "sign_change_witnesses": witnesses})
        roots_for_risk = [analytical_root] if analytical_root is not None else fine
        for root in roots_for_risk:
            score = model.scores(np.array([[root]]))[0]
            third = np.delete(score, [a, b])
            top_gap = float(score[a] - np.max(third)) if len(third) else None
            if top_gap is not None and abs(top_gap) <= 1e-9:
                raise RuntimeError("Unexpected triple tie")
            if top_gap is not None and top_gap < 0:
                continue
            _, gradients = _joint_density(len(model.classes), labels, centers, masses, beta, root)
            derivative = float(gradients[b] - gradients[a])
            if abs(derivative) < 1e-10:
                raise RuntimeError("Irregular top tie")
            numerical_velocity = float(model.field(np.array([[root]]))[0, 0])
            if analytical_root is not None and len(model.classes) == 2:
                # At a tie of two sources their posterior weights are exactly equal.
                velocity = float((centers[i] + centers[j]) / 2 - root)
                if not np.isclose(velocity, numerical_velocity, atol=2e-12, rtol=0):
                    raise RuntimeError("Numerical and analytic single-pair tie velocity differ")
            else:
                velocity = numerical_velocity
            left, right = (a, b) if derivative > 0 else (b, a)
            active.append(dict(root=root, left=left, right=right, score_gap_to_other=top_gap,
                               density_slope_right_minus_left=abs(derivative), velocity=velocity,
                               numerical_flow_velocity=numerical_velocity,
                               coefficient=0.5 * abs(derivative) * velocity ** 2))
    active.sort(key=lambda row: row["root"])
    if not active or any(a["right"] != b["left"] for a, b in zip(active[:-1], active[1:], strict=True)):
        raise RuntimeError("Sorted top boundaries do not compose contiguous decisions")
    roots = np.asarray([row["root"] for row in active])
    inner = (roots[:-1] + roots[1:]) / 2
    probes = np.concatenate(([window[0]], inner, [window[1]]))
    winners = [active[0]["left"], *[row["right"] for row in active]]
    if not np.array_equal(model.predict(probes[:, None]), winners):
        raise RuntimeError("Raster-independent class decision check failed between roots")
    return window, checked, active, roots, winners


def _cdf_error(roots, winners, centers, labels, masses, beta):
    endpoints = np.concatenate(([-np.inf], roots, [np.inf]))
    z = (endpoints[:, None] - centers) / np.sqrt(beta)
    masses_in_cells = np.diff(ndtr(z), axis=0) * masses
    # Add wrong-class joint mass directly rather than subtracting accuracy from 1.
    return fsum(float(weight) for matrix_row, winner in zip(masses_in_cells, winners, strict=True)
                for source_label, weight in zip(labels, matrix_row, strict=True) if source_label != winner)


def _solve_preimages(model, roots):
    def inverse(_, state):
        return -model.field(state[:, None])[:, 0]

    solved = solve_ivp(inverse, (0, HORIZONS[-1]), roots, t_eval=HORIZONS,
                       method="DOP853", atol=2e-15, rtol=2e-13, max_step=0.004)
    if not solved.success or tuple(solved.t) != HORIZONS:
        raise RuntimeError(f"Inverse flow integration failed: {solved.message}")
    return solved.y.T, int(solved.nfev)


def _fixture_report(fixture):
    centers = np.asarray(fixture.centers)
    labels = np.asarray(fixture.labels)
    masses = np.asarray(fixture.masses) / fsum(fixture.masses)
    beta = fixture.beta
    model = MixtureFlow(centers[:, None], labels, masses, beta)
    window, checked, active, roots, winners = _boundary_roster(model, centers, labels, masses, beta)
    priors = np.bincount(labels, weights=masses)
    coefficient = fsum(row["coefficient"] for row in active)
    for row in active:
        joint, _ = _joint_density(len(priors), labels, centers, masses, beta, row["root"])
        row["feature_density"] = fsum(joint)
    flip_coefficient = fsum(row["feature_density"] * abs(row["velocity"]) for row in active)
    bayes_error = _cdf_error(roots, winners, centers, labels, masses, beta)
    inverse, backward_nfe = _solve_preimages(model, roots)
    if fixture.name == "symmetric_binary":
        if not np.allclose(inverse, roots, atol=2e-13, rtol=0):
            raise RuntimeError("Numerical flow drifts from the exact stationary tie")
        inverse = np.broadcast_to(roots, inverse.shape)
    observations = []
    for h, preimages in zip(HORIZONS, inverse, strict=True):
        lower, upper = np.minimum(roots, preimages), np.maximum(roots, preimages)
        if np.any(upper[:-1] >= lower[1:]):
            raise RuntimeError("Short-time crossing strips unexpectedly overlap")

        def contrast(x, left, right):
            values, _ = _joint_density(len(priors), labels, centers, masses, beta, x)
            return abs(values[right] - values[left])

        quadrature = [quad(contrast, lo, hi, args=(face["left"], face["right"]), epsabs=1e-15, epsrel=1e-10)[0]
                      for lo, hi, face in zip(lower, upper, active, strict=True)]
        total_quad = fsum(quadrature)
        mass = fsum(float(np.dot(masses, ndtr((hi - centers) / np.sqrt(beta)) - ndtr((lo - centers) / np.sqrt(beta))))
                    for lo, hi in zip(lower, upper, strict=True))
        error_cdf = _cdf_error(preimages, winners, centers, labels, masses, beta)
        excess_cdf = error_cdf - bayes_error
        if not np.isclose(total_quad, excess_cdf, atol=2e-13, rtol=0):
            raise RuntimeError("Localized quadrature and full Gaussian-CDF risk differ")
        forward = solve_ivp(lambda _, state: model.field(state[:, None])[:, 0], (0, h), preimages,
                            t_eval=[h], method="RK45", rtol=2e-12, atol=2e-14, max_step=0.003)
        if not forward.success:
            raise RuntimeError("Forward recovery of boundary points failed")
        residual = float(np.max(np.abs(forward.y[:, -1] - roots)))
        if residual > 2e-11:
            raise RuntimeError("Inverse-flow round trip did not recover static boundaries")
        observations.append({"horizon": h, "preimages": preimages.tolist(), "error_cdf": error_cdf,
                             "excess_quadrature": total_quad, "excess_cdf": excess_cdf,
                             "excess_per_boundary": quadrature, "ratio_to_horizon_squared": total_quad / h ** 2,
                             "flip_mass": mass, "ratio_to_horizon": mass / h,
                             "round_trip_max_error": residual})
    return {
        "name": fixture.name, "beta": beta, "centers": centers.tolist(), "labels": labels.tolist(),
        "masses": masses.tolist(), "priors": priors.tolist(), "scan_window": window, "pairwise_roots": checked,
        "boundaries": active, "predicted_coefficient": coefficient, "predicted_flip_coefficient": flip_coefficient, "bayes_error": bayes_error,
        "cell_winners": winners, "backward_solver_nfe": backward_nfe, "observations": observations,
    }


def oracle_experiments():
    """Return deterministic fixtures and measurements without any learned case data."""
    return {"schema_version": 1, "horizons": list(HORIZONS), "scan_sizes": list(_SCAN_SIZES),
            "risk": "misclassification probability under the specified source masses as joint class densities",
            "fixtures": [_fixture_report(fixture) for fixture in _FIXTURES]}
