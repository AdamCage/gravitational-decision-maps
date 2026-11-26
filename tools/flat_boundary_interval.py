"""Compare facet slabs with validated interval integration on two fixed affine fixtures.

Run from the repository root as ``python -m tools.flat_boundary_interval``.
"""

from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass
from decimal import Context, Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN
from itertools import combinations
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "crossing_results" / "flat_boundary_interval.json"
PRECISION = 38
FLOOR = Context(prec=PRECISION, rounding=ROUND_FLOOR)
CEILING = Context(prec=PRECISION, rounding=ROUND_CEILING)
NEAREST = Context(prec=PRECISION, rounding=ROUND_HALF_EVEN)
ZERO, ONE, TWO = Decimal(0), Decimal(1), Decimal(2)
STEP = Decimal("0.015625")
HORIZONS = ("0.5", "1.25", "2.25")
TUBE_REFINEMENTS = 2

# This exploratory roster and the integration settings were fixed before running.
FIXTURES = {
    "centered_cospherical": {
        "beta": "1.2",
        "sites": (("1", "0"), (".6", ".8"), ("0", "1"), ("-1", "0"), ("-.6", "-.8"), ("0", "-1")),
        "queries": (("1.34", ".56"), ("1.34", ".669"), ("1.34", ".671"), (".60", ".2999"),
                    (".60", ".3001"), (".60", ".24"), (".05", ".95"), (".02", "-.01")),
    },
    "reflective_rectangle": {
        "beta": "2",
        "sites": (("-1", "-.75"), ("-1", ".75"), ("1", "-.75"), ("1", ".75")),
        "queries": ((".2", ".8"), ("-.2", ".8"), (".02", ".3"), ("-.02", ".3"), (".002", ".3"),
                    ("-.002", ".3"), (".00002", ".3"), ("-.00002", ".3"), (".003", ".004")),
    },
}

# Post hoc comparison, keeping the original all-rival slab as specified.
# Equal-radius cyclic neighbors share a Voronoi ray. For the rectangle,
# the two one-dimensional nearest-site decisions give only axial facets.
ACTIVE_FACETS = {
    "centered_cospherical": ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)),
    "reflective_rectangle": ((0, 1), (0, 2), (1, 3), (2, 3)),
}


@dataclass(frozen=True, slots=True)
class Interval:
    lo: Decimal
    hi: Decimal

    def __add__(self, other):
        other = point(other)
        return Interval(FLOOR.add(self.lo, other.lo), CEILING.add(self.hi, other.hi))

    __radd__ = __add__

    def __neg__(self):
        return Interval(self.hi.copy_negate(), self.lo.copy_negate())

    def __sub__(self, other):
        return self + -point(other)

    def __rsub__(self, other):
        return point(other) + -self

    def __mul__(self, other):
        other = point(other)
        pairs = ((self.lo, other.lo), (self.lo, other.hi), (self.hi, other.lo), (self.hi, other.hi))
        return Interval(min(FLOOR.multiply(a, b) for a, b in pairs),
                        max(CEILING.multiply(a, b) for a, b in pairs))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = point(other)
        if other.lo <= ZERO:
            raise ValueError("Division requires a strictly positive denominator interval")
        return self * Interval(FLOOR.divide(ONE, other.hi), CEILING.divide(ONE, other.lo))

    def exp(self):
        # decimal.exp is correctly rounded to nearest. Adjacent representable
        # numbers widen the computed endpoints in both directions.
        return Interval(NEAREST.next_minus(NEAREST.exp(self.lo)), NEAREST.next_plus(NEAREST.exp(self.hi)))

    def intersect(self, other):
        lo, hi = max(self.lo, other.lo), min(self.hi, other.hi)
        if lo > hi:
            raise ArithmeticError("Two certified enclosures do not intersect")
        return Interval(lo, hi)


def point(value):
    if isinstance(value, Interval):
        return value
    bound = Decimal(value)
    return Interval(bound, bound)


def exact(value):
    if value.lo != value.hi:
        raise ArithmeticError("This projection or time requires exact representation")
    return value.lo


def dot(left, right):
    return sum((a * b for a, b in zip(left, right, strict=True)), point(ZERO))


class Flow:
    def __init__(self, fixture):
        self.beta = point(fixture["beta"])
        self.sites = tuple(tuple(map(point, coordinates)) for coordinates in fixture["sites"])
        self.count = len(self.sites)
        self.pairs = tuple(combinations(range(self.count), 2))
        self.hull = tuple(Interval(min(z.lo for z in column), max(z.hi for z in column)) for column in zip(*self.sites))
        self.delta = tuple(tuple(tuple(x - y for x, y in zip(a, b, strict=True)) for b in self.sites)
                           for a in self.sites)
        squared_norms = tuple(dot(site, site) for site in self.sites)
        self.offset = tuple(tuple((a - b) / TWO for b in squared_norms) for a in squared_norms)
        self.decay = (-point(STEP)).exp()
        self.move = ONE - self.decay
        self.remainder = point(STEP) - ONE + self.decay

    def margin(self, winner, rival, box):
        return dot(self.delta[winner][rival], box) - self.offset[winner][rival]

    def certifies(self, winner, box):
        return all(self.margin(winner, rival, box).lo > ZERO for rival in range(self.count) if winner != rival)

    def winner(self, box):
        return next((winner for winner in range(self.count) if self.certifies(winner, box)), None)

    def posterior(self, box):
        return tuple(point(ONE) / (ONE + sum(
            ((self.margin(rival, i, box) / self.beta).exp() for rival in range(self.count) if rival != i), point(ZERO)
        )) for i in range(self.count))

    def mean(self, probabilities):
        means = []
        for hull, column in zip(self.hull, zip(*self.sites), strict=True):
            lower_anchor = hull.lo + sum((p * (a - hull.lo) for p, a in zip(probabilities, column, strict=True)), point(ZERO))
            upper_anchor = hull.hi - sum((p * (hull.hi - a) for p, a in zip(probabilities, column, strict=True)), point(ZERO))
            means.append(hull.intersect(lower_anchor).intersect(upper_anchor))
        return tuple(means)

    def step(self, box):
        # Variation of constants and the coordinate hull contain every time up
        # to STEP, independent of the posterior and without a first-exit test.
        tube = tuple(x + Interval(ZERO, self.move.hi) * (hull - x) for x, hull in zip(box, self.hull, strict=True))
        for _ in range(TUBE_REFINEMENTS):
            average = self.mean(self.posterior(tube))
            tube = tuple(y.intersect(x + Interval(ZERO, STEP) * (mu - y))
                         for x, y, mu in zip(box, tube, average, strict=True))

        probabilities = self.posterior(tube)
        field = tuple(mu - y for mu, y in zip(self.mean(probabilities), tube, strict=True))
        curvature = [point(ZERO) for _ in box]
        for i, j in self.pairs:
            delta = self.delta[i][j]
            coefficient = probabilities[i] * probabilities[j] * dot(delta, field) / self.beta
            curvature = [value + coefficient * component for value, component in zip(curvature, delta, strict=True)]

        average = self.mean(self.posterior(box))
        endpoint = tuple(y.intersect(self.decay * x + self.move * mu + self.remainder * bound)
                         for x, y, mu, bound in zip(box, tube, average, curvature, strict=True))
        return tube, endpoint

    def slab_loss(self, winner, rival):
        normal = self.delta[winner][rival]
        tangent = (-normal[1], normal[0])
        squared_norm = dot(normal, normal)
        groups = {}
        for site in self.sites:
            offset = self.margin(winner, rival, site)
            factor = (-(offset * offset) / (TWO * self.beta * squared_norm)).exp()
            key = exact(dot(tangent, site))
            mass, weighted_offset = groups.get(key, (point(ZERO), point(ZERO)))
            groups[key] = mass + factor, weighted_offset + factor * offset
        # The normal velocity on the plane in units of the winning margin is a
        # convex average of the exact group ratios. Use their lower bounds.
        return max(ZERO, min((weighted / mass).lo for mass, weighted in groups.values()).copy_negate())


def floating_comparisons(fixture):
    """Evaluate the existing transfer and covariance guards on this exact-input roster."""
    import numpy as np

    from crossing.flow import MixtureFlow
    from tools.flat_boundary_transfer import LocalPlaneGuard, bisector_offsets

    sites = np.array(fixture["sites"], dtype=float)
    queries = np.array(fixture["queries"], dtype=float)
    model = MixtureFlow(sites, np.arange(len(sites)), np.ones(len(sites)), float(fixture["beta"]))
    distances, _ = bisector_offsets(model.centered)
    local = LocalPlaneGuard(model, distances)
    prepared = local.prepare(model.scores(queries))
    diagnostics = {}
    for time in HORIZONS:
        local_stable = local.stable(prepared, float(time))
        covariance = model.certify(queries, float(time))
        diagnostics[time] = (local_stable, covariance)
    return diagnostics


def measure(fixture, active_facets):
    flow = Flow(fixture)
    floating = floating_comparisons(fixture)
    active_neighbors = tuple({right if left == site else left for left, right in active_facets if site in (left, right)}
                             for site in range(flow.count))
    observations = []
    time_by_step = {int(exact(point(t) / STEP)): t for t in HORIZONS}
    if any(Decimal(time) != Decimal(step) * STEP for step, time in time_by_step.items()):
        raise ValueError("Each reported horizon must be an integer multiple of the step")
    for query_index, query in enumerate(fixture["queries"]):
        start = tuple(map(point, query))
        initial = flow.winner(start)
        if initial is None:
            raise ValueError(f"Initial query needs a strict unique winner: {query}")
        rivals = [rival for rival in range(flow.count) if rival != initial]
        losses = [flow.slab_loss(initial, rival) for rival in rivals]
        box = start
        stable = True
        departed = False
        for index in range(1, max(time_by_step) + 1):
            tube, box = flow.step(box)
            stable = stable and flow.certifies(initial, tube)
            endpoint = flow.winner(box)
            departed = departed or (endpoint is not None and endpoint != initial)
            if index not in time_by_step:
                continue
            horizon = time_by_step[index]
            duration = point(horizon).exp() - ONE
            clear_by_rival = {rival: flow.margin(initial, rival, start).lo > (loss * duration).hi
                              for rival, loss in zip(rivals, losses, strict=True)}
            slab = all(clear_by_rival.values())
            active_slab = all(clear_by_rival[rival] for rival in active_neighbors[initial])
            if stable and (departed or endpoint not in (None, initial)):
                raise ArithmeticError("Whole-interval and endpoint certificates contradict")
            if active_slab and departed:
                raise ArithmeticError("Facet slab and interval departure certificate contradict")
            local_stable, covariance = floating[horizon]
            if int(covariance.initial[query_index]) != initial:
                raise ArithmeticError("The floating and exact static labels disagree")
            covariance_endpoint = int(covariance.endpoint[query_index])
            observations.append({
                "query": query,
                "horizon": horizon,
                "initial_site_index": initial,
                "interval_endpoint_site_index": endpoint,
                "interval_endpoint": "abstain" if endpoint is None else "same" if endpoint == initial else "changed",
                "interval_whole_path": "stable" if stable else "departure" if departed else "abstain",
                "slab_whole_path": "stable" if slab else "abstain",
                "interval_endpoint_width": [str((point(axis.hi) - axis.lo).hi) for axis in box],
                "active_facet_slab_whole_path": "stable" if active_slab else "abstain",
                "floating_local_plane_whole_path": "stable" if local_stable[query_index] else "abstain",
                "floating_covariance_whole_path": "stable" if covariance.stable[query_index] else
                "departure" if covariance.forced_crossing[query_index] else "abstain",
                "floating_covariance_endpoint": "abstain" if covariance_endpoint < 0 else
                "same" if covariance_endpoint == initial else "changed",
            })
    return observations


def run():
    data = {name: measure(fixture, ACTIVE_FACETS[name]) for name, fixture in FIXTURES.items()}
    columns = ("interval_endpoint", "interval_whole_path", "slab_whole_path", "active_facet_slab_whole_path",
               "floating_local_plane_whole_path", "floating_covariance_whole_path", "floating_covariance_endpoint")
    return {
        "study": "Exploratory exact-input, outward-rounded affine fixture comparison",
        "precision_decimal_digits": PRECISION,
        "rounding": "Decimal ROUND_FLOOR and ROUND_CEILING for arithmetic; correctly rounded exp widened by next_minus and next_plus",
        "step": str(STEP),
        "tube_refinements": TUBE_REFINEMENTS,
        "horizons": HORIZONS,
        "fixtures": FIXTURES,
        "post_hoc_comparisons": {
            "active_facet_slab_whole_path": "Same outward-rounded slab, limited to genuine Voronoi facets",
            "active_facet_site_indices": ACTIVE_FACETS,
            "floating_local_plane_whole_path": "All-rival query-local guard from the affine transfer study; floating-point diagnostic",
            "floating_covariance": "Combined query-local Taylor, drag and state guards from MixtureFlow; floating-point diagnostics",
        },
        "summary": {
            name: {
                time: {column: dict(Counter(row[column] for row in rows if row["horizon"] == time))
                       for column in columns}
                for time in HORIZONS
            }
            for name, rows in data.items()
        },
        "observations": data,
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(), indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
