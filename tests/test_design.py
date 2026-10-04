"""Versuchsplanung: Sobol-Punkte, Auswahlregeln, aktives Nachsampeln."""

import numpy as np
import pytest

import sur_design as D
import sur_models as M


def test_sobol_points_lie_in_the_design_space_and_use_each_level_equally():
    u, r, cs2 = D.sobol_points(1024, 1)
    assert u.min() >= M.U_MIN and u.max() <= M.U_MAX and r.min() >= M.R_MIN and r.max() <= M.R_MAX
    counts = [int((cs2 == c).sum()) for c in M.CS2_LEVELS]
    assert sum(counts) == 1024 and max(counts) - min(counts) <= 25 and set(cs2) == set(M.CS2_LEVELS)


def test_sobol_points_are_reproducible_and_depend_on_the_seed():
    a, b, c = D.sobol_points(64, 5), D.sobol_points(64, 5), D.sobol_points(64, 6)
    assert all((x == y).all() for x, y in zip(a, b)) and not (a[0] == c[0]).all()


def test_sobol_points_fill_the_space_better_than_random_ones():
    """Kleinster Abstand zwischen zwei Punkten (u, r skaliert): bei Sobol größer als bei Zufall."""
    def min_dist(u, r):
        p = np.column_stack([(u - M.U_MIN) / (M.U_MAX - M.U_MIN), (r - M.R_MIN) / (M.R_MAX - M.R_MIN)])
        d = np.sqrt(((p[:, None] - p[None, :]) ** 2).sum(axis=2)) + np.eye(len(p)) * 9
        return d.min()
    u, r, _ = D.sobol_points(64, 2)
    rng = np.random.default_rng(0)
    rand = min_dist(rng.uniform(M.U_MIN, M.U_MAX, 64), rng.uniform(M.R_MIN, M.R_MAX, 64))
    assert min_dist(u, r) > rand


def test_select_sobol_takes_the_first_pool_points():
    assert D.select_sobol(5) == [0, 1, 2, 3, 4]


def test_select_random_is_reproducible_distinct_and_seed_dependent():
    a, b, c = D.select_random(600, 20, 1), D.select_random(600, 20, 1), D.select_random(600, 20, 2)
    assert a == b and a != c and len(set(a)) == 20 and all(0 <= i < 600 for i in a)


def test_next_active_picks_the_candidate_farthest_from_the_data():
    """Training bei u = 0 und u = 1 (skaliert), Kandidaten dazwischen: größte Unsicherheit in der Mitte; schon gewählte Punkte werden übergangen."""
    ts = np.linspace(0, 1, 11)
    pool = np.column_stack([ts, np.full(11, 0.5), np.full(11, 0.5)])
    gp = M.gp_build(pool[[0, 10]], np.array([1.0, 2.0]), np.log([1.0, 0.3, 0.3, 0.3, 0.05]))
    assert D.next_active(gp, pool, [0, 10]) == 5
    assert D.next_active(gp, pool, [0, 10, 5]) in (4, 6)


def test_select_active_is_a_prefix_family_and_starts_random():
    rng = np.random.default_rng(0)
    x = rng.uniform(size=(80, 3))
    y = np.sin(4 * x[:, 0]) + x[:, 1]
    long = D.select_active(x, y, 14, 3)
    short = D.select_active(x, y, 11, 3)
    assert long[:11] == short and long[:8] == D.select_random(80, 8, 3) and len(set(long)) == 14


def test_select_dispatches_and_rejects_unknown_designs():
    x = np.random.default_rng(0).uniform(size=(40, 3))
    y = x[:, 0]
    assert D.select("sobol", x, y, 4, 0) == [0, 1, 2, 3] and D.select("random", x, y, 4, 1) == D.select_random(40, 4, 1) and len(D.select("active", x, y, 9, 1)) == 9
    with pytest.raises(ValueError):
        D.select("grid", x, y, 4, 0)
