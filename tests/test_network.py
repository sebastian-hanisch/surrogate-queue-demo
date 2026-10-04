"""Simulator und Näherungen: Erlang C, Verkehrsgleichungen, Jackson, Whitt-Zerlegung, Simulation von Hand (Warten am Kran, Rückläufer), Sampler, Reproduzierbarkeit, Simulation gegen Formel."""

import math

import numpy as np
import pytest

import sur_network as N
from conftest import Route, Seq


def test_erlang_c_wait_by_hand():
    """M/M/1, a = 0.5: Wq = 1. M/M/2, a = 1: Erlang B 0.2, P(warten) = 1/3, Wq = 1/3. Ohne Gleichgewicht unendlich."""
    assert N.erlang_c_wait(1, 0.5) == pytest.approx(1.0) and N.erlang_c_wait(2, 1.0) == pytest.approx(1 / 3) and N.erlang_c_wait(2, 2.0) == math.inf


def test_traffic_rates_by_hand():
    """Zwei Stationen, γ = (1, 0), die Hälfte von 2 zurück zu 1: λ = (2, 2)."""
    assert N.traffic_rates([1.0, 0.0], [[0, 1], [0.5, 0]]) == pytest.approx([2.0, 2.0])


def test_arrival_rate_makes_the_crane_utilisation_equal_u():
    """Kran: 2 Geräte, 2.4 min, Rate γ/(1 − r): Auslastung = γ/(1 − r)·2.4/2 = u."""
    for u, r in ((0.3, 0.0), (0.825, 0.2), (0.95, 0.4)):
        g = N.arrival_rate(u, r)
        assert g / (1 - r) * 2.4 / 2 == pytest.approx(u)
    assert N.arrival_rate(0.825, 0.2) == pytest.approx(0.55)


def test_jackson_total_for_light_traffic_is_the_sum_of_the_visits():
    """Ohne Warten: Gate 3 min + Kran 2.4 min + Stapel 4 min = 9.4; mit r = 0.2 besucht der Lkw den Kran und den Stapel im Mittel 1/(1 − r)-mal: 3 + 6.4/0.8 = 11.0."""
    assert N.jackson_total(1e-6, 0.0) == pytest.approx(9.4, abs=1e-3) and N.jackson_total(1e-6, 0.2) == pytest.approx(11.0, abs=1e-3)


def test_jackson_total_matches_the_value_of_the_network_piece():
    """Stück 12: 33 Lkw/h = 0.55 je Minute, 20 % Rückläufer: 19.68 min."""
    assert N.jackson_total(0.55, 0.2) == pytest.approx(19.68, abs=0.01)


def test_departure_scv_by_hand():
    assert N.departure_scv(0.5, 1, 1.0, 0.0) == pytest.approx(0.75) and N.departure_scv(0.5, 1, 1.0, 4.0) == pytest.approx(1.75)
    assert N.departure_scv(0.5, 4, 1.0, 5.0) == pytest.approx(1 + 0.25 * 4 / 2) and N.departure_scv(0.8, 1, 2.0, 1.0) == pytest.approx(1.36)


def test_qna_total_equals_jackson_for_exponential_service_and_matches_stueck_12():
    assert N.qna_total(0.55, 0.2, 1.0) == pytest.approx(N.jackson_total(0.55, 0.2))
    assert N.qna_total(0.55, 0.2, 0.0) == pytest.approx(14.39, abs=0.01) and N.qna_total(0.55, 0.2, 4.0) == pytest.approx(35.55, abs=0.01)


def test_qna_total_grows_with_the_streuung():
    totals = [N.qna_total(0.5, 0.1, c) for c in (0.25, 0.5, 1.0, 2.0, 4.0)]
    assert all(a < b for a, b in zip(totals, totals[1:]))


def test_simulation_by_hand_with_waiting_at_the_crane(queueing_script):
    """Siehe conftest: Gesamtzeiten 5.0 / 5.0 / 6.0, Mittel 16/3 (das Warten von Lkw 3 am Kran: 1.0)."""
    assert N.simulate_total(1.0, 0.0, 1.0, 3, 1, **queueing_script) == pytest.approx(16 / 3)


def test_simulation_by_hand_with_feedback():
    """Ein Lkw, alle Dauern 1, Ankunft bei 1; Routing-Zufall 0.0 (Gate → Kran), 0.0 (Kran → Stapel), 0.4 (Stapel → zurück zum Kran, da 0.4 < 0.5), 0.0 (Kran → Stapel), 0.9 (Stapel verlässt):
    Gate 1–2, Kran 2–3, Stapel 3–4, Kran 4–5, Stapel 5–6: Gesamtzeit 5.0."""
    p = [[0, 1, 0], [0, 0, 1], [0, 0.5, 0]]
    total = N.simulate_total(1.0, 0.5, 1.0, 1, 1, arrival=Seq([1.0]), services=[lambda: 1.0] * 3, rng_route=Route([0.0, 0.0, 0.4, 0.0, 0.9]), p=p)
    assert total == pytest.approx(5.0)


@pytest.mark.parametrize("scv", [0.25, 0.5, 1.0, 2.0, 4.0])
def test_samplers_have_the_requested_mean_and_variation(scv):
    draw = N.make_sampler(2.0, scv, N.SplitMix64(3))
    x = np.array([draw() for _ in range(200_000)])
    assert x.mean() == pytest.approx(2.0, rel=0.015) and x.var() / x.mean() ** 2 == pytest.approx(scv, abs=0.05 * max(scv, 0.2))


def test_fixed_service_has_no_variation():
    assert {N.make_sampler(3.0, 0, N.SplitMix64(1))() for _ in range(5)} == {3.0}


def test_same_seed_same_result_and_other_seed_differs():
    a, b, c = (N.simulate_total(0.4, 0.2, 1.0, 5_000, s) for s in (7, 7, 8))
    assert a == b and a != c


@pytest.mark.parametrize("u,r", [(0.5, 0.2), (0.7, 0.0)])
def test_simulation_matches_jackson_for_exponential_service(u, r):
    g = N.arrival_rate(u, r)
    assert N.simulate_total(g, r, 1.0, 120_000, 11) == pytest.approx(N.jackson_total(g, r), rel=0.04)


def test_simulation_with_strong_streuung_lies_above_jackson_and_near_the_decomposition():
    g, r = N.arrival_rate(0.7, 0.2), 0.2
    sim = N.simulate_total(g, r, 4.0, 150_000, 5)
    assert sim > 1.3 * N.jackson_total(g, r) and sim == pytest.approx(N.qna_total(g, r, 4.0), rel=0.12)
