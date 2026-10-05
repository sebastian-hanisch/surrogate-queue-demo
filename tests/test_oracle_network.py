"""Orakel-Tests des „teuren Simulators“ und der Näherungen (anderer Rechenweg als sur_network.py):
* Jackson: mittlere Zahl im System jeder Station aus der Geburts-Sterbe-Kette des M/M/c statt aus der Erlang-C-Formel;
* Zerlegung nach Whitt: die beiden Streuungsgleichungen des Kran-Stapel-Kreises direkt aufgelöst (affin in ca²) statt per Fixpunktiteration;
* Simulation: unabhängiger Simulator (Besuche in Ankunftsreihenfolge, Server-Frei-Zeiten, NumPy-Zufall) gegen die Netz-Simulation."""

import heapq
import math

import numpy as np
import pytest

import sur_network as N

SERV, MEAN = (3, 2, 4), (3.0, 2.4, 4.0)


def _mmc_mean_number(c, lam, m, size=1500):
    a = lam * m
    p = np.ones(size)
    for k in range(1, size):
        p[k] = p[k - 1] * a / min(k, c)
    p /= p.sum()
    return float((p * np.arange(size)).sum())


def jackson_reference(u, r):
    g = u * (1 - r) * 2 / 2.4                                       # Kran-Auslastung u: γ·2.4/2 /(1 − r) = u
    lam = [g, g / (1 - r), g / (1 - r)]                             # Gate γ; Kran und Stapel γ/(1 − r) (Rückläufer)
    return sum(_mmc_mean_number(SERV[i], lam[i], MEAN[i]) for i in range(3)) / g


def _erlang_c_probability(c, a):
    waiting = a ** c / math.factorial(c) / (1 - a / c)
    return waiting / (sum(a ** j / math.factorial(j) for j in range(c)) + waiting)


def qna_reference(u, r, cs2):
    g = u * (1 - r) * 2 / 2.4
    lam = [g, g / (1 - r), g / (1 - r)]
    rho = [lam[i] * MEAN[i] / SERV[i] for i in range(3)]
    slope = [1 - rho[i] ** 2 for i in range(3)]                     # cd² = A_i + B_i·ca²
    offset = [1 - slope[i] + rho[i] ** 2 * (cs2 - 1) / math.sqrt(SERV[i]) for i in range(3)]
    cd_gate = offset[0] + slope[0] * 1.0                            # externe Ankünfte sind Poisson (ca² = 1)
    # Kran: Zufluss = Gate-Abgänge (Rate γ, cd_gate) + Rückläufer (Rate r·λ_Stapel, c² = 1 + r(cd_Stapel − 1)); Stapel: ca² = cd_Kran
    c0 = g * cd_gate + r * (1 - r) * lam[2] + r ** 2 * lam[2] * (offset[2] + slope[2] * offset[1])
    c1 = r ** 2 * lam[2] * slope[2] * slope[1]
    ca_crane = c0 / (lam[1] - c1)
    ca = [1.0, ca_crane, offset[1] + slope[1] * ca_crane]
    total = 0.0
    for i in range(3):
        a = lam[i] * MEAN[i]
        wq = _erlang_c_probability(SERV[i], a) / (SERV[i] / MEAN[i] - lam[i]) * (ca[i] + cs2) / 2.0
        total += lam[i] * (wq + MEAN[i])
    return total / g


@pytest.mark.parametrize("u", [0.3, 0.6, 0.9, 0.95])
@pytest.mark.parametrize("r", [0.0, 0.2, 0.4])
def test_jackson_and_decomposition_match_the_independent_derivations(u, r):
    g = N.arrival_rate(u, r)
    assert N.jackson_total(g, r) == pytest.approx(jackson_reference(u, r), rel=1e-6)
    for cs2 in (0.25, 1.0, 4.0):
        assert N.qna_total(g, r, cs2) == pytest.approx(qna_reference(u, r, cs2), rel=1e-9)


def _sampler(mean, scv, rng, n):
    if scv == 1:
        return rng.exponential(mean, n)
    if scv < 1:
        k = round(1 / scv)
        return rng.gamma(k, mean / k, n)
    p = 0.5 * (1 + math.sqrt((scv - 1) / (scv + 1)))
    return np.where(rng.random(n) < p, rng.exponential(mean / (2 * p), n), rng.exponential(mean / (2 * (1 - p)), n))


def independent_simulation(u, r, cs2, trucks, seed):
    """Besuche in Ankunftsreihenfolge je Station; Beginn = max(Ankunft, früheste freie Zeit eines Geräts) ist genau FIFO mit mehreren Geräten."""
    rng = np.random.default_rng(seed)
    g = N.arrival_rate(u, r)
    entry = np.cumsum(rng.exponential(1 / g, trucks))
    free = [[0.0] * SERV[i] for i in range(3)]
    for f in free:
        heapq.heapify(f)
    draws = [_sampler(MEAN[i], cs2, rng, trucks * 3) for i in range(3)]
    ptr = [0, 0, 0]
    back = rng.random(trucks * 3)
    nb = 0
    events = [(entry[j], j, 0) for j in range(trucks)]
    heapq.heapify(events)
    sojourn = np.zeros(trucks)
    while events:
        t, j, st = heapq.heappop(events)
        start = max(t, heapq.heappop(free[st]))
        dep = start + draws[st][ptr[st]]
        ptr[st] += 1
        heapq.heappush(free[st], dep)
        if st < 2:
            heapq.heappush(events, (dep, j, st + 1))
        else:
            nb += 1
            if back[nb] < r:
                heapq.heappush(events, (dep, j, 1))
            else:
                sojourn[j] = dep - entry[j]
    return float(sojourn[int(0.05 * trucks):].mean())


@pytest.mark.parametrize("u,r,cs2", [(0.7, 0.2, 4.0), (0.6, 0.35, 0.25), (0.5, 0.1, 1.0)])
def test_network_simulation_matches_the_independent_simulator(u, r, cs2):
    trucks = 30_000
    mine = independent_simulation(u, r, cs2, trucks, 5)
    demo = N.simulate_total(N.arrival_rate(u, r), r, cs2, trucks, 5)
    assert demo == pytest.approx(mine, rel=0.08), (u, r, cs2, demo, mine)
