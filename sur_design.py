"""Versuchsplanung: welche Simulationen werden bezahlt? Drei Wege, Punkte aus einem Kandidatenpool (Sobol-Folge, jeder Punkt schon simuliert) auszuwählen: die ersten n in Sobol-Reihenfolge
(raumfüllend), zufällig, oder aktiv: ab einem zufälligen Start immer der Kandidat, an dem das GP am unsichersten ist."""

import numpy as np
from scipy.stats import qmc

import sur_models as M

DESIGNS = ("sobol", "random", "active")
DESIGN_LABELS = {"sobol": "raumfüllend (Sobol)", "random": "zufällig", "active": "aktiv (größte Unsicherheit)"}
ACTIVE_START = 8
REFIT_AT = (8, 12, 16, 24, 32, 48, 64, 96, 128)


def sobol_points(n, seed):
    """n Punkte des Entwurfsraums aus einer gescrambelten Sobol-Folge: (u, r, cs²). cs² ist eine der fünf Stufen (gleich oft)."""
    pts = qmc.Sobol(3, scramble=True, seed=seed).random(n)
    u = M.U_MIN + (M.U_MAX - M.U_MIN) * pts[:, 0]
    r = M.R_MIN + (M.R_MAX - M.R_MIN) * pts[:, 1]
    cs2 = np.array([M.CS2_LEVELS[min(int(p * len(M.CS2_LEVELS)), len(M.CS2_LEVELS) - 1)] for p in pts[:, 2]])
    return u, r, cs2


def select_sobol(n):
    """Die ersten n Pool-Punkte (der Pool steht in Sobol-Reihenfolge)."""
    return list(range(n))


def select_random(pool_size, n, seed):
    """n zufällige Pool-Punkte (fester Seed)."""
    return [int(i) for i in np.random.default_rng(seed).permutation(pool_size)[:n]]


def next_active(gp, x_pool, chosen):
    """Index des Kandidaten mit der größten Vorhersage-Standardabweichung des GP, der noch nicht gewählt ist."""
    _, sd = gp.predict(x_pool)
    sd = sd.copy()
    sd[list(chosen)] = -1.0
    return int(np.argmax(sd))


def select_active(x_pool, log_w_pool, n, seed, start=ACTIVE_START, refit_at=REFIT_AT):
    """Aktives Nachsampeln auf Logarithmen der Gesamtzeit: zufälliger Start mit `start` Punkten, dann je Schritt der unsicherste Kandidat. Die Hyperparameter werden nur bei den Größen in `refit_at`
    neu angepasst (dazwischen fest), das hält die Rechenzeit klein. Die Folge hängt nicht von n ab: die ersten m Punkte sind für jedes n ≥ m dieselben."""
    chosen = select_random(len(x_pool), start, seed)
    gp = M.gp_fit(x_pool[chosen], log_w_pool[chosen])
    theta = gp.theta
    while len(chosen) < n:
        if len(chosen) in refit_at and len(chosen) > start:
            gp = M.gp_fit(x_pool[chosen], log_w_pool[chosen], theta0=theta)
            theta = gp.theta
        else:
            gp = M.gp_build(x_pool[chosen], log_w_pool[chosen], theta)
        chosen.append(next_active(gp, x_pool, chosen))
    return chosen[:n]


def select(design, x_pool, log_w_pool, n, seed):
    """Index-Liste der gewählten Pool-Punkte für den Entwurf `design`."""
    if design == "sobol":
        return select_sobol(n)
    if design == "random":
        return select_random(len(x_pool), n, seed)
    if design == "active":
        return select_active(x_pool, log_w_pool, n, seed)
    raise ValueError(f"unbekannter Entwurf: {design}")
