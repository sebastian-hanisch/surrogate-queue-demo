"""Auswertung: Datensatz (Pool, Test, Schnitte) aus der vorgerechneten Datei, Modelle auf einer Auswahl trainieren, an den Testpunkten und entlang eines Schnitts auswerten, Lernkurven-Studie.
Die Studie wird beim Bau gerechnet (`generate_precomputed.py`) und hier nur gelesen."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

import sur_constants as C
import sur_design as D
import sur_models as M
import sur_network as N

PRECOMPUTED_PATH = Path(__file__).resolve().parent / "precomputed_sweep.json"


def physics(u, r, cs2):
    """Physikalische Näherungen an einem Punkt: (Jackson, Zerlegung nach Whitt) in Minuten."""
    g = N.arrival_rate(u, r)
    return N.jackson_total(g, r), N.qna_total(g, r, cs2)


def physics_arrays(u, r, cs2):
    pairs = [physics(a, b, c) for a, b, c in zip(u, r, cs2)]
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


@lru_cache(maxsize=1)
def load_precomputed():
    return json.loads(PRECOMPUTED_PATH.read_text(encoding="utf-8"))


class Dataset:
    """Pool (Training, in Sobol-Reihenfolge), Test (Wahrheit = Mittel zweier Läufe) und Näherungswerte."""

    def __init__(self, data):
        p, t = data["pool"], data["test"]
        self.pool_u, self.pool_r, self.pool_cs2, self.pool_w = (np.array(p[k], float) for k in ("u", "r", "cs2", "w"))
        self.test_u, self.test_r, self.test_cs2 = (np.array(t[k], float) for k in ("u", "r", "cs2"))
        self.test_w = 0.5 * (np.array(t["w1"], float) + np.array(t["w2"], float))
        self.test_noise = np.abs(np.log(np.array(t["w1"], float)) - np.log(np.array(t["w2"], float))) / np.sqrt(2.0)
        self.pool_x = M.features(self.pool_u, self.pool_r, self.pool_cs2)
        self.test_x = M.features(self.test_u, self.test_r, self.test_cs2)
        self.pool_log_w = np.log(self.pool_w)
        self.pool_jackson, self.pool_qna = physics_arrays(self.pool_u, self.pool_r, self.pool_cs2)
        self.test_jackson, self.test_qna = physics_arrays(self.test_u, self.test_r, self.test_cs2)
        self.slices = data["slices"]


@lru_cache(maxsize=1)
def dataset():
    return Dataset(load_precomputed()["data"])


def fit_models(ds, idx, nn_seed=0):
    """Trainiert GP, NN und Hybrid auf den Pool-Punkten `idx` (alle auf ln W)."""
    idx = list(idx)
    x, w = ds.pool_x[idx], ds.pool_w[idx]
    return {"gp": M.fit_log_gp(x, w), "nn": M.nn_fit(x, np.log(w), seed=nn_seed), "hybrid": M.fit_hybrid(x, w, ds.pool_qna[idx])}


def predict(models, x, qna):
    """Vorhersagen in Minuten und die log-Standardabweichung des GP / Hybrids an den (skalierten) Punkten x; `qna` = Zerlegungswerte dort."""
    gm, gs = models["gp"].predict(x)
    hm, hs = models["hybrid"].predict(x)
    return {"gp": np.exp(gm), "nn": np.exp(models["nn"].predict(x)), "hybrid": qna * np.exp(hm), "gp_sd": gs, "hybrid_sd": hs}


def coverage(models, ds, level=1.96):
    """Anteil der Testpunkte, deren (logarithmierte) Wahrheit im GP-Intervall ± level·Standardabweichung liegt, und die Korrelation zwischen Standardabweichung und Fehler."""
    mean, sd = models["gp"].predict(ds.test_x)
    err = np.abs(np.log(ds.test_w) - mean)
    return {"coverage": float(np.mean(err < level * sd)), "corr": float(np.corrcoef(sd, err)[0, 1]), "mean_sd": float(sd.mean())}


def evaluate(models, ds, mask=None):
    """Fehler aller Modelle an den Testpunkten (optional nur an `mask`): mittlerer / 90-%- / größter relativer Fehler in W."""
    mask = np.ones(len(ds.test_w), bool) if mask is None else mask
    pred = predict(models, ds.test_x, ds.test_qna)
    out = {k: M.error_summary(pred[k][mask], ds.test_w[mask]) for k in ("gp", "nn", "hybrid")}
    out["jackson"] = M.error_summary(ds.test_jackson[mask], ds.test_w[mask])
    out["qna"] = M.error_summary(ds.test_qna[mask], ds.test_w[mask])
    return out


def slice_curve(models, ds, r, cs2, u_grid):
    """Vorhersagen entlang u bei festem r und cs²: Jackson, Zerlegung, GP (mit Band), NN und Hybrid, in Minuten."""
    u = np.asarray(u_grid, float)
    x = M.features(u, np.full(len(u), r), np.full(len(u), cs2))
    jq = np.array([physics(a, r, cs2) for a in u])
    pred = predict(models, x, jq[:, 1])
    pred.update({"jackson": jq[:, 0], "qna": jq[:, 1]})
    return pred


def slice_truth(ds, r, cs2):
    """Wahrheit auf dem Schnitt (u-Gitter und Mittel zweier Läufe) aus der vorgerechneten Datei."""
    cell = next(x for x in ds.slices if abs(x["r"] - r) < 1e-9 and x["cs2"] == cs2)
    return np.array(C.SLICE_U), np.array(cell["w"], float)


def study_row(study, design, n, model):
    """Eine Zelle der Lernkurven-Studie: Mittel über die Wiederholungen (mean, p90, max)."""
    return study["curves"][design][str(n)][model]


def run_curves_for_seed(ds, design, seed, sizes=C.SIZES):
    """Lernkurve eines Entwurfs für einen Seed: Fehler und GP-Güte je Größe (Studie, Build-Zeit)."""
    idx_all = D.select(design, ds.pool_x, ds.pool_log_w, max(sizes), seed)
    out = {}
    for n in sizes:
        models = fit_models(ds, idx_all[:n], nn_seed=seed)
        out[n] = {"errors": evaluate(models, ds), "gp": coverage(models, ds)}
    return out
