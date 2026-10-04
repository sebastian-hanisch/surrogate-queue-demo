"""Rechnet die teuren Teile vor (Build-Zeit, nicht in der App): `python generate_precomputed.py [Prozesse] [study]` schreibt `precomputed_sweep.json`.

  data   Simulationen des Netzes aus Stück 12: Kandidatenpool (600 Sobol-Punkte × 60 000 Lkw), Testpunkte als Wahrheit (100 Punkte × 2 unabhängige Läufe à 150 000 Lkw) und Schnitte durch die Fläche
         (5 Rückläufer-Anteile × 5 Streuungen × 14 Auslastungen × 2 Läufe à 150 000 Lkw)
  study  Lernkurven von GP, NN und Hybrid für drei Entwürfe (Sobol, zufällig, aktiv) und neun Größen mit je 5 Wiederholungen, Extrapolation (Training nur u ≤ 0.8) und Näherungen als Baseline

Mit dem Argument `study` werden nur die Studien neu gerechnet (die Simulationsdaten aus der vorhandenen Datei bleiben)."""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import sur_constants as C
import sur_design as D
import sur_evaluation as E
import sur_models as M
import sur_network as N


def _sim(args):
    u, r, cs2, trucks, seed = args
    return N.simulate_total(N.arrival_rate(u, r), r, cs2, trucks, seed)


def build_data(workers):
    pu, pr, pc = (a[:C.POOL_SIZE] for a in D.sobol_points(1024, C.POOL_SEED))
    tu, tr, tc = (a[:C.TEST_SIZE] for a in D.sobol_points(128, C.TEST_SEED))
    jobs = [(float(pu[i]), float(pr[i]), float(pc[i]), C.POOL_TRUCKS, 90_000 + i) for i in range(C.POOL_SIZE)]
    jobs += [(float(tu[i]), float(tr[i]), float(tc[i]), C.TEST_TRUCKS, 50_000 + 2 * i + k) for i in range(C.TEST_SIZE) for k in range(2)]
    cells = [(r, c) for r in C.SLICE_R for c in C.CS2_OPTIONS]
    jobs += [(u, r, c, C.SLICE_TRUCKS, 200_000 + 100 * ci + 2 * ui + k) for ci, (r, c) in enumerate(cells) for ui, u in enumerate(C.SLICE_U) for k in range(2)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(_sim, jobs, chunksize=4))
    pool_w = out[:C.POOL_SIZE]
    test_w = np.array(out[C.POOL_SIZE:C.POOL_SIZE + 2 * C.TEST_SIZE]).reshape(C.TEST_SIZE, 2)
    slice_w = np.array(out[C.POOL_SIZE + 2 * C.TEST_SIZE:]).reshape(len(cells), len(C.SLICE_U), 2).mean(axis=2)
    return {"pool": {"u": pu.tolist(), "r": pr.tolist(), "cs2": pc.tolist(), "w": pool_w},
            "test": {"u": tu.tolist(), "r": tr.tolist(), "cs2": tc.tolist(), "w1": test_w[:, 0].tolist(), "w2": test_w[:, 1].tolist()},
            "slices": [{"r": r, "cs2": c, "w": slice_w[i].tolist()} for i, (r, c) in enumerate(cells)]}


def _curve_task(args):
    design, seed, data = args
    ds = E.Dataset(data)
    return design, seed, E.run_curves_for_seed(ds, design, seed)


def _extrap_task(args):
    seed, data = args
    ds = E.Dataset(data)
    inside = [i for i in range(C.POOL_SIZE) if ds.pool_u[i] <= C.EXTRAP_U]
    pick = [inside[i] for i in np.random.default_rng(seed).permutation(len(inside))[:C.EXTRAP_N]]
    models = E.fit_models(ds, pick, nn_seed=seed)
    return E.evaluate(models, ds, mask=ds.test_u > C.EXTRAP_U)


def _mean_dicts(dicts):
    if isinstance(dicts[0], dict):
        return {k: _mean_dicts([d[k] for d in dicts]) for k in dicts[0]}
    return float(np.mean(dicts))


def build_study(data, workers):
    ds = E.Dataset(data)
    tasks = [("sobol", 0, data)] + [(d, s, data) for d in ("random", "active") for s in range(C.STUDY_REPS)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        curve_res = list(ex.map(_curve_task, tasks, chunksize=1))
        extrap_res = list(ex.map(_extrap_task, [(s, data) for s in range(C.STUDY_REPS)], chunksize=1))
    curves = {}
    for design in C.DESIGNS:
        runs = [r for d, s, r in curve_res if d == design]
        curves[design] = {str(n): _mean_dicts([run[n] for run in runs]) for n in C.SIZES}
        curves[design]["reps"] = len(runs)
    mask = ds.test_u > C.EXTRAP_U
    base = {"jackson": M.error_summary(ds.test_jackson, ds.test_w), "qna": M.error_summary(ds.test_qna, ds.test_w)}
    return {"sizes": list(C.SIZES), "reps": C.STUDY_REPS, "curves": curves, "baselines": base,
            "extrapolation": {"n_test": int(mask.sum()), "n_train": C.EXTRAP_N, "models": _mean_dicts(extrap_res)},
            "noise": {"median": float(np.median(ds.test_noise)), "p90": float(np.quantile(ds.test_noise, 0.9)), "max": float(ds.test_noise.max())}}


def main(workers, study_only):
    t0 = time.time()
    if study_only:
        data = json.loads(E.PRECOMPUTED_PATH.read_text(encoding="utf-8"))["data"]
    else:
        data = build_data(workers)
        print(f"Simulationen fertig nach {time.time() - t0:.0f} s", flush=True)
    study = build_study(data, workers)
    E.PRECOMPUTED_PATH.write_text(json.dumps({"data": data, "study": study}), encoding="utf-8")
    print(f"fertig in {time.time() - t0:.0f} s -> {E.PRECOMPUTED_PATH}")


if __name__ == "__main__":
    args = sys.argv[1:]
    workers = int(args[0]) if args and args[0].isdigit() else min(14, os.cpu_count() or 1)
    main(workers, "study" in args)
