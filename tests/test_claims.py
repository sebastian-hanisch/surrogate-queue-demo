"""JEDE Zahl aus README und App-Texten wird hier nachgerechnet: Studien-Zahlen aus der vorgerechneten Datei (Mittel über Wiederholungen; 600 Pool-Punkte à 60 000 Lkw, 100 Testpunkte à 2 × 150 000 Lkw),
Näherungen exakt aus den Formeln. Die Datei ist fest; ändern sich die Zahlen nach einer neuen Rechnung, müssen README und Hilfetexte nachgezogen werden. Fehler = relativer Fehler der Gesamtzeit."""

import numpy as np
import pytest

import sur_constants as C
import sur_evaluation as E
import sur_network as N

PRE = E.load_precomputed()
S = PRE["study"]
DS = E.dataset()
CURVES = S["curves"]


def mean_err(design, n, model):
    return CURVES[design][str(n)]["errors"][model]["mean"]


def pct(x, digits=1):
    return round(100 * x, digits)


def test_the_test_truth_is_precise():
    """README: Rauschen der Wahrheit je Lauf (log) im Median 0.4 %, 90-%-Quantil 2.2 %, größter Wert 7.8 %; Testpunkte mit u > 0.8: 23; Pool-Punkte mit u ≤ 0.8: 461."""
    assert pct(S["noise"]["median"]) == 0.4 and pct(S["noise"]["p90"]) == 2.2 and pct(S["noise"]["max"]) == 7.8
    assert S["extrapolation"]["n_test"] == 23 and int((DS.pool_u <= C.EXTRAP_U).sum()) == 461


def test_the_baselines():
    """README: Jackson ignoriert cs² und liegt im Mittel 10.9 % (90-%-Quantil 28.6 %, größter 49.3 %) neben der Wahrheit, die Zerlegung 2.9 % (7.1 %, 19.5 %); Testbereich 10 … 70 min."""
    b = S["baselines"]
    assert (pct(b["jackson"]["mean"]), pct(b["jackson"]["p90"]), pct(b["jackson"]["max"])) == (10.9, 28.6, 49.3)
    assert (pct(b["qna"]["mean"]), pct(b["qna"]["p90"]), pct(b["qna"]["max"])) == (2.9, 7.1, 19.5)
    assert round(DS.test_w.min()) == 10 and round(DS.test_w.max()) == 70


def test_the_decomposition_is_worst_for_strong_streuung_at_high_load():
    """README: Die Zerlegung liegt am schlechtesten bei cs² = 4 und hoher Auslastung (u > 0.8: bis 19 %)."""
    e = np.abs(DS.test_qna / DS.test_w - 1)
    worst = np.argsort(-e)[:3]
    assert all(DS.test_cs2[i] == 4.0 and DS.test_u[i] > 0.8 for i in worst) and round(100 * e[worst[0]]) == 19


def test_the_slice_at_the_corner_of_the_design_space():
    """PRESET_HELP: 40 % Rückläufer, cs² = 4, u = 0.95: Simulation 105 min, Jackson 56, Zerlegung 133 (die Zerlegung liegt dort +26 % darüber)."""
    _, tw = E.slice_truth(DS, 0.4, 4.0)
    j, q = E.physics(0.95, 0.4, 4.0)
    assert round(tw[-1]) == 105 and round(j) == 56 and round(q) == 133 and round(100 * (q / tw[-1] - 1)) == 26


def test_learning_curves_for_the_three_designs():
    """README: GP mit raumfüllender Auswahl 4.2 / 2.4 / 1.7 / 1.5 / 1.0 / 1.2 / 1.1 / 1.0 / 1.3 % bei n = 8 … 128; NN 2.9 / 2.8 / 2.6 / 2.8 / 2.4 / 2.2 / 2.0 / 1.8 / 1.9; Hybrid 1.5 / 1.4 / 1.6 / 1.6 / 1.1 / 1.1 / 1.5 / 1.3 / 1.2."""
    sizes = C.SIZES
    assert [pct(mean_err("sobol", n, "gp")) for n in sizes] == [4.2, 2.4, 1.7, 1.5, 1.0, 1.2, 1.1, 1.0, 1.3]
    assert [pct(mean_err("sobol", n, "nn")) for n in sizes] == [2.9, 2.8, 2.6, 2.8, 2.4, 2.2, 2.0, 1.8, 1.9]
    assert [pct(mean_err("sobol", n, "hybrid")) for n in sizes] == [1.5, 1.4, 1.6, 1.6, 1.1, 1.1, 1.5, 1.3, 1.2]


def test_when_a_surrogate_beats_the_decomposition():
    """README: Erste Größe, bei der der mittlere Fehler unter der Zerlegung (2.9 %) liegt: GP und NN raumfüllend n = 12, aktiv 24, zufällig 32; das Hybrid schon bei n = 8 (alle Entwürfe)."""
    q = S["baselines"]["qna"]["mean"]

    def first(design, model):
        return next((n for n in C.SIZES if mean_err(design, n, model) < q), None)
    assert [first("sobol", m) for m in ("gp", "nn", "hybrid")] == [12, 12, 8]
    assert [first("active", m) for m in ("gp", "nn", "hybrid")] == [24, 24, 8]
    assert [first("random", m) for m in ("gp", "nn", "hybrid")] == [32, 32, 8]
    assert all(mean_err(d, 8, "hybrid") < S["baselines"]["jackson"]["mean"] / 4 for d in C.DESIGNS)


def test_the_design_matters_most_for_few_simulations():
    """README: GP bei n = 8: raumfüllend 4.2 %, zufällig und aktiv (gleicher zufälliger Start) 12.6 %; bei n = 32: 1.0 / 2.1 / 1.6; n = 64: 1.1 / 1.4 / 1.0; n = 128: 1.3 / 1.1 / 0.9 %."""
    got = [[pct(mean_err(d, n, "gp")) for d in C.DESIGNS] for n in (8, 32, 64, 128)]
    assert got == [[4.2, 12.6, 12.6], [1.0, 2.1, 1.6], [1.1, 1.4, 1.0], [1.3, 1.1, 0.9]]


def test_active_sampling_lowers_the_worst_case():
    """README: größter Fehler bei n = 64: GP raumfüllend 12 %, zufällig 15 %, aktiv 6 %; NN 23 / 23 / 7; Hybrid 14 / 15 / 11."""
    got = {d: [round(100 * CURVES[d]["64"]["errors"][m]["max"]) for m in ("gp", "nn", "hybrid")] for d in C.DESIGNS}
    assert got == {"sobol": [12, 23, 14], "random": [15, 23, 15], "active": [6, 7, 11]}


def test_gp_intervals_are_not_calibrated():
    """README: Abdeckung des nominalen 95-%-Intervalls (n = 8 … 128): raumfüllend 98 / 92 / 86 / 81 / 71 / 82 / 86 / 91 / 87 %; zufällig 64 … 94 % (n ≥ 32: 92 … 94 %); aktiv 64 … 100 %."""
    cov = {d: [round(100 * CURVES[d][str(n)]["gp"]["coverage"]) for n in C.SIZES] for d in C.DESIGNS}
    assert cov["sobol"] == [98, 92, 86, 81, 71, 82, 86, 91, 87]
    assert cov["random"] == [64, 80, 90, 87, 92, 92, 93, 94, 93] and cov["active"] == [64, 77, 80, 89, 97, 95, 100, 99, 100]


def test_the_gp_deviation_tracks_the_error_except_after_active_sampling():
    """README: Korrelation von GP-Standardabweichung und Fehler 0.4 … 0.8 bei raumfüllender und zufälliger Auswahl; nach aktivem Nachsampeln fällt sie auf 0.04 (n = 128)."""
    for d in ("sobol", "random"):
        corr = [CURVES[d][str(n)]["gp"]["corr"] for n in C.SIZES]
        assert round(min(corr), 1) >= 0.4 and round(max(corr), 1) <= 0.8
    assert round(CURVES["active"]["128"]["gp"]["corr"], 2) == 0.04 and CURVES["active"]["8"]["gp"]["corr"] > CURVES["active"]["128"]["gp"]["corr"]


def test_extrapolation_numbers():
    """README: Training nur u ≤ 0.8 (64 Punkte), Test u > 0.8 (23 Punkte): Fehler Jackson 26.4 %, Zerlegung 6.9 %, GP 13.0 % (größter 38 %), NN 11.0 % (43 %), Hybrid 4.0 % (13 %)."""
    m = S["extrapolation"]["models"]
    assert [pct(m[k]["mean"]) for k in ("jackson", "qna", "gp", "nn", "hybrid")] == [26.4, 6.9, 13.0, 11.0, 4.0]
    assert [round(100 * m[k]["max"]) for k in ("gp", "nn", "hybrid")] == [38, 43, 13]
    assert m["hybrid"]["mean"] < m["qna"]["mean"] < m["nn"]["mean"] < m["gp"]["mean"]


def test_cost_section_numbers_in_the_readme_are_orders_of_magnitude_only():
    """README: eine Simulation mit 60 000 Lkw rund eine Sekunde, die GP-Vorhersage für 100 Punkte Bruchteile einer Millisekunde bis wenige Millisekunden."""
    import time
    t = time.perf_counter()
    N.simulate_total(N.arrival_rate(0.8, 0.2), 0.2, 1.0, 20_000, 1)
    sim = (time.perf_counter() - t) * 3
    assert 0.2 < sim < 15
    models = E.fit_models(DS, list(range(32)))
    t = time.perf_counter()
    for _ in range(20):
        models["gp"].predict(DS.test_x)
    assert (time.perf_counter() - t) / 20 < 0.05


def test_app_preset_help_quotes_the_study():
    """PRESET_HELP: 8 raumfüllend GP 4.2 / NN 2.9 / Hybrid 1.5 (Zerlegung 2.9, Jackson 10.9); 64: 1.1 / 2.0 / 1.5; 24 aktiv: 2.3 / 2.6 / 2.8, raumfüllend mit 24 GP 1.5."""
    assert [pct(mean_err("sobol", 8, k)) for k in ("gp", "nn", "hybrid", "qna", "jackson")] == [4.2, 2.9, 1.5, 2.9, 10.9]
    assert [pct(mean_err("sobol", 64, k)) for k in ("gp", "nn", "hybrid")] == [1.1, 2.0, 1.5]
    assert [pct(mean_err("active", 24, k)) for k in ("gp", "nn", "hybrid")] == [2.3, 2.6, 2.8] and pct(mean_err("sobol", 24, "gp")) == 1.5
    for name, text in C.PRESET_HELP.items():
        assert text and name in C.PRESETS
