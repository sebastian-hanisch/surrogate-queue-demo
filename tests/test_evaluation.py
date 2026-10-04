"""Auswertung: Datensatz, Modelle auf einer Auswahl, Fehler, Abdeckung, Schnitte, Vollständigkeit und Stimmigkeit der vorgerechneten Studie."""

import numpy as np
import pytest

import sur_constants as C
import sur_design as D
import sur_evaluation as E
import sur_models as M
import sur_network as N

PRE = E.load_precomputed()
STUDY = PRE["study"]
DS = E.dataset()


def test_dataset_has_the_expected_shape_and_follows_the_sobol_order():
    assert len(DS.pool_w) == C.POOL_SIZE and len(DS.test_w) == C.TEST_SIZE and DS.pool_x.shape == (C.POOL_SIZE, 3) and DS.test_x.shape == (C.TEST_SIZE, 3)
    u, r, cs2 = D.sobol_points(1024, C.POOL_SEED)
    assert (DS.pool_u == u[:C.POOL_SIZE]).all() and (DS.pool_r == r[:C.POOL_SIZE]).all() and (DS.pool_cs2 == cs2[:C.POOL_SIZE]).all()
    tu, _, _ = D.sobol_points(128, C.TEST_SEED)
    assert (DS.test_u == tu[:C.TEST_SIZE]).all()
    assert (DS.pool_w > 0).all() and (DS.test_w > 9).all() and (DS.pool_log_w == np.log(DS.pool_w)).all()


def test_the_test_truth_is_the_mean_of_two_runs_and_the_noise_is_small():
    t = PRE["data"]["test"]
    assert DS.test_w == pytest.approx(0.5 * (np.array(t["w1"]) + np.array(t["w2"])))
    assert np.median(DS.test_noise) < 0.01 and DS.test_noise.max() < 0.1


def test_physics_values_agree_with_the_network_module():
    j, q = E.physics(0.825, 0.2, 4.0)
    assert j == pytest.approx(19.68, abs=0.01) and q == pytest.approx(35.55, abs=0.01)
    assert DS.test_jackson[0] == pytest.approx(E.physics(DS.test_u[0], DS.test_r[0], DS.test_cs2[0])[0])


def test_models_trained_on_pool_points_run_and_the_summary_has_all_models():
    models = E.fit_models(DS, D.select_sobol(24))
    res = E.evaluate(models, DS)
    assert set(res) == {"gp", "nn", "hybrid", "jackson", "qna"} and all(set(v) == {"mean", "p90", "max"} for v in res.values())
    assert res["gp"]["mean"] < res["jackson"]["mean"] and res["hybrid"]["mean"] < res["jackson"]["mean"]


def test_evaluate_respects_a_mask():
    models = E.fit_models(DS, D.select_sobol(16))
    mask = DS.test_u > 0.8
    assert E.evaluate(models, DS, mask)["qna"] == M.error_summary(DS.test_qna[mask], DS.test_w[mask])


def test_coverage_returns_a_fraction_a_correlation_and_the_mean_sd():
    cov = E.coverage(E.fit_models(DS, D.select_sobol(64)), DS)
    assert 0.5 < cov["coverage"] <= 1.0 and -1 <= cov["corr"] <= 1 and cov["mean_sd"] > 0


def test_predict_gives_physics_times_residual_for_the_hybrid():
    models = E.fit_models(DS, D.select_sobol(16))
    pred = E.predict(models, DS.test_x, DS.test_qna)
    mean, _ = models["hybrid"].predict(DS.test_x)
    assert pred["hybrid"] == pytest.approx(DS.test_qna * np.exp(mean)) and pred["gp"].shape == (C.TEST_SIZE,)


def test_slice_curve_and_truth_lookup():
    models = E.fit_models(DS, D.select_sobol(24))
    u = np.linspace(0.3, 0.95, 5)
    cur = E.slice_curve(models, DS, 0.2, 4.0, u)
    assert {"gp", "nn", "hybrid", "gp_sd", "jackson", "qna"} <= set(cur) and cur["gp"].shape == (5,) and cur["jackson"][0] == pytest.approx(E.physics(0.3, 0.2, 4.0)[0])
    tu, tw = E.slice_truth(DS, 0.2, 4.0)
    assert list(tu) == list(C.SLICE_U) and len(tw) == 14 and np.all(np.diff(tw) > 0)                  # die Gesamtzeit steigt mit der Auslastung


def test_the_slice_truth_matches_the_formula_for_exponential_service_at_low_load():
    tu, tw = E.slice_truth(DS, 0.0, 1.0)
    assert tw[0] == pytest.approx(E.physics(float(tu[0]), 0.0, 1.0)[0], rel=0.03)


def test_the_study_covers_every_design_size_and_model():
    assert STUDY["sizes"] == list(C.SIZES) and STUDY["reps"] == C.STUDY_REPS
    for design in C.DESIGNS:
        for n in C.SIZES:
            cell = STUDY["curves"][design][str(n)]
            assert set(cell["errors"]) == {"gp", "nn", "hybrid", "jackson", "qna"} and set(cell["gp"]) == {"coverage", "corr", "mean_sd"}
    assert STUDY["curves"]["sobol"]["reps"] == 1 and STUDY["curves"]["random"]["reps"] == C.STUDY_REPS and STUDY["curves"]["active"]["reps"] == C.STUDY_REPS


def test_the_study_baselines_equal_the_errors_of_the_approximations():
    assert STUDY["baselines"]["jackson"] == pytest.approx(M.error_summary(DS.test_jackson, DS.test_w))
    assert STUDY["baselines"]["qna"] == pytest.approx(M.error_summary(DS.test_qna, DS.test_w))
    assert STUDY["noise"]["median"] == pytest.approx(float(np.median(DS.test_noise)))


def test_the_extrapolation_study_uses_only_the_points_above_the_training_range():
    ex = STUDY["extrapolation"]
    assert ex["n_test"] == int((DS.test_u > C.EXTRAP_U).sum()) and ex["n_train"] == C.EXTRAP_N and set(ex["models"]) == {"gp", "nn", "hybrid", "jackson", "qna"}
    mask = DS.test_u > C.EXTRAP_U
    assert ex["models"]["qna"] == pytest.approx(M.error_summary(DS.test_qna[mask], DS.test_w[mask]))


def test_a_live_run_reproduces_a_cell_of_the_study_for_the_deterministic_design():
    """Sobol, n = 32, gleiche Rechnung wie in der Studie: gleiche Fehler (innerhalb Rundung der Optimierung)."""
    run = E.run_curves_for_seed(DS, "sobol", 0, sizes=(32,))
    cell = STUDY["curves"]["sobol"]["32"]
    for key in ("gp", "hybrid"):
        assert run[32]["errors"][key]["mean"] == pytest.approx(cell["errors"][key]["mean"], rel=0.05)
    assert run[32]["errors"]["qna"]["mean"] == pytest.approx(cell["errors"]["qna"]["mean"])


def test_physics_array_helper_matches_the_pointwise_function():
    j, q = E.physics_arrays([0.5, 0.7], [0.1, 0.3], [1.0, 4.0])
    assert j[1] == pytest.approx(E.physics(0.7, 0.3, 4.0)[0]) and q[0] == pytest.approx(N.qna_total(N.arrival_rate(0.5, 0.1), 0.1, 1.0))
