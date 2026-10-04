"""Abbildungen: gesperrte Achsen, Zahl der Linien, Beschriftungen."""

import numpy as np

import sur_constants as C
import sur_design as D
import sur_evaluation as E
import sur_visualization as V

PRE = E.load_precomputed()
STUDY = PRE["study"]
DS = E.dataset()


def _locked(fig):
    return all(ax.fixedrange for ax in fig.select_xaxes()) and all(ax.fixedrange for ax in fig.select_yaxes())


def _setup():
    idx = D.select_sobol(16)
    models = E.fit_models(DS, idx)
    u = np.linspace(0.3, 0.95, 20)
    curves = E.slice_curve(models, DS, 0.2, 4.0, u)
    tu, tw = E.slice_truth(DS, 0.2, 4.0)
    preds = E.predict(models, DS.test_x, DS.test_qna)
    preds["qna"] = DS.test_qna
    return idx, u, curves, tu, tw, preds


def test_all_charts_lock_their_axes():
    idx, u, curves, tu, tw, preds = _setup()
    figs = [V.build_design_chart(DS.pool_u[idx], DS.pool_r[idx], DS.pool_cs2[idx]), V.build_slice_chart(u, curves, tu, tw, 20, 4.0), V.build_parity_chart(DS.test_w, preds),
            V.build_learning_chart(STUDY, "sobol", "mean"), V.build_designs_chart(STUDY, "gp"), V.build_coverage_chart(STUDY, "random"), V.build_extrapolation_chart(STUDY)]
    assert all(_locked(f) for f in figs)


def test_design_chart_has_one_trace_per_streuung_level():
    idx = D.select_sobol(40)
    fig = V.build_design_chart(DS.pool_u[idx], DS.pool_r[idx], DS.pool_cs2[idx])
    assert [t.name for t in fig.data] == [f"cs² = {c:g}" for c in sorted(V.CS2_COLORS)]
    assert sum(len(t.x) for t in fig.data) == 40


def test_slice_chart_layers():
    _, u, curves, tu, tw, _ = _setup()
    fig = V.build_slice_chart(u, curves, tu, tw, 20, 4.0)
    assert [t.name for t in fig.data] == ["GP 95-%-Band", V.LABELS["jackson"], V.LABELS["qna"], V.LABELS["gp"], V.LABELS["nn"], V.LABELS["hybrid"], "Simulation (Wahrheit)"]
    assert fig.layout.yaxis.type == "log" and "100" in fig.layout.yaxis.ticktext


def test_parity_chart_has_the_diagonal_and_one_cloud_per_model():
    *_, preds = _setup()
    fig = V.build_parity_chart(DS.test_w, preds)
    assert [t.name for t in fig.data] == ["perfekt", V.LABELS["gp"], V.LABELS["nn"], V.LABELS["hybrid"], V.LABELS["qna"]] and len(fig.data[1].x) == C.TEST_SIZE


def test_learning_chart_shows_three_models_and_two_baselines_for_every_metric():
    for metric in V.METRICS:
        fig = V.build_learning_chart(STUDY, "random", metric)
        assert [t.name for t in fig.data] == [V.LABELS[k] for k in ("gp", "nn", "hybrid", "jackson", "qna")]
        assert list(fig.data[0].y) == [STUDY["curves"]["random"][str(n)]["errors"]["gp"][metric] for n in C.SIZES]
        assert fig.data[3].y[0] == STUDY["baselines"]["jackson"][metric]


def test_designs_chart_has_one_line_per_design():
    fig = V.build_designs_chart(STUDY, "hybrid", "p90")
    assert len(fig.data) == 3 and list(fig.data[1].y) == [STUDY["curves"]["random"][str(n)]["errors"]["hybrid"]["p90"] for n in C.SIZES]


def test_coverage_chart_marks_the_nominal_level():
    fig = V.build_coverage_chart(STUDY, "sobol")
    assert any(abs(s.y0 - 0.95) < 1e-12 for s in fig.layout.shapes) and fig.layout.yaxis.range == (0, 1.05)


def test_extrapolation_chart_has_five_bars():
    fig = V.build_extrapolation_chart(STUDY)
    assert list(fig.data[0].x) == [V.LABELS[k] for k in ("jackson", "qna", "gp", "nn", "hybrid")] and len(fig.data[0].y) == 5
