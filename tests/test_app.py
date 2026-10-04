"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Randwerte, jeder Entwurf, lokale Regler, Würfel-Knopf, Permalink-Grenzen, Abschnitte, Footer."""

import random
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import sur_constants as C
import sur_visualization as V

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=900)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def _table(at, start):
    return next(m.value for m in at.markdown if m.value.startswith(start))


def test_default_run_has_no_exception_and_shows_the_reference_values():
    at = _run()
    _ok(at)
    assert _metric(at, V.LABELS["jackson"]) == "10.9 %" and _metric(at, V.LABELS["qna"]) == "2.9 %"                  # Näherungen: feste Werte über die 100 Testpunkte
    for key in ("gp", "nn", "hybrid"):
        assert float(_metric(at, V.LABELS[key]).split()[0]) < 10.9
    assert len(at.get("plotly_chart")) == 7


def test_all_sections_are_present():
    at = _run()
    assert [s.value for s in at.subheader] == ["📐 Lernkurven: wie viele Simulationen sind genug?", "🔬 Welche Punkte simulieren?", "🔬 Wie sicher ist sich das GP?",
                                              "🔬 Extrapolation: jenseits des Trainings", "🔬 Was kostet das?", "🚧 Wo die Annahmen enden"]
    assert any(m.value.startswith("## 🔗 Von der Simulation zur Fläche") for m in at.markdown)
    assert any("Diese Demo ist Teil des Portfolios" in c.value for c in at.caption)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert (at.session_state["n_slider"], at.session_state["design_select"], at.session_state["r_slider"], at.session_state["cs2_select"]) == (p["n"], p["design"], p["r_pct"], p["cs2"])
    assert at.metric


@pytest.mark.parametrize("kw", [dict(n_slider=8, design_select="sobol"), dict(n_slider=64, design_select="random"), dict(n_slider=64, design_select="active"),
                                 dict(n_slider=8, design_select="active", r_slider=0, cs2_select=0.25), dict(n_slider=40, r_slider=40, cs2_select=4.0),
                                 dict(n_slider=16, r_slider=0, cs2_select=1.0)])
def test_extreme_settings_and_every_design_run(kw):
    _ok(_run(**kw))


def test_the_learning_table_has_one_row_per_size_and_the_metric_regulator_changes_it():
    at = _run()
    table = _table(at, "| Simulationen n | Gauß-Prozess")
    assert table.count("\n| ") == len(C.SIZES)
    other = _run(metric_select="max")
    _ok(other)
    assert _table(other, "| Simulationen n | Gauß-Prozess") != table


def test_the_info_names_the_first_size_where_the_models_beat_the_decomposition():
    at = _run(design_select="random")
    text = next(i.value for i in at.info if "Das reine GP unterbietet sie ab" in i.value)
    assert "Zerlegung hat ohne jede Simulation 2.9 %" in text and "ab n = 32" in text


def test_the_design_comparison_regulator_runs_for_every_model():
    for model in ("gp", "nn", "hybrid"):
        at = _run(design_model_select=model)
        _ok(at)
        assert len(at.get("plotly_chart")) == 7


def test_coverage_table_and_extrapolation_table_are_complete():
    at = _run()
    assert _table(at, "| Simulationen n | Abdeckung").count("\n| ") == len(C.SIZES)
    ex = _table(at, "| Modell | mittlerer Fehler | größter Fehler |")
    assert ex.count("\n| ") == 5 and "| Hybrid (Zerlegung + GP) | 4.0 %" in ex


def test_the_cost_section_reports_measured_times():
    at = _run()
    assert _metric(at, "Eine Simulation (60 000 Lkw)").endswith(" s") and _metric(at, "GP: 100 Vorhersagen").endswith(" ms")
    assert any("Fragen an das Netz kosten per Simulation" in i.value for i in at.info)


def test_dice_button_changes_the_seed_and_the_random_design(monkeypatch):
    """Der Würfel zieht sonst einen unseeded Zufalls-Seed; deshalb ist der gewürfelte Seed im Test fest."""
    monkeypatch.setattr(random, "randint", lambda a, b: 508145)
    at = _run(design_select="random")
    old_seed = at.session_state["seed_input"]
    old = _metric(at, V.LABELS["gp"])
    next(b for b in at.button if b.label == "🎲 Neuen Lauf würfeln").click().run()
    _ok(at)
    assert at.session_state["seed_input"] == 508145 != old_seed and _metric(at, V.LABELS["gp"]) != old


def test_permalink_values_are_clamped_and_snapped():
    at = AppTest.from_file(APP, default_timeout=900)
    at.query_params["n"] = "27"
    at.query_params["r"] = "26"
    at.query_params["cs2"] = "3.4"
    at.query_params["design"] = "random"
    at.run()
    _ok(at)
    assert at.session_state["n_slider"] == 28 and at.session_state["r_slider"] == 30 and at.session_state["cs2_select"] == 4.0 and at.session_state["design_select"] == "random"


def test_permalink_ignores_garbage_and_unknown_designs():
    at = AppTest.from_file(APP, default_timeout=900)
    at.query_params["n"] = "viele"
    at.query_params["design"] = "gitter"
    at.query_params["seed"] = "nan"
    at.run()
    _ok(at)
    assert at.session_state["n_slider"] == C.DEFAULT_N and at.session_state["design_select"] == C.DEFAULT_DESIGN and at.session_state["seed_input"] == C.DEFAULT_SEED


def test_no_sentence_wide_comma_replacement_in_the_app_source():
    """Regressionsschutz: `.replace(",", ".")` auf einem ganzen (verketteten) Satz macht aus Kommas im Fließtext Punkte; Tausender nur über `fmt_int`."""
    source = Path(APP).read_text(encoding="utf-8")
    assert '.replace(",", ".")' not in source
