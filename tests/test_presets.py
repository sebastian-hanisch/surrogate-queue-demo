"""Presets: Vollständigkeit, gültige Werte, Permalink-Konstanten, Formatierer."""

import pytest

import sur_constants as C
import sur_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_ORDER) and len(C.PRESETS) == 4
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS) and C.PRESET_HELP[name]


def test_preset_values_are_valid_and_match_the_setting_specs():
    for preset in C.PRESETS.values():
        assert C.N_MIN <= preset["n"] <= C.N_MAX and P.snap_n(preset["n"]) == preset["n"]
        assert C.R_PCT_MIN <= preset["r_pct"] <= C.R_PCT_MAX and P.snap_r(preset["r_pct"]) == preset["r_pct"]
        assert preset["cs2"] in C.CS2_OPTIONS and preset["design"] in C.DESIGNS
        for key, state_key in P.PRESET_KEYS.items():
            P.SETTING_SPECS[state_key].caster(preset[key])


def test_preset_names_state_the_values_they_set():
    assert C.PRESETS["Wenige Simulationen"]["n"] == C.N_MIN and C.PRESETS["Genug Simulationen"]["n"] == C.N_MAX
    assert C.PRESETS["Aktiv nachsampeln"]["design"] == "active" and C.PRESETS["Rückläufer und starke Streuung"]["r_pct"] == 40


def test_the_defaults_match_the_first_neutral_preset_values():
    p = C.PRESETS["Aktiv nachsampeln"]
    assert (p["n"], p["r_pct"], p["cs2"], p["seed"]) == (C.DEFAULT_N, C.DEFAULT_R_PCT, C.DEFAULT_CS2, C.DEFAULT_SEED) and C.DEFAULT_DESIGN == "sobol"


def test_bounds_and_url_params():
    assert P.bounds("seed_input") == (0, C.SEED_MAX) and P.bounds("n_slider") == (8, 64) and P.bounds("r_slider") == (0, 40)
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


@pytest.mark.parametrize("value,expected", [(0, 8), (9, 8), (11, 12), (25, 24), (27, 28), (99, 64)])
def test_n_snaps_to_the_step_from_the_lower_bound(value, expected):
    assert P.snap_n(value) == expected


@pytest.mark.parametrize("value,expected", [(0, 0), (4, 0), (6, 10), (26, 30), (99, 40)])
def test_feedback_share_snaps_to_the_step_inside_the_bounds(value, expected):
    assert P.snap_r(value) == expected


@pytest.mark.parametrize("value,expected", [(0.1, 0.25), (0.3, 0.25), (0.4, 0.5), (0.7, 0.5), (0.9, 1.0), (1.4, 1.0), (1.6, 2.0), (3.4, 4.0), (9.0, 4.0)])
def test_streuung_snaps_to_the_nearest_option(value, expected):
    assert P.snap_to_option("cs2_select", value) == expected


def test_formatters():
    assert C.fmt_int(150000) == "150.000" and C.fmt_pct(0.2) == "20 %" and C.fmt_pct(0.0898, 1) == "9.0 %"


def test_the_pool_and_study_constants_are_consistent():
    assert C.POOL_SIZE >= max(C.SIZES) and max(C.SIZES) >= C.N_MAX and C.EXTRAP_N <= C.POOL_SIZE and len(C.SLICE_U) == 14 and C.SLICE_U[0] == 0.3 and C.SLICE_U[-1] == 0.95
    assert set(C.CS2_OPTIONS) == {0.25, 0.5, 1.0, 2.0, 4.0}
