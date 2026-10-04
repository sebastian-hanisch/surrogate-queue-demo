"""Surrogat-Modelle – Gauß-Prozess und neuronales Netz als Ersatz für die Simulation - Stück 13 der Konzepte-Linie "Warteschlangentheorie und Simulation"
Sebastian Hanisch - Operations Research und Machine Learning

Das Terminal-Netz aus Stück 12 ist hier der „teure Simulator“. Ein Gauß-Prozess und ein kleines neuronales Netz lernen aus wenigen bezahlten Simulationen die Gesamtzeit eines Lkw über der Auslastung des
Krans, dem Rückläuferanteil und der Streuung der Dauer. Verglichen werden sie mit der Jackson-Formel und der Zerlegungsnäherung, über Lernkurven, drei Arten der Punktauswahl, die Unsicherheit des GP und
die Extrapolation. Siehe README.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import sur_constants as C
import sur_design as D
import sur_evaluation as E
import sur_network as N
from sur_presets import (apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params)
from sur_visualization import (LABELS, METRICS, build_coverage_chart, build_design_chart, build_designs_chart, build_extrapolation_chart, build_learning_chart, build_parity_chart, build_slice_chart)

st.set_page_config(page_title="Surrogat-Modelle – Sebastian Hanisch", layout="wide")

PRE = E.load_precomputed()
STUDY = PRE["study"]
DS = E.dataset()
U_GRID = np.linspace(0.3, 0.95, 66)


@st.cache_data(show_spinner=False)
def _fit(n, design, seed):
    idx = D.select(design, DS.pool_x, DS.pool_log_w, n, seed)
    return idx, E.fit_models(DS, idx, nn_seed=seed % 1000)


@st.cache_data(show_spinner=False)
def _timing():
    """Zeit einer Simulation (20 000 Lkw) und einer GP-Vorhersage für 100 Punkte, auf diesem Rechner gemessen."""
    t = time.perf_counter()
    N.simulate_total(N.arrival_rate(0.8, 0.2), 0.2, 1.0, 20_000, 1)
    sim = time.perf_counter() - t
    models = E.fit_models(DS, list(range(32)))
    t = time.perf_counter()
    for _ in range(20):
        models["gp"].predict(DS.test_x)
    pred = (time.perf_counter() - t) / 20
    return sim, pred


def _first_n_below(design, model, level):
    """Kleinste Größe der Studie, bei der der mittlere Fehler des Modells unter `level` liegt (None, wenn nie)."""
    for n in STUDY["sizes"]:
        if STUDY["curves"][design][str(n)]["errors"][model]["mean"] < level:
            return n
    return None


def _pct(x, digits=1):
    return C.fmt_pct(x, digits)


st.title("🧪 Surrogat-Modelle – ein Ersatz für die Simulation")
st.markdown(
    """
Jede Frage an das Terminal-Netz aus Stück 12 kostet eine **Simulation**: rund eine Sekunde für 60 000 Lkw, und für ein Prozent Genauigkeit viel mehr. Für Entwurfsfragen („wie viel Last verträgt der Kran bei starker Streuung?“) braucht
man tausende Antworten. Ein **Surrogat** lernt aus wenigen bezahlten Simulationen die Antwort als Funktion der Eingaben und gibt sie danach in Mikrosekunden. Hier konkurrieren ein **Gauß-Prozess (GP)** und ein **neuronales Netz (NN)**,
dazu ein **Hybrid**, das nur den Rest zur Zerlegungsnäherung lernt, gegen die **Formeln** aus den Stücken davor. Die ehrliche Frage lautet: Wann lohnt sich das, und wann ist die Formel schon gut genug?
"""
)
st.caption(
    "Stück 13 der Linie „Warteschlangentheorie und Simulation“ (das letzte), baut auf [jackson-network-demo](https://sebastianhanisch-jackson-network-demo.streamlit.app/) (Stück 12: das Netz als Simulator) auf; "
    "das neuronale Netz folgt [mlp-backprop-demo](https://sebastianhanisch-mlp-backprop-demo.streamlit.app/), die Intervalle sind ein Fall für [forecast-interval-demo](https://sebastianhanisch-forecast-interval-demo.streamlit.app/)."
)

with st.expander("So wird aus der Simulation ein Ersatzmodell", expanded=True):
    st.markdown(
        """
- **Eingaben (3):** Auslastung des Krans u (0.3 bis 0.95), Anteil der Rückläufer r (0 bis 40 %), Streuung der Dauer cs² (0.25 bis 4). **Ausgabe:** mittlere Gesamtzeit eines Lkw. Gelernt wird ihr Logarithmus,
  denn sie wächst wie 1/(1 − u).
- **Gauß-Prozess:** Mittelwert und Unsicherheit aus einem Kern (Matérn-5/2, ein Längenmaß je Eingabe) und Rauschen; die Hyperparameter maximieren die Randwahrscheinlichkeit. Er sagt auch, **wo er sich unsicher ist**.
- **Neuronales Netz:** 3 → 32 → 32 → 1 mit tanh, Rückwärtsableitung von Hand, L-BFGS. Ein Wert je Punkt, keine Unsicherheit.
- **Hybrid:** das GP lernt nur den Rest zwischen Simulation und Zerlegung nach Whitt (Stück 12). Wo die Physik stimmt, bleibt wenig zu lernen.
- **Bezahlt** wird jede Simulation mit 60 000 Lkw (billig, verrauscht); die **Wahrheit** an 100 Testpunkten stammt aus je zwei Läufen mit 150 000 Lkw.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESET_ORDER))
for i, name in enumerate(C.PRESET_ORDER):
    with preset_cols[i]:
        st.button(name, key=f"preset_{name}", width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name] or None)

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n = st.slider("Bezahlte Simulationen n", *bounds("n_slider"), step=C.N_STEP, key="n_slider", help="Wie viele Punkte aus dem Pool simuliert (und dem Modell gezeigt) werden.")
    design = st.selectbox("Auswahl der Punkte", C.DESIGNS, key="design_select", format_func=lambda d: D.DESIGN_LABELS[d],
                          help="Raumfüllend: die ersten n der Sobol-Folge; zufällig; aktiv: immer dort simulieren, wo das GP am unsichersten ist.")
    r_pct = st.slider("Schnitt: Rückläufer", *bounds("r_slider"), step=C.R_PCT_STEP, key="r_slider", format="%d %%", help="Nur für die Schnittdarstellung: Rückläuferanteil.")
    cs2 = st.select_slider("Schnitt: Streuung der Dauer (cs²)", options=C.CS2_OPTIONS, key="cs2_select", format_func=lambda v: f"{v:g}", help="Nur für die Schnittdarstellung: cs² an allen Stationen.")
    seed = st.number_input("Zufalls-Seed", min_value=bounds("seed_input")[0], max_value=bounds("seed_input")[1], step=1, key="seed_input", help="Bestimmt den zufälligen Start bzw. die zufällige Auswahl und die Startgewichte des NN.")
    st.button("🎲 Neuen Lauf würfeln", on_click=randomize_seed)

n, r_pct, cs2, seed = int(n), int(r_pct), float(cs2), int(seed)
sync_query_params({"n_slider": n, "design_select": design, "r_slider": r_pct, "cs2_select": cs2, "seed_input": seed})

with st.spinner("Wähle Punkte und trainiere GP, NN und Hybrid …" + (" (aktives Nachsampeln braucht einige Sekunden)" if design == "active" else "")):
    idx, models = _fit(n, design, seed)
res = E.evaluate(models, DS)
cov = E.coverage(models, DS)

st.markdown("---")
st.markdown("## 🔗 Von der Simulation zur Fläche")
st.caption(f"{n} bezahlte Simulationen, Auswahl: {D.DESIGN_LABELS[design]}. Fehler = relativer Fehler der Gesamtzeit an 100 Testpunkten, die das Modell nie gesehen hat.")
st.plotly_chart(build_design_chart(DS.pool_u[idx], DS.pool_r[idx], DS.pool_cs2[idx]), width="stretch", key=f"design_{n}_{design}_{seed}")
m1 = st.columns(5)
for col, key in zip(m1, ("jackson", "qna", "gp", "nn", "hybrid")):
    col.metric(f"{LABELS[key]}", _pct(res[key]["mean"]), help=f"mittlerer Fehler; 90-%-Quantil {_pct(res[key]['p90'])}, größter Fehler {_pct(res[key]['max'], 0)}")
rows = "\n".join(f"| {LABELS[k]} | {_pct(res[k]['mean'])} | {_pct(res[k]['p90'])} | {_pct(res[k]['max'], 0)} |" for k in ("jackson", "qna", "gp", "nn", "hybrid"))
st.markdown("| Modell | mittlerer Fehler | 90-%-Quantil | größter Fehler |\n|---|---|---|---|\n" + rows)

st.markdown("#### Schnitt durch die Fläche")
curves = E.slice_curve(models, DS, r_pct / 100, cs2, U_GRID)
truth_u, truth_w = E.slice_truth(DS, r_pct / 100, cs2)
col_a, col_b = st.columns(2)
with col_a:
    st.plotly_chart(build_slice_chart(U_GRID, curves, truth_u, truth_w, r_pct, cs2), width="stretch", key=f"slice_{n}_{design}_{seed}_{r_pct}_{cs2}")
with col_b:
    pred_test = E.predict(models, DS.test_x, DS.test_qna)
    pred_test["qna"] = DS.test_qna
    st.plotly_chart(build_parity_chart(DS.test_w, pred_test), width="stretch", key=f"parity_{n}_{design}_{seed}")
best = min(("gp", "nn", "hybrid"), key=lambda k: res[k]["mean"])
st.info(
    f"Mit {n} bezahlten Simulationen ({D.DESIGN_LABELS[design]}) liegt das beste Modell ({LABELS[best]}) im Mittel {_pct(res[best]['mean'])} neben der Wahrheit, die Zerlegung {_pct(res['qna']['mean'])}, die Jackson-Formel "
    f"{_pct(res['jackson']['mean'])} (die Formel kennt cs² nicht). Das Band im Schnitt ist das 95-%-Intervall des GP; die Punkte sind die Simulation (zwei Läufe à 150 000 Lkw)."
)

st.markdown("---")
st.subheader("📐 Lernkurven: wie viele Simulationen sind genug?")
metric = st.select_slider("Fehlermaß", options=list(METRICS), value="mean", key="metric_select", format_func=lambda m: METRICS[m])
st.markdown(
    f"Mittel über {STUDY['curves'][design]['reps']} Wiederholungen ({'ein fester Entwurf' if design == 'sobol' else 'zufälliger Start bzw. zufällige Teilmenge'}) für die Auswahl **{D.DESIGN_LABELS[design]}**. "
    "Die waagerechten Linien sind die Näherungen ohne jedes Training."
)
st.plotly_chart(build_learning_chart(STUDY, design, metric), width="stretch", key=f"learning_{design}_{metric}")
header = "| Simulationen n | " + " | ".join(LABELS[k] for k in ("gp", "nn", "hybrid")) + " |\n|---|---|---|---|\n"
body = "\n".join(f"| {s} | " + " | ".join(_pct(STUDY["curves"][design][str(s)]["errors"][k][metric]) for k in ("gp", "nn", "hybrid")) + " |" for s in STUDY["sizes"])
st.markdown(header + body)
qm = STUDY["baselines"]["qna"]["mean"]
firsts = {k: _first_n_below(design, k, qm) for k in ("gp", "nn", "hybrid")}
st.info(
    f"Die Zerlegung hat ohne jede Simulation {_pct(qm)} mittleren Fehler. Das reine GP unterbietet sie ab n = {firsts['gp'] or '–'}, das NN ab n = {firsts['nn'] or '–'}, das Hybrid ab n = {firsts['hybrid'] or '–'} "
    f"(Auswahl: {D.DESIGN_LABELS[design]}). Wer eine gute Näherung hat, braucht erst ab dort ein Surrogat; das Hybrid ist bei wenigen Simulationen im Vorteil, weil es nur noch den Rest lernen muss."
)

st.markdown("---")
st.subheader("🔬 Welche Punkte simulieren?")
model_pick = st.select_slider("Modell für den Vergleich", options=["gp", "nn", "hybrid"], value="gp", key="design_model_select", format_func=lambda m: LABELS[m])
st.markdown(
    "Dasselbe Modell, drei Arten, die bezahlten Punkte zu wählen: **raumfüllend** (Sobol-Folge, jeder Punkt liegt weit von den anderen), **zufällig**, und **aktiv** (ab einem zufälligen Start immer dort simulieren, wo "
    "das GP am unsichersten ist)."
)
st.plotly_chart(build_designs_chart(STUDY, model_pick), width="stretch", key=f"designs_{model_pick}")
rows = "\n".join(f"| {s} | " + " | ".join(_pct(STUDY["curves"][dz][str(s)]["errors"][model_pick]["mean"]) for dz in C.DESIGNS) + " |" for s in STUDY["sizes"])
st.markdown("| Simulationen n | raumfüllend | zufällig | aktiv |\n|---|---|---|---|\n" + rows)
small, large = STUDY["sizes"][0], STUDY["sizes"][-1]
cv = STUDY["curves"]
st.info(
    f"Bei n = {small} liegt {LABELS[model_pick]} mit raumfüllender Auswahl bei {_pct(cv['sobol'][str(small)]['errors'][model_pick]['mean'])}, zufällig bei {_pct(cv['random'][str(small)]['errors'][model_pick]['mean'])}, aktiv bei "
    f"{_pct(cv['active'][str(small)]['errors'][model_pick]['mean'])} (der aktive Start ist zufällig). Bei n = {large}: {_pct(cv['sobol'][str(large)]['errors'][model_pick]['mean'])} / "
    f"{_pct(cv['random'][str(large)]['errors'][model_pick]['mean'])} / {_pct(cv['active'][str(large)]['errors'][model_pick]['mean'])}. Aktives Nachsampeln holt den schlechten Start mit wachsendem n ein und senkt den größten Fehler; bei wenigen Punkten ist die raumfüllende Vorplanung besser."
)

st.markdown("---")
st.subheader("🔬 Wie sicher ist sich das GP?")
st.markdown(
    "Das GP liefert zu jeder Vorhersage eine Standardabweichung. Ein 95-%-Intervall sollte in 95 von 100 Fällen die Wahrheit enthalten. Gemessen wird das an den 100 Testpunkten (in Logarithmen, einschließlich "
    "des Rauschens der Wahrheit); dazu, ob die Standardabweichung dort groß ist, wo der Fehler groß ist."
)
st.plotly_chart(build_coverage_chart(STUDY, design), width="stretch", key=f"coverage_{design}")
rows = "\n".join(f"| {s} | {_pct(STUDY['curves'][design][str(s)]['gp']['coverage'], 0)} | {STUDY['curves'][design][str(s)]['gp']['corr']:.2f} |" for s in STUDY["sizes"])
st.markdown("| Simulationen n | Abdeckung des 95-%-Intervalls | Korrelation Standardabweichung und Fehler |\n|---|---|---|\n" + rows)
st.info(
    f"Mit den {n} gewählten Punkten deckt das 95-%-Intervall des GP an den Testpunkten {_pct(cov['coverage'], 0)} ab; Korrelation zwischen Standardabweichung und Fehler {cov['corr']:.2f}. "
    "Bei wenigen Simulationen ist das GP zu sicher; die Standardabweichung zeigt trotzdem, wo der Fehler groß ist. Ein NN liefert beides nicht."
)

st.markdown("---")
st.subheader("🔬 Extrapolation: jenseits des Trainings")
ex = STUDY["extrapolation"]
st.markdown(
    f"Training nur mit Punkten bis u = {C.EXTRAP_U:g} ({ex['n_train']} zufällige davon), getestet an den {ex['n_test']} Testpunkten darüber, wo die Gesamtzeit steil ansteigt. Mittel über {STUDY['reps']} Wiederholungen."
)
st.plotly_chart(build_extrapolation_chart(STUDY), width="stretch", key="extrapolation")
st.markdown("| Modell | mittlerer Fehler | größter Fehler |\n|---|---|---|\n" + "\n".join(f"| {LABELS[k]} | {_pct(ex['models'][k]['mean'])} | {_pct(ex['models'][k]['max'], 0)} |" for k in ("jackson", "qna", "gp", "nn", "hybrid")))
st.info(
    f"Oberhalb des Trainingsbereichs liegt das Hybrid bei {_pct(ex['models']['hybrid']['mean'])}, das reine GP bei {_pct(ex['models']['gp']['mean'])} und das NN bei {_pct(ex['models']['nn']['mean'])}. "
    "Ein reines Surrogat kennt nur seine Daten; das Hybrid erbt die Physik der Zerlegung auch dort, wo es nie gelernt hat."
)

st.markdown("---")
st.subheader("🔬 Was kostet das?")
sim_s, pred_s = _timing()
per_60k = sim_s * 3
c3 = st.columns(3)
c3[0].metric("Eine Simulation (60 000 Lkw)", f"{per_60k:.1f} s", help="auf diesem Rechner gemessen (20 000 Lkw, mal drei)")
c3[1].metric("GP: 100 Vorhersagen", f"{pred_s * 1000:.1f} ms", help="auf diesem Rechner gemessen")
c3[2].metric(f"Training für n = {n}", f"≈ {n * per_60k:.0f} s", help="n Simulationen mit 60 000 Lkw; das Anpassen des Modells selbst dauert Sekundenbruchteile")
queries = 10_000
st.info(
    f"{C.fmt_int(queries)} Fragen an das Netz kosten per Simulation (60 000 Lkw) rund {queries * per_60k / 3600:.1f} Stunden, per GP-Surrogat {queries / 100 * pred_s:.2f} s, dazu einmalig die {n} Trainingssimulationen "
    f"(≈ {n * per_60k:.0f} s). Das Surrogat lohnt sich, wenn viele Fragen anstehen (Optimierung, Sweeps). Bei wenigen Fragen ist eine Simulation billiger."
)

st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Wenige, glatte Eingaben** | Mit drei Eingaben reichen Dutzende Simulationen. Jede weitere Eingabe vervielfacht den Raum; ein GP braucht dann viele Punkte, und die Wahl des Entwurfs wird wichtiger. | kein Folgestück |
| **Die Wahrheit ist eine Simulation** | Das Surrogat erbt jeden Fehler des Simulators (hier das Modell aus Stück 12) und sein Rauschen (60 000 Lkw je Punkt). Es ersetzt die Wirklichkeit nicht. | **[Jackson-Netze](https://sebastianhanisch-jackson-network-demo.streamlit.app/)** |
| **Das Modell erklärt die Streuung** | Das neuronale Netz liefert keine Unsicherheit; das GP liefert eine, aber bei wenigen Punkten zu sichere. Intervalle mit kalibrierter Abdeckung braucht man anderswo. | **[forecast-interval-demo](https://sebastianhanisch-forecast-interval-demo.streamlit.app/)** (konforme Intervalle) |
| **Training und Einsatz im selben Bereich** | Jenseits der Trainingsdaten wird ein reines Surrogat unzuverlässig, besonders am Rand der Stabilität (u → 1). Das Hybrid mildert das, löst es nicht. | **[Prioritätsklassen](https://sebastianhanisch-priority-queue-demo.streamlit.app/)** und andere Stücke liefern die Physik |
| **Das neuronale Netz ist gut eingestellt** | Hier ein fester Aufbau (32 × 32, tanh), kein Abstimmen der Einstellungen; ein anderes Netz ändert die Rangfolge. | **[mlp-backprop-demo](https://sebastianhanisch-mlp-backprop-demo.streamlit.app/)** |
"""
)
st.caption(
    "Verwandt im Portfolio: [jackson-network-demo](https://sebastianhanisch-jackson-network-demo.streamlit.app/) (Stück 12: der Simulator), [mg1-kingman-demo](https://sebastianhanisch-mg1-kingman-demo.streamlit.app/) "
    "(Stück 10: Kingman und Allen-Cunneen), [markov-queue-demo](https://sebastianhanisch-markov-queue-demo.streamlit.app/) (Zusatzstück: wo exakte Rechnung noch möglich ist), "
    "[mlp-backprop-demo](https://sebastianhanisch-mlp-backprop-demo.streamlit.app/) (das neuronale Netz) und [forecast-interval-demo](https://sebastianhanisch-forecast-interval-demo.streamlit.app/) (Intervalle mit Abdeckung)."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Eingabe.** $x = (-\ln(1-u),\; r,\; \log_2 c_s^2)$, auf $[0,1]^3$ skaliert; Ausgabe $y = \ln W$ (mittlere Gesamtzeit), standardisiert.

**Gauß-Prozess.** $k(x,x') = \sigma_f^2\,(1 + \sqrt5 d + \tfrac53 d^2)\,e^{-\sqrt5 d}$ mit $d^2 = \sum_i (x_i - x'_i)^2/\ell_i^2$; $K = k(X,X) + \sigma_n^2 I$. Vorhersage: $\mu(x_*) = k_*^\top K^{-1} y$,
$\sigma^2(x_*) = \sigma_f^2 + \sigma_n^2 - k_*^\top K^{-1} k_*$. Hyperparameter: $\max \; -\tfrac12 y^\top K^{-1} y - \tfrac12\ln|K| - \tfrac n2 \ln 2\pi$.

**Neuronales Netz.** $\hat y = w_3^\top \tanh(W_2 \tanh(W_1 x + b_1) + b_2) + b_3$, Verlust $\tfrac1n\sum(\hat y - y)^2 + \alpha\sum\|W\|^2$, Gradient durch Rückwärtsableitung, Vollbatch-L-BFGS.

**Hybrid.** $\ln W = \ln W_{\text{Zerlegung}} + g(x)$ mit einem GP $g$ auf dem Rest.

**Aktive Auswahl.** Aus dem Pool der nächste Punkt $x^\star = \arg\max_x \sigma(x)$ unter den noch nicht gewählten.

Implementiert in `sur_models.py` (GP, NN, Hybrid), `sur_design.py` (Auswahl), `sur_network.py` (Simulator und Näherungen), `sur_evaluation.py`.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html))."
)
