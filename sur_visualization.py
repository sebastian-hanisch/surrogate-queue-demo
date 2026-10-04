"""Plotly-Abbildungen der Surrogat-Demo: Entwurf, Schnitt durch die Fläche, Treffergenauigkeit (Parity), Lernkurven, Entwürfe im Vergleich, GP-Abdeckung, Extrapolation. Achsen sind gesperrt
(fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen."""

import numpy as np
import plotly.graph_objects as go

import sur_constants as C

COLORS = {"jackson": "#9d9d9d", "qna": "#54a24b", "gp": "#4c78a8", "nn": "#e45756", "hybrid": "#b279a2", "truth": "#222222"}
LABELS = {"jackson": "Formel (Jackson)", "qna": "Zerlegung (Whitt)", "gp": "Gauß-Prozess", "nn": "Neuronales Netz", "hybrid": "Hybrid (Zerlegung + GP)"}
CS2_COLORS = {0.25: "#4c78a8", 0.5: "#72b7b2", 1.0: "#54a24b", 2.0: "#f58518", 4.0: "#e45756"}
MODELS = ("gp", "nn", "hybrid")
METRICS = {"mean": "mittlerer Fehler", "p90": "90-%-Quantil des Fehlers", "max": "größter Fehler"}
TICKS = [10, 20, 50, 100, 200, 500]


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height, legend_y=-0.28, top=10):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=top, b=10), legend=dict(orientation="h", y=legend_y), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _log_ticks(values):
    lo, hi = min(values), max(values)
    ticks = [t for t in TICKS if lo / 1.5 <= t <= hi * 1.5] or [round(lo), round(hi)]
    return ticks, [str(t) for t in ticks]


def build_design_chart(u, r, cs2):
    """Die bezahlten Simulationen im Entwurfsraum: Auslastung des Krans gegen Rückläufer, Farbe = Streuung der Dauer."""
    fig = go.Figure()
    u, r, cs2 = np.asarray(u), np.asarray(r), np.asarray(cs2)
    for level in sorted(CS2_COLORS):
        m = cs2 == level
        fig.add_trace(go.Scatter(x=u[m], y=r[m] * 100, mode="markers", marker=dict(size=10, color=CS2_COLORS[level], line=dict(color="white", width=1)), name=f"cs² = {level:g}"))
    fig.update_xaxes(title_text="Auslastung des Krans u", range=[0.28, 0.97])
    fig.update_yaxes(title_text="Rückläufer (%)", range=[-2, 42])
    return _base(fig, 300)


def build_slice_chart(u, curves, truth_u, truth_w, r_pct, cs2, show=MODELS):
    """Schnitt durch die Fläche bei festem r und cs²: Gesamtzeit über die Auslastung des Krans. Wahrheit als Punkte, Näherungen gestrichelt, GP mit 95-%-Band."""
    fig = go.Figure()
    u = np.asarray(u)
    sd = np.asarray(curves["gp_sd"])
    gp = np.asarray(curves["gp"])
    fig.add_trace(go.Scatter(x=np.r_[u, u[::-1]], y=np.r_[gp * np.exp(1.96 * sd), (gp * np.exp(-1.96 * sd))[::-1]], fill="toself", fillcolor="rgba(76,120,168,0.18)", line=dict(width=0),
                             hoverinfo="skip", name="GP 95-%-Band"))
    for key in ("jackson", "qna"):
        fig.add_trace(go.Scatter(x=u, y=curves[key], mode="lines", line=dict(color=COLORS[key], width=2, dash="dash"), name=LABELS[key]))
    for key in show:
        fig.add_trace(go.Scatter(x=u, y=curves[key], mode="lines", line=dict(color=COLORS[key], width=2.5), name=LABELS[key]))
    fig.add_trace(go.Scatter(x=truth_u, y=truth_w, mode="markers", marker=dict(color=COLORS["truth"], size=9, symbol="diamond"), name="Simulation (Wahrheit)"))
    ticks, texts = _log_ticks(list(truth_w) + list(curves["qna"]))
    fig.update_xaxes(title_text=f"Auslastung des Krans u ({r_pct} % Rückläufer, cs² = {cs2:g})")
    fig.update_yaxes(title_text="Gesamtzeit eines Lkw (min, logarithmisch)", type="log", tickvals=ticks, ticktext=texts)
    return _base(fig, 360, legend_y=-0.35)


def build_parity_chart(truth, preds, show=("gp", "nn", "hybrid", "qna")):
    """Treffergenauigkeit an den 100 Testpunkten: Wahrheit (x) gegen Vorhersage (y); auf der Diagonale liegt die perfekte Vorhersage."""
    fig = go.Figure()
    lo, hi = float(np.min(truth)) * 0.9, float(np.max(truth)) * 1.1
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=COLORS["truth"], dash="dot"), name="perfekt", hoverinfo="skip"))
    for key in show:
        fig.add_trace(go.Scatter(x=truth, y=preds[key], mode="markers", marker=dict(color=COLORS[key], size=6, opacity=0.75), name=LABELS[key]))
    ticks, texts = _log_ticks([lo, hi])
    fig.update_xaxes(title_text="Wahrheit (min)", type="log", tickvals=ticks, ticktext=texts, range=[np.log10(lo), np.log10(hi)])
    fig.update_yaxes(title_text="Vorhersage (min)", type="log", tickvals=ticks, ticktext=texts, range=[np.log10(lo), np.log10(hi)])
    return _base(fig, 360, legend_y=-0.35)


def build_learning_chart(study, design, metric, show=MODELS):
    """Lernkurve: Fehler an den Testpunkten über die Zahl der bezahlten Simulationen; Näherungen waagerecht."""
    sizes = study["sizes"]
    fig = go.Figure()
    for key in show:
        fig.add_trace(go.Scatter(x=sizes, y=[study["curves"][design][str(n)]["errors"][key][metric] for n in sizes], mode="lines+markers", line=dict(color=COLORS[key], width=2.5), name=LABELS[key]))
    for key in ("jackson", "qna"):
        fig.add_trace(go.Scatter(x=[sizes[0], sizes[-1]], y=[study["baselines"][key][metric]] * 2, mode="lines", line=dict(color=COLORS[key], dash="dash", width=2), name=LABELS[key]))
    fig.update_xaxes(title_text="bezahlte Simulationen n", type="log", tickvals=sizes, ticktext=[str(n) for n in sizes])
    fig.update_yaxes(title_text=f"{METRICS[metric]} der Gesamtzeit", type="log", tickformat=".0%", tickvals=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5], ticktext=["0.5 %", "1 %", "2 %", "5 %", "10 %", "20 %", "50 %"])
    return _base(fig, 340)


def build_designs_chart(study, model, metric="mean"):
    """Gleiches Modell, drei Entwürfe: Fehler über die Zahl der Simulationen."""
    sizes = study["sizes"]
    colors = {"sobol": "#4c78a8", "random": "#9d9d9d", "active": "#e45756"}
    names = {"sobol": "raumfüllend (Sobol)", "random": "zufällig", "active": "aktiv (größte Unsicherheit)"}
    fig = go.Figure()
    for design in C.DESIGNS:
        fig.add_trace(go.Scatter(x=sizes, y=[study["curves"][design][str(n)]["errors"][model][metric] for n in sizes], mode="lines+markers", line=dict(color=colors[design], width=2.5), name=names[design]))
    fig.update_xaxes(title_text="bezahlte Simulationen n", type="log", tickvals=sizes, ticktext=[str(n) for n in sizes])
    fig.update_yaxes(title_text=f"{METRICS[metric]} ({LABELS[model]})", type="log", tickvals=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5], ticktext=["0.5 %", "1 %", "2 %", "5 %", "10 %", "20 %", "50 %"])
    return _base(fig, 340)


def build_coverage_chart(study, design):
    """Abdeckung der nominalen 95-%-Intervalle des GP an den Testpunkten über die Zahl der Simulationen; 95 % ist das Soll."""
    sizes = study["sizes"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sizes, y=[study["curves"][design][str(n)]["gp"]["coverage"] for n in sizes], mode="lines+markers", line=dict(color=COLORS["gp"], width=2.5), name="Abdeckung des GP-Intervalls"))
    fig.add_hline(y=0.95, line=dict(color=COLORS["truth"], dash="dash"), annotation_text="Soll 95 %")
    fig.update_xaxes(title_text="bezahlte Simulationen n", type="log", tickvals=sizes, ticktext=[str(n) for n in sizes])
    fig.update_yaxes(title_text="Anteil der Testpunkte im Intervall", tickformat=".0%", range=[0, 1.05])
    return _base(fig, 320)


def build_extrapolation_chart(study, metric="mean"):
    """Fehler oberhalb des Trainingsbereichs (Training nur u ≤ 0.8, Test u > 0.8): Modelle und Näherungen."""
    ex = study["extrapolation"]["models"]
    keys = ["jackson", "qna", "gp", "nn", "hybrid"]
    fig = go.Figure(go.Bar(x=[LABELS[k] for k in keys], y=[ex[k][metric] for k in keys], marker_color=[COLORS[k] for k in keys], text=[f"{ex[k][metric]:.1%}".replace("%", " %") for k in keys], textposition="outside"))
    fig.update_yaxes(title_text=METRICS[metric], tickformat=".0%", rangemode="tozero")
    return _base(fig, 320)
