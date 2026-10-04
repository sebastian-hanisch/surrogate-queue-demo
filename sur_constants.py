"""Konstanten der Surrogat-Demo: Entwurfsraum, Pool- und Teststudie, Regler, Voreinstellungen. Zeiten in Minuten."""


def fmt_int(n):
    """Ganzzahl mit Punkt als Tausendertrenner (10000 -> 10.000)."""
    return f"{n:,}".replace(",", ".")


def fmt_pct(x, digits=0):
    """Anteil als Prozent mit Leerzeichen (0.086 -> "9 %", mit digits=1 "8.6 %")."""
    return f"{x:.{digits}%}".replace("%", " %")


POOL_SIZE = 600                                                       # Kandidatenpool (Sobol-Reihenfolge), je Punkt eine Simulation mit POOL_TRUCKS Lkw
POOL_TRUCKS = 60_000
TEST_SIZE = 100                                                       # Testpunkte als Wahrheit: je zwei unabhängige Läufe à TEST_TRUCKS Lkw
TEST_TRUCKS = 150_000
POOL_SEED, TEST_SEED = 4242, 777                                      # Seeds der gescrambelten Sobol-Folgen

SLICE_R = (0.0, 0.1, 0.2, 0.3, 0.4)                                   # Schnitte durch die Fläche: Wahrheit je (r, cs², u)
SLICE_U = tuple(round(0.3 + 0.05 * i, 2) for i in range(14))          # 0.30 … 0.95
SLICE_TRUCKS = 150_000

SIZES = (8, 12, 16, 24, 32, 48, 64, 96, 128)                          # Zahl bezahlter Simulationen in der Lernkurve
STUDY_REPS = 5                                                        # Wiederholungen je Größe (zufälliger Start / zufällige Teilmenge)
EXTRAP_U = 0.8                                                        # Extrapolation: Training nur u ≤ 0.8, Test darüber
EXTRAP_N = 64

N_MIN, N_MAX, N_STEP, DEFAULT_N = 8, 64, 4, 24                        # Regler: Zahl der Simulationen
R_PCT_MIN, R_PCT_MAX, R_PCT_STEP, DEFAULT_R_PCT = 0, 40, 10, 20       # Schnitt: Rückläufer
CS2_OPTIONS = (0.25, 0.5, 1.0, 2.0, 4.0)
DEFAULT_CS2 = 4.0
DESIGNS = ("sobol", "random", "active")
DEFAULT_DESIGN = "sobol"
SEED_MAX = 999999
DEFAULT_SEED = 35

PRESET_ORDER = ("Wenige Simulationen", "Genug Simulationen", "Aktiv nachsampeln", "Rückläufer und starke Streuung")


def _preset(n=DEFAULT_N, design=DEFAULT_DESIGN, r_pct=DEFAULT_R_PCT, cs2=DEFAULT_CS2):
    return {"n": n, "design": design, "r_pct": r_pct, "cs2": cs2, "seed": DEFAULT_SEED}


PRESETS = {
    "Wenige Simulationen": _preset(n=8),
    "Genug Simulationen": _preset(n=64),
    "Aktiv nachsampeln": _preset(design="active"),
    "Rückläufer und starke Streuung": _preset(r_pct=40),
}
# Zahlen aus der vorgerechneten Studie (Mittel über Wiederholungen) und dem Schnitt; tests/test_claims.py rechnet jede nach
PRESET_HELP = {
    "Wenige Simulationen": "8 raumfüllende Simulationen: mittlerer Fehler laut Studie GP 4.2 %, NN 2.9 %, Hybrid 1.5 %; die Zerlegung ohne jede Simulation 2.9 %, die Jackson-Formel 10.9 %.",
    "Genug Simulationen": "64 raumfüllende Simulationen: GP 1.1 %, NN 2.0 %, Hybrid 1.5 % mittlerer Fehler (Zerlegung 2.9 %, Jackson 10.9 %).",
    "Aktiv nachsampeln": "24 aktiv gewählte Simulationen (Mittel über 5 zufällige Starts): GP 2.3 %, NN 2.6 %, Hybrid 2.8 %; raumfüllend mit 24 wären es GP 1.5 %.",
    "Rückläufer und starke Streuung": "Schnitt bei 40 % Rückläufer und cs² = 4: bei u = 0.95 braucht ein Lkw 105 min (Simulation); die Jackson-Formel sagt 56 min, die Zerlegung 133 min.",
}
