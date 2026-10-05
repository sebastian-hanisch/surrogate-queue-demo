# Surrogat-Modelle – ein Ersatz für die Simulation (Streamlit-Demo)

**[→ Demo live ausprobieren](https://sebastianhanisch-surrogate-queue-demo.streamlit.app/)**

---

Interaktive Demo zu **Ersatzmodellen für teure Simulationen**. **Dreizehntes und letztes Stück der Konzepte-Linie „Warteschlangentheorie und Simulation“** im Portfolio von
[Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning): ein Verfahren, ein wachsendes Beispiel, jedes Folgestück hebt genau eine Annahme auf.

Das Terminal-Netz aus [jackson-network-demo](https://github.com/sebastian-hanisch/jackson-network-demo) (Stück 12) ist hier der „teure Simulator“. Ein **Gauß-Prozess (GP)** und ein kleines **neuronales Netz (NN)**
lernen aus wenigen bezahlten Simulationen die mittlere Gesamtzeit eines Lkw über der Auslastung des Krans, dem Rückläuferanteil und der Streuung der Dauer; ein **Hybrid** lernt nur den Rest zur Zerlegungsnäherung
nach Whitt. Verglichen wird mit den **Formeln**, die es schon gibt. Die ehrliche Frage lautet: Wann lohnt sich ein Surrogat, und wann ist die Formel gut genug?

## Kernfrage

Wie viele Simulationen braucht ein Surrogat, um eine gute Näherung zu schlagen, welche Punkte sollte man simulieren, wie sicher ist sich das Modell, und was passiert außerhalb des Trainingsbereichs?

## Modell und Methodik

- **Simulator** (`sur_network.py`): das Netz aus Stück 12 (Gate 3 Geräte, Kran 2, Stapel 4; Rückläufer vom Stapel zum Kran), hier als Kopie der nötigen Teile. Eingaben: Auslastung des Krans **u** (0.3 bis 0.95), Rückläufer **r**
  (0 bis 40 %), Streuung **cs²** der Dauer an allen Stationen (0.25, 0.5, 1, 2, 4). Ausgabe: mittlere Gesamtzeit eines Lkw. Näherungen ohne Training: **Jackson** (kennt cs² nicht) und **Zerlegung** nach Whitt.
- **Gauß-Prozess** (`sur_models.py`): Kern Matérn-5/2 mit einem Längenmaß je Eingabe plus Rauschen; Hyperparameter durch Maximieren der Randwahrscheinlichkeit (L-BFGS-B, vier feste Startpunkte); Vorhersage aus der Cholesky-Zerlegung,
  Mittelwert und Standardabweichung. Alles von Hand in NumPy und SciPy.
- **Neuronales Netz:** 3 → 32 → 32 → 1 mit tanh, quadratischer Fehler mit L2-Strafe, Rückwärtsableitung von Hand, Vollbatch-L-BFGS, Glorot-Start mit festem Seed. Keine Unsicherheit.
- **Hybrid:** GP auf dem Rest ln W − ln W_Zerlegung. **Merkmale:** (−ln(1 − u), r, log₂ cs²), auf [0, 1] skaliert; gelernt wird ln W.
- **Daten:** Kandidatenpool aus **600 Punkten** einer gescrambelten Sobol-Folge, je eine Simulation mit 60 000 Lkw (billig, verrauscht). **Wahrheit:** 100 Testpunkte, je zwei unabhängige Läufe à 150 000 Lkw, gemittelt.
- **Auswahl der bezahlten Punkte** (`sur_design.py`): die ersten n der Sobol-Folge (raumfüllend), zufällig, oder **aktiv** (ab einem zufälligen Start mit 8 Punkten immer der Kandidat mit der größten GP-Standardabweichung).
- **Gegenproben:** (1) Kern, Randwahrscheinlichkeit und Vorhersage des GP **gegen scikit-learn** bei festen Hyperparametern (Orakel nur im Test); (2) Gradient des NN gegen Differenzenquotienten und eine von Hand gerechnete
  Vorwärtsrechnung; (3) das Netz selbst: Simulation von Hand (Warten am Kran, Rückläufer), Jackson und Zerlegung gegen Stück 12; (4) das GP findet die unwichtige Eingabe (ARD) und interpoliert ohne Rauschen;
  (5) Lernkurven in der Studie gegen eine Neurechnung einer Zelle.
- **Vorgerechnete Studie** (`generate_precomputed.py` → `precomputed_sweep.json`, rund fünfeinhalb Minuten parallel): Lernkurven für neun Größen (8 bis 128) und drei Auswahlarten, je 5 Wiederholungen
  (zufälliger Start bzw. zufällige Teilmenge; raumfüllend ist deterministisch, 1 Lauf), Extrapolation, Näherungen als Vergleich, dazu Schnitte durch die Fläche (5 Rückläufer-Anteile × 5 Streuungen × 14 Auslastungen).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. Fehler = relativer Fehler der Gesamtzeit an den 100 Testpunkten (Testbereich 10 bis 70 min), Mittel über die Wiederholungen.

| Frage | Befund |
|---|---|
| Wie gut ist die Wahrheit? | Das Rauschen eines Testlaufs (log) beträgt im Median 0.4 %, im 90-%-Quantil 2.2 %, höchstens 7.8 %: die Testwerte sind genauer als jedes Modell, aber an einzelnen Punkten nicht unbegrenzt genau. |
| Was leisten die Näherungen ohne Training? | **Jackson** (ignoriert cs²) liegt im Mittel **10.9 %** daneben (90-%-Quantil 28.6 %, größter Fehler 49.3 %). Die **Zerlegung** liegt bei **2.9 %** (7.1 % / 19.5 %), am schlechtesten bei cs² = 4 und hoher Auslastung. Sie ist ein starker Konkurrent. |
| Wie lernen die Modelle (raumfüllend)? | Mittlerer Fehler bei n = 8 / 12 / 16 / 24 / 32 / 48 / 64 / 96 / 128: **GP 4.2 / 2.4 / 1.7 / 1.5 / 1.0 / 1.2 / 1.1 / 1.0 / 1.3 %**, NN 2.9 / 2.8 / 2.6 / 2.8 / 2.4 / 2.2 / 2.0 / 1.8 / 1.9 %, **Hybrid 1.5 / 1.4 / 1.6 / 1.6 / 1.1 / 1.1 / 1.5 / 1.3 / 1.2 %**. Das NN bleibt bei etwa 2 bis 3 %, das GP erreicht etwa 1 %. |
| Wann schlägt ein Surrogat die Zerlegung (2.9 %)? | Das **Hybrid schon bei n = 8** (alle Auswahlarten). GP und NN bei raumfüllender Auswahl ab **n = 12**, bei aktiver ab 24, bei zufälliger ab **32**. Ohne gute Näherung wären es mindestens so viele Simulationen; mit ihr lernt das Hybrid nur noch den Rest. |
| Welche Punkte simulieren? | Die Auswahl zählt am meisten bei wenigen Simulationen. GP bei n = 8: raumfüllend **4.2 %**, zufällig und aktiv (gleicher zufälliger Start) **12.6 %**. Bei n = 32: 1.0 / 2.1 / 1.6 %; n = 64: 1.1 / 1.4 / 1.0 %; n = 128: 1.3 / 1.1 / 0.9 % (raumfüllend / zufällig / aktiv). Aktives Nachsampeln holt den schlechten Start ein (n = 64: 1.0 % gegen 1.1 % raumfüllend; n = 128: 0.9 gegen 1.3 %) und senkt den **größten Fehler** (n = 64: GP 6 % gegen 12 % raumfüllend und 15 % zufällig; NN 7 / 23 / 23 %; Hybrid 11 / 14 / 15 %); bei wenigen Punkten ist die raumfüllende Vorplanung klar besser. |
| Wie sicher ist sich das GP? | Das nominale 95-%-Intervall **deckt nie verlässlich 95 % ab**: raumfüllend 98 / 92 / 86 / 81 / **71** / 82 / 86 / 91 / 87 %, zufällig 64 bis 94 % (ab n = 32 zwischen 92 und 94 %), aktiv 64 bis 100 %. Die Standardabweichung folgt dem Fehler (Korrelation 0.4 bis 0.8) bei raumfüllender und zufälliger Auswahl; **nach aktivem Nachsampeln fällt sie auf 0.04** (n = 128): die Unsicherheit ist überall gleich klein, weil dort gemessen wurde, wo sie groß war. |
| Was passiert außerhalb des Trainingsbereichs? | Training nur mit u ≤ 0.8 (64 von 461 Punkten), Test an den 23 Punkten darüber: **Jackson 26.4 %, Zerlegung 6.9 %, GP 13.0 % (größter Fehler 38 %), NN 11.0 % (43 %), Hybrid 4.0 % (13 %)**. Das reine GP und das NN extrapolieren schlechter als die Zerlegung; nur das Hybrid ist besser. |
| Wo liegt die Zerlegung am meisten daneben? | In der Ecke des Entwurfsraums: bei 40 % Rückläufern, cs² = 4 und u = 0.95 braucht ein Lkw nach der Simulation **105 min**, die Jackson-Formel sagt 56 min, die Zerlegung 133 min (+26 %). |

## Befunde und Korrekturen gegenüber der Vorab-Messreihe

- **Das reine GP extrapoliert in der Studie deutlich schlechter als in der Vorab-Messreihe.** Mit scikit-learn lag das GP bei 6.3 % (Zerlegung 6.9 %), mit der eigenen Implementierung bei 13.0 %; das GP wählt je nach Hyperparametern
  sehr unterschiedlich glatte Fortsetzungen. Die README nennt die Zahlen der eigenen Implementierung, die auch die App rechnet.
- **GP und NN sind bei raumfüllender Auswahl schon ab n = 12 besser als die Zerlegung**, nicht erst ab etwa 32 (so die Vorab-Messreihe mit zufälligen Teilmengen). Die Auswahl der Punkte erklärt den Unterschied;
  mit zufälliger Auswahl sind es wie dort 32.
- **Aktives Nachsampeln ist kein Gewinn bei wenigen Punkten.** Die Vorab-Messreihe verglich nur bis n = 64 und mit anderem GP; die Studie zeigt: bei wenigen Punkten ist es wegen des zufälligen Starts schlechter als raumfüllend, ab n = 64 gleichauf
  oder etwas besser, mit kleinerem größtem Fehler, aber mit einer wertlos werdenden Standardabweichung (Korrelation bis 0.04).
- **Die GP-Intervalle sind bei wenigen Punkten nicht zu sicher, sondern unberechenbar:** bei raumfüllender Auswahl 98 % bei n = 8 (das GP ist dort sehr unsicher) und 71 % bei n = 32, nicht monoton.

## Ehrliche Grenzen

- **Drei Eingaben.** Das GP kommt mit Dutzenden Punkten aus; jede weitere Eingabe vervielfacht den Raum. Mit zehn Eingaben sähen die Lernkurven anders aus, nicht gemessen.
- **Die Wahrheit ist selbst eine Simulation.** Das Surrogat erbt jeden Fehler des Netz-Modells aus Stück 12 und sein Rauschen (60 000 Lkw je Pool-Punkt). Es sagt nichts über die Wirklichkeit.
- **Streuung nur in fünf Stufen** (die Erlang- und Hyperexponential-Familien aus Stück 12), die Zahl der Geräte je Station ist fest; ein anderes Terminal ist ein anderer Simulator.
- **Ein festes NN** (32 × 32, tanh, L2-Strafe 10⁻³), ohne Abstimmen der Einstellungen und ohne Ensemble; ein anderes Netz ändert die Rangfolge. Das GP ist ebenfalls nur mit einem Kern (Matérn-5/2) gerechnet.
- **Studie mit 5 Wiederholungen** (raumfüllend: eine Folge); die Mittel schwanken von Größe zu Größe um Zehntel Prozentpunkte (z. B. raumfüllendes GP 1.0 bei n = 32, 1.2 bei n = 48), die Rangfolgen sind erst über mehrere Größen belastbar.
- **Die Zeiten der App** (Simulation und Vorhersage) werden auf dem jeweiligen Rechner gemessen und hängen von ihm ab; belastbar ist nur die Größenordnung (rund eine Sekunde je Simulation mit 60 000 Lkw, Millisekunden je GP-Vorhersage).
- **Intervalle ohne Kalibrierung.** Konforme Intervalle, die die Abdeckung im stationären Fall treffen (siehe forecast-interval-demo), sind nicht gerechnet.

## Verwandte Demos im Portfolio

- [`jackson-network-demo`](https://github.com/sebastian-hanisch/jackson-network-demo) (Stück 12): der Simulator und die Zerlegung, auf der das Hybrid beruht.
- [`mg1-kingman-demo`](https://github.com/sebastian-hanisch/mg1-kingman-demo) (Stück 10): Kingman und Allen-Cunneen.
- [`markov-queue-demo`](https://github.com/sebastian-hanisch/markov-queue-demo) (Zusatzstück): wo die exakte Rechnung noch möglich ist.
- [`mlp-backprop-demo`](https://github.com/sebastian-hanisch/mlp-backprop-demo): das neuronale Netz mit Rückwärtsableitung und Gradienten-Check, hier als Surrogat eingesetzt.
- [`forecast-interval-demo`](https://github.com/sebastian-hanisch/forecast-interval-demo): konforme Intervalle, die im stationären Fall die Nennabdeckung treffen und dem GP fehlen.

## Bewusst nicht umgesetzt

Dies ist das letzte Stück der Linie; nichts davon hat ein Folgestück:

| Annahme | Verweis |
|---|---|
| Wenige, glatte Eingaben | kein Folgestück |
| Die Wahrheit ist eine Simulation | [Jackson-Netze](https://github.com/sebastian-hanisch/jackson-network-demo) (der Simulator) |
| Intervalle mit Abdeckung | [forecast-interval-demo](https://github.com/sebastian-hanisch/forecast-interval-demo) |
| Training und Einsatz im selben Bereich | [Prioritätsklassen](https://github.com/sebastian-hanisch/priority-queue-demo) und andere Stücke liefern die Physik für das Hybrid |
| Gut eingestelltes neuronales Netz | [mlp-backprop-demo](https://github.com/sebastian-hanisch/mlp-backprop-demo) |

## Tests

146 Tests, rund 85 Sekunden: der Simulator (Simulation von Hand mit Warten am Kran und Rückläufern, Jackson und Zerlegung gegen Stück 12, Sampler, Reproduzierbarkeit, Simulation gegen Formel), das GP (Kern von Hand, Randwahrscheinlichkeit und Vorhersage
gegen scikit-learn, Interpolation, Unsicherheit, ARD, Hybrid), das NN (Vorwärtsrechnung von Hand, Gradient gegen Differenzenquotienten, Glorot-Start, Anpassung), die Versuchsplanung (Sobol, zufällig, aktiv von Hand), Auswertung und Vollständigkeit der
vorgerechneten Studie (inklusive Neurechnung einer Zelle), Presets und Permalink, Diagramme (gesperrte Achsen), AppTest-Rauchtests mit festem Würfel-Seed, der Smoke-Test der Portfolio-Vorlage, ein Quelltext-Test gegen Satz-Komma-Fehler, Orakeltests für den Simulator (Jackson aus der Geburts-Sterbe-Kette, Zerlegung direkt aufgelöst, unabhängiger Netz-Simulator) und
`test_claims.py` für jede Zahl dieser README.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `sur_network.py` | Simulator (Netz aus Stück 12), Jackson, Zerlegung |
| `sur_models.py` | Gauß-Prozess, neuronales Netz, Hybrid, Merkmale, Fehlermaße |
| `sur_design.py` | Sobol, zufällige und aktive Auswahl der Punkte |
| `sur_evaluation.py` | Datensatz, Training auf einer Auswahl, Fehler, Abdeckung, Schnitte, Lesen der Studie |
| `generate_precomputed.py` | rechnet Daten und Studie vor → `precomputed_sweep.json` |
| `sur_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `sur_presets.py`, `sur_constants.py` | Presets, Permalink, Grenzen |
| `tests/` | siehe oben |

## Literatur

- Sacks, J., Welch, W. J., Mitchell, T. J., Wynn, H. P. (1989): Design and analysis of computer experiments. *Statistical Science* 4(4), 409–423 (Gauß-Prozesse als Ersatz für teure Simulationen).
- Rasmussen, C. E., Williams, C. K. I. (2006): *Gaussian Processes for Machine Learning*. MIT Press (Kern, Randwahrscheinlichkeit, Vorhersage).
- Sobol, I. M. (1967): On the distribution of points in a cube and the approximate evaluation of integrals. *USSR Computational Mathematics and Mathematical Physics* 7(4), 86–112 (die raumfüllende Folge).
- Whitt, W. (1983): The queueing network analyzer. *The Bell System Technical Journal* 62(9), 2779–2815 (die Zerlegung, auf der das Hybrid beruht).

## Lokal ausführen

```
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/ -v`. Studie neu rechnen: `python generate_precomputed.py` (nur die Studie: `python generate_precomputed.py 14 study`).

Gebaut mit Streamlit und Plotly (die Modelle rechnen NumPy und SciPy).

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Warteschlangentheorie: M/M/1 bis Surrogat](https://sebastianhanisch.net/konzepte-warteschlangentheorie.html).
