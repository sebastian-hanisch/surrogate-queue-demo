import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class Seq:
    """Liefert vorgegebene Werte der Reihe nach (Zwischenankunftszeiten, Dauern) statt Zufall: Mini-Instanzen von Hand gerechnet."""

    def __init__(self, values):
        self.values = list(values)
        self.i = 0

    def __call__(self):
        v = self.values[self.i]
        self.i += 1
        return v


class Route:
    """Routing-Zufall mit festen Werten: uniform() liefert der Reihe nach die vorgegebenen Zahlen."""

    def __init__(self, values):
        self.values = list(values)
        self.i = 0

    def uniform(self):
        v = self.values[self.i]
        self.i += 1
        return v


@pytest.fixture
def queueing_script():
    """Drei Stationen (Gate 3, Kran 2, Stapel 4 Geräte), Lkw 1 / 2 / 3 kommen bei t = 1 / 2 / 3. Dauer Gate 1.5, Kran 3.0, Stapel 0.5, Weg Gate → Kran → Stapel → Ausgang.
    Von Hand: Gate ohne Warten: fertig bei 2.5 / 3.5 / 4.5. Kran (2 Geräte): Lkw 1 von 2.5 bis 5.5, Lkw 2 von 3.5 bis 6.5, Lkw 3 kommt bei 4.5, beide Geräte sind belegt, es startet bei 5.5 und endet bei 8.5.
    Stapel ohne Warten: Lkw 1 von 5.5 bis 6.0 (Gesamtzeit 5.0), Lkw 2 von 6.5 bis 7.0 (5.0), Lkw 3 von 8.5 bis 9.0 (6.0). Mittel 16/3."""
    return dict(arrival=Seq([1, 1, 1]), services=[lambda: 1.5, lambda: 3.0, lambda: 0.5], p=[[0, 1, 0], [0, 0, 1], [0, 0, 0]])
