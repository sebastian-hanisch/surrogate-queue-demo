"""Der „teure Simulator“ der Demo: das Terminal-Netz Gate → Kran → Stapel mit Rückläufern aus jackson-network-demo (Stück 12), hier als Kopie der nötigen Teile, damit das Repo für sich läuft.

Enthält die Formeln (Verkehrsgleichungen, Jackson, Whitt-Zerlegung) und die Netz-Simulation ohne Platzgrenzen (SplitMix64, Dauer fest / Erlang-k / exponentiell / hyperexponentiell). Zeiten in Minuten.
Eingaben des Entwurfsraums: u = Auslastung des Krans, r = Rückläufer, cs² = Streuung der Dauer an allen Stationen; die Ankunftsrate folgt aus u und r (`arrival_rate`)."""

import heapq
import math
from collections import deque

import numpy as np

STATIONS = (("Gate", 3, 3.0), ("Kran", 2, 2.4), ("Stapel", 4, 4.0))
SERVERS = tuple(s[1] for s in STATIONS)
MEANS = tuple(s[2] for s in STATIONS)
CRANE = 1                                                              # Index des Krans

_MASK = (1 << 64) - 1


def routing_matrix(r):
    """Gate → Kran → Stapel; vom Stapel gehen r der Lkw zurück zum Kran (Umstapeln), die übrigen verlassen das Netz."""
    return [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, float(r), 0.0]]


def arrival_rate(u, r):
    """Ankünfte je Minute, für die der Kran (2 Geräte, 2.4 min, Rate γ/(1 − r)) zu Anteil u ausgelastet ist: γ = u·(1 − r)·c/m."""
    return u * (1.0 - r) * SERVERS[CRANE] / MEANS[CRANE]


# --- Formeln ---------------------------------------------------------------------------------------------------------------------------------------

def erlang_c_wait(c, a):
    """Mittlere Wartezeit eines M/M/c in Abfertigungsdauern bei Angebot a (Erlang), stabil über die Erlang-B-Rekursion; unendlich ohne Gleichgewicht."""
    if a >= c:
        return math.inf
    b = 1.0
    for j in range(1, c + 1):
        b = a * b / (j + a * b)
    rho = a / c
    return b / (1.0 - rho * (1.0 - b)) / (c * (1.0 - rho))


def traffic_rates(gamma, p):
    """Verkehrsgleichungen λ = γ + Pᵀλ."""
    p = np.asarray(p, float)
    return np.linalg.solve(np.eye(len(p)) - p.T, np.asarray(gamma, float))


def jackson_total(gamma_rate, r):
    """Gesamtzeit eines Lkw nach Jackson (jede Station M/M/c, Dauer exponentiell): Summe der mittleren Zahlen geteilt durch γ (Little)."""
    lam = traffic_rates([gamma_rate, 0.0, 0.0], routing_matrix(r))
    total = 0.0
    for i in range(3):
        a = lam[i] * MEANS[i]
        total += lam[i] * (erlang_c_wait(SERVERS[i], a) * MEANS[i] + MEANS[i])
    return total / gamma_rate


def departure_scv(rho, c, ca2, cs2):
    """Streuung des Abgangsprozesses einer Station nach Whitt: 1 + (1 − ρ²)(ca² − 1) + ρ²(cs² − 1)/√c."""
    return 1.0 + (1.0 - rho ** 2) * (ca2 - 1.0) + rho ** 2 * (cs2 - 1.0) / math.sqrt(c)


def qna_total(gamma_rate, r, cs2, tol=1e-12, max_iter=10_000):
    """Zerlegungsnäherung (Whitt) für das Terminal: jede Station G/G/c nach Allen-Cunneen (Erlang-C-Wartezeit mal (ca² + cs²)/2), Streuung der Ankunftsströme per Fixpunktiteration
    (Verzweigung c² = 1 + p(cd² − 1), Zusammenfluss mit den Raten gewichtet). Gesamtzeit in Minuten."""
    p = np.asarray(routing_matrix(r), float)
    gamma = np.array([gamma_rate, 0.0, 0.0])
    lam = traffic_rates(gamma, p)
    rho = np.array([lam[i] * MEANS[i] / SERVERS[i] for i in range(3)])
    ca2 = np.ones(3)
    for _ in range(max_iter):
        cd2 = np.array([departure_scv(rho[i], SERVERS[i], ca2[i], cs2) for i in range(3)])
        new = np.ones(3)
        for j in range(3):
            flow = gamma[j]
            for i in range(3):
                if p[i, j] > 0:
                    flow += lam[i] * p[i, j] * (1.0 + p[i, j] * (cd2[i] - 1.0))
            new[j] = flow / lam[j]
        done = float(np.abs(new - ca2).max()) < tol
        ca2 = new
        if done:
            break
    total = 0.0
    for i in range(3):
        wq = erlang_c_wait(SERVERS[i], lam[i] * MEANS[i]) * MEANS[i] * (ca2[i] + cs2) / 2.0
        total += lam[i] * (wq + MEANS[i])
    return total / gamma_rate


# --- Simulation ------------------------------------------------------------------------------------------------------------------------------------

class SplitMix64:
    """Kleiner, gut gemischter 64-Bit-Zufallsgenerator (Vigna); reine Ganzzahl-Arithmetik."""

    def __init__(self, seed):
        self.state = seed & _MASK

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return z ^ (z >> 31)

    def uniform(self):
        return (self.next() >> 11) * (1.0 / (1 << 53))

    def expovariate(self, rate):
        return -math.log(1.0 - self.uniform()) / rate


def make_sampler(mean, scv, rng):
    """Funktion ohne Argument, die Werte mit Mittel `mean` und Variationskoeffizient² `scv` zieht: fest, Erlang-k, exponentiell oder Hyperexponential (gleiche Phasenanteile am Mittel)."""
    if scv == 0:
        return lambda: mean
    if scv == 1:
        return lambda: rng.expovariate(1.0 / mean)
    if scv < 1:
        k = round(1.0 / scv)
        return lambda: sum(rng.expovariate(k / mean) for _ in range(k))
    p1 = 0.5 * (1.0 + math.sqrt((scv - 1.0) / (scv + 1.0)))
    rate1, rate2 = 2.0 * p1 / mean, 2.0 * (1.0 - p1) / mean
    return lambda: rng.expovariate(rate1) if rng.uniform() < p1 else rng.expovariate(rate2)


def simulate_total(gamma_rate, r, cs2, trucks, seed, warm_fraction=0.05, arrival=None, services=None, rng_route=None, p=None):
    """Mittlere Gesamtzeit eines Lkw (Minuten) im Terminal-Netz: `trucks` Poisson-Ankünfte mit Rate `gamma_rate` am Gate, FIFO, Rückläufer nach `r`; die ersten warm_fraction der Abgänge zählen nicht.
    `arrival`, `services` (Liste je Station), `rng_route` und `p` lassen sich zum Testen von außen vorgeben (Mini-Instanzen von Hand)."""
    n = 3
    p = routing_matrix(r) if p is None else p
    arr_rng = SplitMix64(seed)
    svc_rngs = [SplitMix64(seed + 1_000 * (i + 1) + 99_991) for i in range(n)]
    route_rng = rng_route or SplitMix64(seed + 777_777)
    arrival = arrival or (lambda: arr_rng.expovariate(gamma_rate))
    services = services or [make_sampler(MEANS[i], cs2, svc_rngs[i]) for i in range(n)]
    cum = []
    for i in range(n):
        acc, row = 0.0, []
        for j in range(n):
            acc += p[i][j]
            row.append(acc)
        cum.append(row)
    queue = [deque() for _ in range(n)]
    busy = [0] * n
    ev, seq, entry, sojourn = [], 0, {}, []
    warm = int(warm_fraction * trucks)
    finished, next_job, arrivals_left = 0, 0, trucks

    def try_start(i, now):
        nonlocal seq
        while busy[i] < SERVERS[i] and queue[i]:
            job = queue[i].popleft()
            busy[i] += 1
            seq += 1
            heapq.heappush(ev, (now + services[i](), seq, 1, i, job))

    seq += 1
    heapq.heappush(ev, (arrival(), seq, 0, -1, -1))
    while ev and finished < trucks:
        t, _, kind, i, job = heapq.heappop(ev)
        if kind == 0:
            arrivals_left -= 1
            entry[next_job] = t
            queue[0].append(next_job)
            next_job += 1
            try_start(0, t)
            if arrivals_left > 0:
                seq += 1
                heapq.heappush(ev, (t + arrival(), seq, 0, -1, -1))
        else:
            u = route_rng.uniform()
            dest = next((j for j in range(n) if u < cum[i][j]), None)
            busy[i] -= 1
            if dest is None:
                sojourn.append(t - entry.pop(job))
                finished += 1
            else:
                queue[dest].append(job)
                try_start(dest, t)
            try_start(i, t)
    kept = sojourn[warm:]
    return sum(kept) / len(kept)
