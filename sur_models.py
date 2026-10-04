"""Die Surrogat-Modelle von Hand in NumPy/SciPy: ein Gauß-Prozess (GP) und ein kleines neuronales Netz (NN), dazu die Merkmalsabbildung des Entwurfsraums und das Hybrid-Modell.

**GP.** Kern k(x, x') = σ_f² · Matérn-5/2(d) mit d² = Σ_i (x_i − x'_i)²/ℓ_i² (ein Längenmaß je Eingabe, „ARD“), dazu Rauschen σ_n² auf der Diagonale. Die Hyperparameter (σ_f, ℓ₁…ℓ₃, σ_n) maximieren die
logarithmierte Randwahrscheinlichkeit (L-BFGS-B, feste Startpunkte); die Vorhersage ist Mittelwert und Standardabweichung aus der Cholesky-Zerlegung. Ausgaben werden vorher standardisiert.

**NN.** Vollverbundenes Netz 3 → H → H → 1 mit tanh, quadratischer Fehler plus L2-Strafe, Rückwärtsableitung von Hand, Vollbatch-L-BFGS-B, Startgewichte nach Glorot aus einem festen Seed.

**Hybrid.** Das GP lernt nur den Rest zwischen der Simulation und einer physikalischen Näherung (Zerlegung nach Whitt), jeweils in Logarithmen."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

SQRT5 = np.sqrt(5.0)

# Entwurfsraum: u = Auslastung des Krans, r = Rückläufer, cs² = Streuung der Dauer
U_MIN, U_MAX = 0.3, 0.95
R_MIN, R_MAX = 0.0, 0.4
CS2_LEVELS = (0.25, 0.5, 1.0, 2.0, 4.0)
LO = np.array([-np.log(1 - U_MIN), R_MIN, np.log2(CS2_LEVELS[0])])
HI = np.array([-np.log(1 - U_MAX), R_MAX, np.log2(CS2_LEVELS[-1])])


def features(u, r, cs2):
    """Eingabe der Modelle: (−ln(1 − u), r, log₂ cs²), auf [0, 1] skaliert. Die Gesamtzeit wächst wie 1/(1 − u); −ln(1 − u) streckt den Rand."""
    x = np.column_stack([-np.log(1.0 - np.asarray(u, float)), np.asarray(r, float), np.log2(np.asarray(cs2, float))])
    return (x - LO) / (HI - LO)


# --- Gauß-Prozess ----------------------------------------------------------------------------------------------------------------------------------

def matern52(x1, x2, ls):
    """Matérn-5/2-Kern (ohne Faktor σ_f²) mit Längenmaß je Eingabe: (1 + √5 d + 5d²/3)·exp(−√5 d)."""
    diff = (x1[:, None, :] - x2[None, :, :]) / ls
    d = np.sqrt(np.maximum((diff ** 2).sum(axis=2), 0.0))
    return (1.0 + SQRT5 * d + 5.0 * d ** 2 / 3.0) * np.exp(-SQRT5 * d)


def log_marginal_likelihood(theta, x, y):
    """Log-Randwahrscheinlichkeit eines GP mit θ = (ln σ_f, ln ℓ₁, ln ℓ₂, ln ℓ₃, ln σ_n) für standardisierte Ausgaben y: −½yᵀK⁻¹y − Σ ln L_ii − n/2 ln 2π."""
    sf, ls, sn = np.exp(theta[0]), np.exp(theta[1:4]), np.exp(theta[4])
    k = sf ** 2 * matern52(x, x, ls) + (sn ** 2 + 1e-10) * np.eye(len(x))
    try:
        c, low = cho_factor(k, lower=True)
    except np.linalg.LinAlgError:
        return -1e10
    alpha = cho_solve((c, low), y)
    return float(-0.5 * y @ alpha - np.log(np.diag(c)).sum() - 0.5 * len(x) * np.log(2 * np.pi))


BOUNDS = [(np.log(0.1), np.log(10.0))] + [(np.log(0.05), np.log(20.0))] * 3 + [(np.log(1e-4), np.log(0.3))]
STARTS = [np.log([1.0, 0.3, 0.3, 0.3, 0.05]), np.log([1.0, 1.0, 1.0, 1.0, 0.01]), np.log([2.0, 3.0, 3.0, 3.0, 0.1]), np.log([1.0, 0.5, 2.0, 1.0, 0.02])]


@dataclass
class GP:
    x: np.ndarray
    theta: np.ndarray
    mu: float
    sd: float
    alpha: np.ndarray
    chol: tuple

    def predict(self, xs):
        """Mittelwert und Standardabweichung (einschließlich Rauschen, wie bei einer neuen Messung) an den Punkten xs, in den Einheiten der ursprünglichen Ausgabe."""
        sf, ls, sn = np.exp(self.theta[0]), np.exp(self.theta[1:4]), np.exp(self.theta[4])
        ks = sf ** 2 * matern52(xs, self.x, ls)
        mean = ks @ self.alpha
        v = cho_solve(self.chol, ks.T)
        var = np.maximum(sf ** 2 + sn ** 2 - (ks * v.T).sum(axis=1), 1e-12)
        return mean * self.sd + self.mu, np.sqrt(var) * self.sd


def gp_build(x, y, theta):
    """GP mit festen Hyperparametern θ aus den Trainingsdaten (x skaliert, y in Originaleinheiten, wird standardisiert)."""
    mu, sd = float(np.mean(y)), float(np.std(y)) + 1e-12
    ys = (np.asarray(y, float) - mu) / sd
    sf, ls, sn = np.exp(theta[0]), np.exp(theta[1:4]), np.exp(theta[4])
    k = sf ** 2 * matern52(x, x, ls) + (sn ** 2 + 1e-10) * np.eye(len(x))
    chol = cho_factor(k, lower=True)
    return GP(x=np.asarray(x, float), theta=np.asarray(theta, float), mu=mu, sd=sd, alpha=cho_solve(chol, ys), chol=chol)


def gp_fit(x, y, starts=STARTS, theta0=None):
    """GP mit Hyperparametern aus der Maximierung der Log-Randwahrscheinlichkeit (L-BFGS-B ab festen Startpunkten, bestes Ergebnis); mit `theta0` nur ein Start (schneller, für das aktive Nachsampeln)."""
    y = np.asarray(y, float)
    ys = (y - y.mean()) / (y.std() + 1e-12)
    best, best_val = None, np.inf
    for t0 in ([np.asarray(theta0, float)] if theta0 is not None else starts):
        res = minimize(lambda t: -log_marginal_likelihood(t, x, ys), t0, method="L-BFGS-B", bounds=BOUNDS)
        if res.fun < best_val:
            best, best_val = res.x, res.fun
    return gp_build(x, y, best)


# --- Neuronales Netz -------------------------------------------------------------------------------------------------------------------------------

def nn_shapes(hidden):
    return [(3, hidden), (hidden,), (hidden, hidden), (hidden,), (hidden, 1), (1,)]


def nn_unpack(w, hidden):
    out, i = [], 0
    for shape in nn_shapes(hidden):
        size = int(np.prod(shape))
        out.append(w[i:i + size].reshape(shape))
        i += size
    return out


def nn_init(hidden, seed):
    """Glorot-Startgewichte aus einem festen Seed (numpy-Generator), Achsenabschnitte null."""
    rng = np.random.default_rng(seed)
    parts = []
    for shape in nn_shapes(hidden):
        if len(shape) == 2:
            lim = np.sqrt(6.0 / (shape[0] + shape[1]))
            parts.append(rng.uniform(-lim, lim, shape).ravel())
        else:
            parts.append(np.zeros(shape).ravel())
    return np.concatenate(parts)


def nn_forward(w, x, hidden):
    w1, b1, w2, b2, w3, b3 = nn_unpack(w, hidden)
    h1 = np.tanh(x @ w1 + b1)
    h2 = np.tanh(h1 @ w2 + b2)
    return (h2 @ w3 + b3).ravel(), (h1, h2)


def nn_loss_grad(w, x, y, hidden, alpha):
    """Mittlerer quadratischer Fehler plus alpha·Σw² (Gewichte, nicht Achsenabschnitte) und sein Gradient (Rückwärtsableitung von Hand)."""
    w1, b1, w2, b2, w3, b3 = nn_unpack(w, hidden)
    pred, (h1, h2) = nn_forward(w, x, hidden)
    n = len(x)
    err = pred - y
    loss = float(np.mean(err ** 2)) + alpha * float(sum((m ** 2).sum() for m in (w1, w2, w3)))
    d3 = (2.0 / n) * err[:, None]
    g_w3 = h2.T @ d3 + 2 * alpha * w3
    g_b3 = d3.sum(axis=0)
    d2 = (d3 @ w3.T) * (1 - h2 ** 2)
    g_w2 = h1.T @ d2 + 2 * alpha * w2
    g_b2 = d2.sum(axis=0)
    d1 = (d2 @ w2.T) * (1 - h1 ** 2)
    g_w1 = x.T @ d1 + 2 * alpha * w1
    g_b1 = d1.sum(axis=0)
    return loss, np.concatenate([g.ravel() for g in (g_w1, g_b1, g_w2, g_b2, g_w3, g_b3)])


@dataclass
class NN:
    w: np.ndarray
    hidden: int
    mu: float
    sd: float

    def predict(self, xs):
        return nn_forward(self.w, np.asarray(xs, float), self.hidden)[0] * self.sd + self.mu


def nn_fit(x, y, hidden=32, alpha=1e-3, seed=0, max_iter=1500):
    """NN-Fit: standardisierte Ausgabe, Vollbatch-L-BFGS-B ab Glorot-Start mit festem Seed."""
    y = np.asarray(y, float)
    mu, sd = float(y.mean()), float(y.std()) + 1e-12
    ys = (y - mu) / sd
    res = minimize(nn_loss_grad, nn_init(hidden, seed), args=(np.asarray(x, float), ys, hidden, alpha), jac=True, method="L-BFGS-B", options={"maxiter": max_iter})
    return NN(w=res.x, hidden=hidden, mu=mu, sd=sd)


# --- Modelle auf der Gesamtzeit (in Logarithmen) ---------------------------------------------------------------------------------------------------

def fit_log_gp(x, w, theta0=None):
    """GP auf ln W; Vorhersage in W (Mittel: exp des log-Mittels) mit log-Standardabweichung."""
    return gp_fit(x, np.log(w), theta0=theta0)


def fit_hybrid(x, w, physics, theta0=None):
    """Hybrid: GP auf dem Rest ln W − ln W_physik; Vorhersage W = W_physik · exp(Rest)."""
    return gp_fit(x, np.log(w) - np.log(physics), theta0=theta0)


def relative_errors(pred, truth):
    """|vorhergesagt / wahr − 1| je Punkt."""
    return np.abs(np.asarray(pred, float) / np.asarray(truth, float) - 1.0)


def error_summary(pred, truth):
    """Mittlerer, mittlerer 90-%-Quantil- und größter relativer Fehler."""
    e = relative_errors(pred, truth)
    return {"mean": float(e.mean()), "p90": float(np.quantile(e, 0.9)), "max": float(e.max())}
