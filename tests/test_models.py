"""GP und NN von Hand und gegen unabhängige Orakel (scikit-learn für das GP bei festen Hyperparametern, Differenzenquotienten für den Gradienten des NN)."""

import math

import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

import sur_models as M


def test_features_map_the_design_space_onto_the_unit_cube():
    x = M.features([0.3, 0.95, 0.6], [0.0, 0.4, 0.2], [0.25, 4.0, 1.0])
    assert x[0] == pytest.approx([0, 0, 0]) and x[1] == pytest.approx([1, 1, 1])
    assert x[2][2] == pytest.approx(0.5) and x[2][1] == pytest.approx(0.5)                         # cs² = 1 liegt in der Mitte von log₂ (−2 … 2), r = 0.2 in der Mitte von 0 … 0.4
    assert x[2][0] == pytest.approx((-math.log(0.4) + math.log(0.7)) / (-math.log(0.05) + math.log(0.7)))


def test_matern52_by_hand():
    """d = 1, ein Längenmaß 1: (1 + √5 + 5/3)·e^{−√5} = 4.9027347 · 0.1069... ; k(x, x) = 1."""
    k = M.matern52(np.array([[0.0]]), np.array([[1.0]]), np.array([1.0]))[0, 0]
    assert k == pytest.approx((1 + math.sqrt(5) + 5 / 3) * math.exp(-math.sqrt(5)))
    assert M.matern52(np.array([[0.3, 0.2]]), np.array([[0.3, 0.2]]), np.array([1.0, 2.0]))[0, 0] == pytest.approx(1.0)


def test_matern52_length_scale_acts_per_dimension():
    a, b = np.array([[0.0, 0.0]]), np.array([[0.5, 0.5]])
    same = M.matern52(a, b, np.array([1.0, 1.0]))[0, 0]
    long_second = M.matern52(a, b, np.array([1.0, 100.0]))[0, 0]
    assert long_second > same
    assert long_second == pytest.approx(M.matern52(a, np.array([[0.5, 0.0]]), np.array([1.0, 100.0]))[0, 0], abs=1e-4)      # die zweite Eingabe spielt fast keine Rolle


def test_kernel_matrix_is_symmetric_and_positive_semidefinite():
    x = np.random.default_rng(0).uniform(size=(30, 3))
    k = M.matern52(x, x, np.array([0.3, 0.5, 2.0]))
    assert np.allclose(k, k.T) and np.linalg.eigvalsh(k).min() > -1e-9


def _sk_gp(theta, x, ys):
    sf, ls, sn = np.exp(theta[0]), np.exp(theta[1:4]), np.exp(theta[4])
    kernel = ConstantKernel(sf ** 2, "fixed") * Matern(length_scale=ls, nu=2.5, length_scale_bounds="fixed") + WhiteKernel(sn ** 2, "fixed")
    return GaussianProcessRegressor(kernel, optimizer=None, normalize_y=False, alpha=1e-10).fit(x, ys)


def test_log_marginal_likelihood_matches_scikit_learn():
    rng = np.random.default_rng(1)
    x = rng.uniform(size=(25, 3))
    ys = np.sin(4 * x[:, 0]) + 0.3 * x[:, 1] + 0.05 * rng.normal(size=25)
    ys = (ys - ys.mean()) / ys.std()
    theta = np.log([1.7, 0.4, 0.9, 3.0, 0.08])
    assert M.log_marginal_likelihood(theta, x, ys) == pytest.approx(_sk_gp(theta, x, ys).log_marginal_likelihood_value_, abs=1e-6)


def test_gp_prediction_matches_scikit_learn():
    rng = np.random.default_rng(2)
    x = rng.uniform(size=(20, 3))
    y = 5.0 + 2.0 * np.sin(3 * x[:, 0]) + x[:, 1]
    theta = np.log([1.5, 0.5, 1.0, 2.0, 0.05])
    gp = M.gp_build(x, y, theta)
    ys = (y - y.mean()) / (y.std() + 1e-12)
    xs = rng.uniform(size=(15, 3))
    mean, sd = _sk_gp(theta, x, ys).predict(xs, return_std=True)
    pm, ps = gp.predict(xs)
    assert pm == pytest.approx(mean * gp.sd + gp.mu, abs=1e-8)
    assert ps == pytest.approx(sd * gp.sd, rel=1e-5)


def test_gp_interpolates_with_tiny_noise_and_returns_the_prior_far_away():
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    y = np.array([1.0, 3.0, 2.0, 5.0])
    gp = M.gp_build(x, y, np.log([1.0, 0.5, 0.5, 0.5, 1e-4]))
    mean, sd = gp.predict(x)
    assert mean == pytest.approx(y, abs=1e-3) and (sd < 0.01 * y.std() + 1e-3).all()
    far_mean, far_sd = gp.predict(np.array([[50.0, 50.0, 50.0]]))
    assert far_mean[0] == pytest.approx(y.mean(), abs=1e-6) and far_sd[0] == pytest.approx(gp.sd * math.sqrt(1.0 + 1e-8), rel=1e-6)


def test_gp_uncertainty_is_smaller_near_data_than_far_from_it():
    x = np.array([[0.1, 0.5, 0.5], [0.15, 0.5, 0.5], [0.2, 0.5, 0.5]])
    gp = M.gp_build(x, np.array([1.0, 1.2, 1.1]), np.log([1.0, 0.3, 0.3, 0.3, 0.05]))
    _, sd = gp.predict(np.array([[0.15, 0.5, 0.5], [0.9, 0.5, 0.5]]))
    assert sd[0] < sd[1]


def test_gp_fit_maximises_the_marginal_likelihood_and_finds_the_irrelevant_input():
    """y hängt nur von der ersten und zweiten Eingabe ab; die dritte bekommt ein großes Längenmaß (ARD). Die angepassten Hyperparameter haben eine bessere Randwahrscheinlichkeit als jeder Startwert."""
    rng = np.random.default_rng(3)
    x = rng.uniform(size=(60, 3))
    y = np.sin(4 * x[:, 0]) + 0.5 * x[:, 1] + 0.01 * rng.normal(size=60)
    gp = M.gp_fit(x, y)
    ys = (y - y.mean()) / y.std()
    best_start = max(M.log_marginal_likelihood(t, x, ys) for t in M.STARTS)
    assert M.log_marginal_likelihood(gp.theta, x, ys) >= best_start - 1e-9
    ls = np.exp(gp.theta[1:4])
    assert ls[2] > 5 * ls[0] and ls[2] >= 10.0                                                    # die dritte Eingabe wird ignoriert (Längenmaß am oberen Rand)
    xs = rng.uniform(size=(100, 3))
    truth = np.sin(4 * xs[:, 0]) + 0.5 * xs[:, 1]
    assert np.abs(gp.predict(xs)[0] - truth).max() < 0.1


def test_gp_fit_with_a_given_start_uses_only_that_start():
    rng = np.random.default_rng(4)
    x = rng.uniform(size=(20, 3))
    y = x[:, 0] ** 2 + x[:, 1]
    full = M.gp_fit(x, y)
    warm = M.gp_fit(x, y, theta0=full.theta)
    assert M.log_marginal_likelihood(warm.theta, x, (y - y.mean()) / y.std()) >= M.log_marginal_likelihood(full.theta, x, (y - y.mean()) / y.std()) - 1e-6


def test_hybrid_with_a_constant_residual_reproduces_the_physics_times_the_constant():
    rng = np.random.default_rng(5)
    x = rng.uniform(size=(12, 3))
    physics = 10 * np.exp(rng.uniform(size=12))
    w = 1.07 * physics
    gp = M.fit_hybrid(x, w, physics)
    xs = rng.uniform(size=(5, 3))
    mean, _ = gp.predict(xs)
    assert np.exp(mean) == pytest.approx(np.full(5, 1.07), rel=1e-6)


def test_nn_forward_by_hand():
    """Ein verdecktes Neuron je Schicht: x = (0.5, 0, 0), W₁ = (1, 0, 0)ᵀ, b₁ = 0, W₂ = 2, b₂ = 0.1, W₃ = 1.5, b₃ = −0.2:
    h₁ = tanh 0.5, h₂ = tanh(2·h₁ + 0.1), Ausgabe 1.5·h₂ − 0.2."""
    w = np.concatenate([[1.0, 0.0, 0.0], [0.0], [2.0], [0.1], [1.5], [-0.2]])
    out, _ = M.nn_forward(w, np.array([[0.5, 0.0, 0.0]]), 1)
    h1 = math.tanh(0.5)
    h2 = math.tanh(2 * h1 + 0.1)
    assert out[0] == pytest.approx(1.5 * h2 - 0.2)


def test_nn_gradient_matches_finite_differences():
    rng = np.random.default_rng(6)
    x, y = rng.uniform(size=(9, 3)), rng.normal(size=9)
    w = M.nn_init(4, 1) + 0.1 * rng.normal(size=len(M.nn_init(4, 1)))
    loss, grad = M.nn_loss_grad(w, x, y, 4, 1e-2)
    num = np.zeros_like(w)
    for i in range(len(w)):
        e = np.zeros_like(w)
        e[i] = 1e-6
        num[i] = (M.nn_loss_grad(w + e, x, y, 4, 1e-2)[0] - M.nn_loss_grad(w - e, x, y, 4, 1e-2)[0]) / 2e-6
    assert grad == pytest.approx(num, rel=1e-5, abs=1e-8) and loss > 0


def test_nn_loss_by_hand_for_a_single_point_without_penalty():
    """Vorhersage 1.5·h₂ − 0.2 (siehe oben), Ziel 0: Verlust = Vorhersage²."""
    w = np.concatenate([[1.0, 0.0, 0.0], [0.0], [2.0], [0.1], [1.5], [-0.2]])
    pred = 1.5 * math.tanh(2 * math.tanh(0.5) + 0.1) - 0.2
    loss, _ = M.nn_loss_grad(w, np.array([[0.5, 0.0, 0.0]]), np.array([0.0]), 1, 0.0)
    assert loss == pytest.approx(pred ** 2)


def test_nn_init_is_glorot_and_reproducible():
    a, b, c = M.nn_init(8, 3), M.nn_init(8, 3), M.nn_init(8, 4)
    assert (a == b).all() and not (a == c).all()
    w1, b1, w2, b2, w3, b3 = M.nn_unpack(a, 8)
    assert np.abs(w1).max() <= math.sqrt(6 / 11) and np.abs(w2).max() <= math.sqrt(6 / 16) and (b1 == 0).all() and (b3 == 0).all() and len(a) == 3 * 8 + 8 + 64 + 8 + 8 + 1


def test_nn_fits_a_smooth_function_and_is_deterministic():
    rng = np.random.default_rng(7)
    x = rng.uniform(size=(80, 3))
    y = np.sin(3 * x[:, 0]) + 0.5 * x[:, 1] + 2.0
    nn = M.nn_fit(x, y, seed=1)
    xs = rng.uniform(size=(100, 3))
    assert np.abs(nn.predict(xs) - (np.sin(3 * xs[:, 0]) + 0.5 * xs[:, 1] + 2.0)).mean() < 0.05
    assert (M.nn_fit(x, y, seed=1).predict(xs) == nn.predict(xs)).all()


def test_error_summary_by_hand():
    s = M.error_summary([1.1, 0.9, 1.0], [1.0, 1.0, 1.0])
    assert s["mean"] == pytest.approx(0.2 / 3) and s["max"] == pytest.approx(0.1) and s["p90"] == pytest.approx(0.1)
    assert M.relative_errors([2.0], [1.0])[0] == pytest.approx(1.0)
