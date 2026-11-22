"""Fast CPU tests: run with ``pytest -q`` (about 30 seconds)."""

import numpy as np
import pytest
import torch

from lore import LoReConfig, build_basis, count_trainable, evaluate_users, fit_fewshot, fit_joint
from lore.core import user_balanced_bt_loss
from lore.init import adapt, initial_weights, select_by_support
from lore.models import FixedAnchorResidualBasis
from lore.projections import fit_pca, random_orthogonal
from lore.synthetic import SyntheticConfig, make_synthetic
from lore.utils import pack_users

D = 4096


@pytest.fixture(scope="module")
def projections():
    X = torch.randn(600, D)
    p = fit_pca(X, n_components=512)
    p["random512"] = random_orthogonal(D, 512)
    return p


def test_parameter_counts_match_report(projections):
    expected = {
        "pca5": 0, "pca512_linear": 2560, "random512_linear": 2560, "linear": 20480,
        "mlp8": 32821, "mlp64": 262533, "residual16": 86096, "fixed_anchor_residual": 16384,
    }
    anchor = torch.randn(D)
    for name, n in expected.items():
        m = build_basis(name, D, 5, projections, anchor)
        assert count_trainable(m) == n, name
        out = m(torch.randn(3, D))
        assert out.shape == (3, 5), name


@pytest.mark.parametrize("name", ["linear", "mlp8", "mlp64", "residual16", "fixed_anchor_residual", "pca5"])
def test_bases_are_odd(name, projections):
    m = build_basis(name, D, 5, projections, torch.randn(D))
    x = torch.randn(8, D)
    with torch.no_grad():
        assert torch.allclose(m(x), -m(-x), atol=1e-4)


def test_naive_mlp_is_not_odd():
    m = build_basis("naive_mlp8", D, 5)
    x = torch.randn(8, D)
    with torch.no_grad():
        assert not torch.allclose(m(x), -m(-x), atol=1e-3)


def test_fixed_anchor_directions_orthogonal_and_normed():
    a = torch.randn(D)
    m = FixedAnchorResidualBasis(a, 4)
    d = m.directions()
    assert torch.allclose(d @ (a / a.norm()), torch.zeros(4), atol=1e-4)
    assert torch.allclose(d.norm(dim=1), a.norm().expand(4), rtol=1e-4)


def test_user_balanced_loss_weights_users_equally():
    s = torch.tensor([0.0, 0.0, 0.0, 10.0])
    u = torch.tensor([0, 0, 0, 1])
    # user 0 mean loss = log 2, user 1 ~ 0
    assert user_balanced_bt_loss(s, u, 2).item() == pytest.approx(np.log(2), abs=1e-3)


def test_pack_users_roundtrip():
    X, idx, order = pack_users({"a": torch.ones(2, 3), "b": torch.zeros(3, 3)})
    assert X.shape == (5, 3) and idx.tolist() == [0, 0, 1, 1, 1] and order == ["a", "b"]


@pytest.fixture(scope="module")
def synth():
    return make_synthetic(SyntheticConfig(D=64, n_seen=80, n_unseen=40, n_test=200, seed=1))


def test_lore_learns_synthetic_low_rank_users(synth):
    cfg = LoReConfig(K=4, joint_steps=300, fewshot_steps=200, joint_lr=0.05, logit_scale=1.0, seed=0)
    X, u, _ = pack_users(synth["seen_train"])
    res = fit_joint(build_basis("linear", 64, 4), X, u, len(synth["seen_train"]), cfg)
    Xt, ut, _ = pack_users(synth["seen_test"])
    seen = evaluate_users(res["basis"], Xt, ut, res["W_logits"], cfg.logit_scale)["mean_accuracy"]
    Xf, uf, _ = pack_users(synth["unseen_fewshot"])
    few = fit_fewshot(res["basis"], Xf, uf, len(synth["unseen_fewshot"]), cfg)
    Xq, uq, _ = pack_users(synth["unseen_test"])
    unseen = evaluate_users(res["basis"], Xq, uq, few["W_logits"], cfg.logit_scale)["mean_accuracy"]
    assert seen > 0.7 and unseen > 0.65


def test_orientation_shortcut_is_detected_by_symmetric_eval(synth):
    """A non-odd model reaches ~100% standard accuracy but ~50% symmetric accuracy."""
    cfg = LoReConfig(K=4, joint_steps=300, joint_lr=0.01, logit_scale=1.0, seed=0)
    X, u, _ = pack_users(synth["seen_train"])
    res = fit_joint(build_basis("naive_mlp8", 64, 4), X, u, len(synth["seen_train"]), cfg)
    Xt, ut, _ = pack_users(synth["seen_test"])
    std = evaluate_users(res["basis"], Xt, ut, res["W_logits"], 1.0)["mean_accuracy"]
    sym = evaluate_users(res["basis"], Xt, ut, res["W_logits"], 1.0, symmetric=True)["mean_accuracy"]
    assert std > 0.95
    assert sym < 0.7


def test_fewshot_converges_from_any_initialisation():
    """Convexity in w: different starts reach (almost) the same query accuracy."""
    torch.manual_seed(0)
    K, users = 5, 30
    w_true = torch.distributions.Dirichlet(torch.full((K,), 0.3)).sample((users,))
    Z = torch.randn(users, 400, K) * 50
    sign = torch.where((Z * w_true.unsqueeze(1)).sum(-1, keepdim=True) > 0, 1.0, -1.0)
    Z = Z * sign  # orient as chosen - rejected under the true weights
    Zs, Zq = Z[:, :60], Z[:, 60:]
    W_seen = torch.distributions.Dirichlet(torch.full((K,), 0.3)).sample((100,))
    finals = []
    for m in ["random_logits", "uniform", "population_mean", "nearest_seen_user"]:
        w0 = initial_weights(m, Zs, W_seen, seed=0)
        finals.append(adapt(Zs, Zq, w0, steps=300, eval_at=(300,))[-1]["query_accuracy"])
    assert max(finals) - min(finals) < 0.01


def test_select_by_support_picks_true_weights():
    torch.manual_seed(0)
    K = 4
    cands = torch.eye(K)
    Z = torch.randn(10, 50, K)
    Z[:, :, 2] = Z[:, :, 2].abs() * 5  # only basis 2 explains every pair
    _, idx = select_by_support(Z, cands)
    assert (idx == 2).all()
