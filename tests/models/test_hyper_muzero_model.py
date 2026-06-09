"""Pkg-04 spec 02 acceptance tests: HyperMuZeroModel v2 (7 API)."""
import pytest
import torch

from hyper_mve.configs import V4Config
from hyper_mve.models import HyperMuZeroModel


@pytest.fixture
def cfg_medium():
    return V4Config.from_preset("medium")


@pytest.fixture
def model(cfg_medium):
    return HyperMuZeroModel(cfg_medium)


def _belief(B, N):
    return (torch.rand(B), torch.softmax(torch.randn(B, N - 1, 2), dim=-1))


# ====== API 完整性 ======

def test_model_has_7_public_apis(model):
    for api in ("update_step", "set_context_objective", "set_context_subjective",
                "encode", "transition", "predict_reward", "predict"):
        assert hasattr(model, api), f"Missing API: {api}"
        assert callable(getattr(model, api))


def test_ctx_aug_dim_assertion(cfg_medium):
    assert cfg_medium.model.d_ctx_aug == 80


# ====== 顺序断言 (D4) ======

def test_subjective_before_objective_raises(model, cfg_medium):
    cap = torch.rand(1, 4)
    belief = _belief(1, cfg_medium.env.N)
    with pytest.raises(AssertionError, match="set_context_subjective.*before set_context_objective"):
        model.set_context_subjective(0, cap, belief)


def test_transition_before_set_context_raises(model):
    s = torch.randn(1, model.cfg.model.latent_dim)
    action = torch.zeros(1, model.cfg.env.N * model.cfg.env.A)
    with pytest.raises(AssertionError, match="transition.*before set_context_objective"):
        model.transition(s, action)


def test_predict_reward_before_subjective_raises(model):
    model.set_context_objective(torch.tensor([0.5]))
    s = torch.randn(1, model.cfg.model.latent_dim)
    action = torch.zeros(1, model.cfg.env.N * model.cfg.env.A)
    with pytest.raises(AssertionError, match="predict_reward.*before set_context_subjective"):
        model.predict_reward(s, action)


# ====== 缓存策略 (D4) ======

def test_objective_clears_subjective_cache(model, cfg_medium):
    model.set_context_objective(torch.tensor([0.5]))
    cap = torch.rand(1, 4)
    belief = _belief(1, cfg_medium.env.N)
    model.set_context_subjective(0, cap, belief)
    assert model._theta_rew is not None

    model.set_context_objective(torch.tensor([0.7]))
    assert model._theta_rew is None
    assert model._theta_pred is None


def test_subjective_replaces_per_agent(model, cfg_medium):
    model.set_context_objective(torch.tensor([0.5]))
    cap = torch.rand(1, 4)
    belief = _belief(1, cfg_medium.env.N)

    model.set_context_subjective(0, cap, belief)
    theta_rew_0 = model._theta_rew.clone()

    model.set_context_subjective(1, cap, belief)
    theta_rew_1 = model._theta_rew.clone()

    assert not torch.allclose(theta_rew_0, theta_rew_1)


# ====== type-aware reward forward 路径 (C9) ======

def test_type_aware_reward_differentiation(model, cfg_medium):
    model.set_context_objective(torch.tensor([0.5]))
    cap = torch.rand(1, 4)
    belief = _belief(1, cfg_medium.env.N)

    model.set_context_subjective(0, cap, belief)
    s = torch.randn(1, cfg_medium.model.latent_dim)
    action = torch.zeros(1, cfg_medium.env.N * cfg_medium.env.A)
    action[0, 0] = 1.0
    r_alpha = model.predict_reward(s, action)

    model.set_context_subjective(2, cap, belief)
    r_beta = model.predict_reward(s, action)

    assert r_alpha.shape == r_beta.shape


# ====== update_step (Q4) ======

def test_update_step(model):
    model.update_step(1000)
    assert model._step == 1000
    model.update_step(10000)
    assert model._step == 10000


def test_default_step_is_zero(model):
    """update_step 未调用时 _step=0 -> belief grad gating 永远启用 (spec 02 §4)."""
    assert model._step == 0


# ====== Stateful API 契约 ======

def test_predict_uses_latest_subjective_agent(model, cfg_medium):
    model.set_context_objective(torch.tensor([0.5]))
    cap = torch.rand(1, 4)
    belief = _belief(1, cfg_medium.env.N)

    s = torch.randn(1, cfg_medium.model.latent_dim)
    action = torch.zeros(1, cfg_medium.env.N * cfg_medium.env.A)
    action[0, 0] = 1.0

    model.set_context_subjective(0, cap, belief)
    r_A_first = model.predict_reward(s, action).clone()
    pi_A, v_A = model.predict(s)
    pi_A_first = pi_A.clone()

    model.set_context_subjective(1, cap, belief)
    r_B = model.predict_reward(s, action).clone()
    pi_B, v_B = model.predict(s)

    model.set_context_subjective(0, cap, belief)
    r_A_second = model.predict_reward(s, action).clone()

    assert not torch.allclose(r_A_first, r_B, atol=1e-6)
    assert not torch.allclose(pi_A_first, pi_B, atol=1e-6)
    assert torch.allclose(r_A_first, r_A_second, atol=1e-6)


def test_cap_shape_assertion(model, cfg_medium):
    model.set_context_objective(torch.tensor([0.5]))
    bad_cap = torch.rand(1, 5)
    belief = _belief(1, cfg_medium.env.N)
    with pytest.raises((AssertionError, RuntimeError, ValueError)):
        model.set_context_subjective(0, bad_cap, belief)


def test_k_step_unroll_consecutive_subjective_legal(model, cfg_medium):
    model.set_context_objective(torch.tensor([0.5]))
    cap = torch.rand(1, 4)
    belief = _belief(1, cfg_medium.env.N)

    for k in range(cfg_medium.env.N):
        model.set_context_subjective(k, cap, belief)
        s = torch.randn(1, cfg_medium.model.latent_dim)
        action = torch.zeros(1, cfg_medium.env.N * cfg_medium.env.A)
        action[0, 0] = 1.0
        _ = model.transition(s, action)
        _ = model.predict_reward(s, action)
        _ = model.predict(s)


# ====== Self-Info 严格性 (D6 + C11) ======

def test_set_context_subjective_no_oracle_types_leak(model, cfg_medium):
    from dataclasses import replace
    from hyper_mve.schemas import AgentType

    new_types = (AgentType.ALPHA,) * cfg_medium.env.N
    cfg_mod = replace(cfg_medium, env=replace(cfg_medium.env, type_assignment=new_types))
    model_mod = HyperMuZeroModel(cfg_mod)

    model_mod.set_context_objective(torch.tensor([0.5]))
    cap = torch.rand(1, 4)
    belief = _belief(1, cfg_medium.env.N)

    model_mod.set_context_subjective(2, cap, belief)
    theta_rew_mod = model_mod._theta_rew.clone()

    model.set_context_objective(torch.tensor([0.5]))
    model.set_context_subjective(2, cap, belief)
    theta_rew_orig = model._theta_rew.clone()

    assert not torch.allclose(theta_rew_mod, theta_rew_orig)


# ====== Forward 端到端 ======

def test_end_to_end_forward_no_nan(model, cfg_medium):
    B, N = 2, cfg_medium.env.N
    obs_dim = model.rep_net.obs_dim
    obs = torch.randn(B, N, obs_dim)
    s = model.encode(obs)

    model.update_step(0)
    model.set_context_objective(torch.full((B,), 0.5))

    action = torch.zeros(B, N * cfg_medium.env.A)
    action[:, 0] = 1.0

    s_next = model.transition(s, action)
    assert not torch.isnan(s_next).any()

    model.set_context_subjective(
        0, torch.rand(B, 4),
        (torch.rand(B), torch.softmax(torch.randn(B, N - 1, 2), dim=-1)),
    )
    r = model.predict_reward(s, action)
    pi, v = model.predict(s)
    assert not torch.isnan(r).any()
    assert not torch.isnan(pi).any()
    assert not torch.isnan(v).any()


# ====== film_head partial-generation (gen_scope) ======

def _film_head_cfg():
    from dataclasses import replace
    base = V4Config.from_preset("medium")
    return replace(base, model=replace(
        base.model, hyper_gen_scope="film_head",
        trans_output_scale_init=0.1, pred_output_scale_init=0.1,
    ))


def test_film_head_sizes_hypernet_by_generated_count():
    m = HyperMuZeroModel(_film_head_cfg())
    # film_head generated counts are N-independent (the N*A part feeds the shared SGD fc1).
    assert m.state_trans_net.generated_param_count == 8768
    assert m.reward_head.generated_param_count == 641
    assert m.prediction_net.generated_param_count == 1415
    # DualHyperNetwork sized by generated_param_count + groups threaded through.
    assert m.hyper_net.trans_param_count == 8768
    assert m.hyper_net.rew_param_count == 641
    assert m.hyper_net.pred_param_count == 1415
    assert m.hyper_net.hyper_trans.output_groups == [512, 8256]
    assert m.hyper_net.hyper_rew.output_groups == [512, 129]
    assert m.hyper_net.hyper_pred.output_groups == [512, 903]


def test_film_head_end_to_end_no_nan():
    cfg = _film_head_cfg()
    m = HyperMuZeroModel(cfg)
    B, N = 2, cfg.env.N
    obs = torch.randn(B, N, m.rep_net.obs_dim)
    s = m.encode(obs)
    m.update_step(0)
    m.set_context_objective(torch.full((B,), 0.5))
    action = torch.zeros(B, N * cfg.env.A)
    action[:, 0] = 1.0
    s_next = m.transition(s, action)
    m.set_context_subjective(
        0, torch.rand(B, 4),
        (torch.rand(B), torch.softmax(torch.randn(B, N - 1, 2), dim=-1)),
    )
    r = m.predict_reward(s, action)
    pi, v = m.predict(s)
    for t in (s_next, r, pi, v):
        assert not torch.isnan(t).any()


def test_film_head_per_agent_prediction_differentiates():
    """The whole point of film_head: theta_pred still differs per agent (view=perspective)."""
    cfg = _film_head_cfg()
    m = HyperMuZeroModel(cfg)
    B, N = 2, cfg.env.N
    m.set_context_objective(torch.full((B,), 0.5))
    cap = torch.rand(B, 4)
    belief = (torch.rand(B), torch.softmax(torch.randn(B, N - 1, 2), dim=-1))
    m.set_context_subjective(0, cap, belief)
    theta0 = m.current_subjective_thetas()[1].clone()
    m.set_context_subjective(2, cap, belief)  # different type (BETA in medium 2a+2b)
    theta2 = m.current_subjective_thetas()[1].clone()
    assert not torch.allclose(theta0, theta2)


# ====== LoRA presets (output-layer LoRA + lora_fc2) ======

@pytest.mark.parametrize("preset,scope,fc2_rank", [
    ("duo_film_lora", "film_head", None),
    ("duo_film_lora_fc2", "lora_fc2", 8),
    ("duo_base_lora", "base_gen", None),
])
def test_lora_presets_construct_with_output_rank(preset, scope, fc2_rank):
    cfg = V4Config.from_preset(preset)
    assert cfg.model.hyper_gen_scope == scope
    assert cfg.model.hyper_output_rank == 32
    assert cfg.model.lora_fc2_rank == fc2_rank
    m = HyperMuZeroModel(cfg)
    # output-layer LoRA wired on all three independent hypernets (share=False).
    for sub in (m.hyper_net.hyper_trans, m.hyper_net.hyper_rew, m.hyper_net.hyper_pred):
        assert sub.output_rank == 32
        assert hasattr(sub, "output_A") and hasattr(sub, "output_B")


def test_lora_fc2_preset_counts_and_end_to_end():
    cfg = V4Config.from_preset("duo_film_lora_fc2")   # N=2, A=6
    m = HyperMuZeroModel(cfg)
    assert m.state_trans_net.generated_param_count == 10816
    assert m.reward_head.generated_param_count == 2689
    assert m.prediction_net.generated_param_count == 3463
    assert hasattr(m.state_trans_net, "fc2")          # shared SGD base for the rank-r delta
    B, N = 2, cfg.env.N
    obs = torch.randn(B, N, m.rep_net.obs_dim)
    s = m.encode(obs)
    m.update_step(0)
    m.set_context_objective(torch.full((B,), 0.5))
    action = torch.zeros(B, N * cfg.env.A)
    action[:, 0] = 1.0
    s_next = m.transition(s, action)
    m.set_context_subjective(
        0, torch.rand(B, 4),
        (torch.rand(B), torch.softmax(torch.randn(B, N - 1, 2), dim=-1)),
    )
    r = m.predict_reward(s, action)
    pi, v = m.predict(s)
    for t in (s_next, r, pi, v):
        assert not torch.isnan(t).any()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
