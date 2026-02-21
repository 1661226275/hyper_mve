"""
Hyper-MuZero Configuration (v4.6)

All hyperparameters for environment, model, and training.
Aligned with DESIGN_DOC_FINAL.md (v4.6).
"""


class BaseConfig:
    """Shared config for all experiments."""

    # ── Environment ─────────────────────────────────────────────
    episode_limit = 25
    num_agents = 4          # 2 fixed adv + 1 variable + 1 prey
    num_actions = 5         # discrete: stop/left/right/down/up

    # obs dims per agent (from Phase 1 test):
    # Agent 0,1,2 (adversaries): 16;  Agent 3 (prey): 14
    obs_dim_n = [16, 16, 16, 14]
    joint_obs_dim = sum(obs_dim_n)   # 62
    # Joint action: one-hot per agent, concatenated
    # N_agents * num_actions = 4 * 5 = 20
    joint_action_dim = num_agents * num_actions  # 20

    # ── Model ───────────────────────────────────────────────────
    latent_dim = 64         # RepresentationNet output
    hidden_dim = 128        # hidden layers in most networks
    rule_emb_dim = 16       # rule embedding dimension
    id_emb_dim = 16         # agent id embedding dimension

    # ── Training (MuZero-style) ──────────────────────────────────
    max_train_steps = 200000
    lr = 1e-4
    lr_min = 5e-6               # [v4.4] CosineAnnealingLR eta_min
    adam_eps = 1e-5             # [v4.4] Adam optimizer epsilon
    gamma = 0.95
    batch_size = 256            # [v4.4] 从 512 降至 256，降低 replay ratio
    buffer_size = 5000
    min_buffer_size = 1000      # [v4.4] Warmup guard: don't train until buffer >= this
    episodes_per_iter = 8      # [v4.4] 每轮收集 episode 数 (原 4, 硬编码)
    train_steps_per_iter = 8    # [v4.4] 每轮训练步数 (原 16, 硬编码)
    grad_clip = 10.0
    unroll_K = 5            # MuZero unroll steps
    n_step = 5              # N-step return bootstrap

    # Loss weights
    w_policy = 1.0
    w_value = 0.25          # [v4.0 建议值: 0.25 (EfficientZero标准), 原: 1.0]
    w_reward = 1.0
    w_consist = 0.5         # [v4.1 调整] 从 2.0 降至 0.5，防止初期坍缩 (l_pol长期不降)

    # v4.0 HyperMuZero Improvements
    proj_dim = 64           # Projector 投影维度
    w_context = 0.01        # Context Hinge Loss 权重 (Exp3)
    target_context_std = 0.1  # Hinge Variance 阈值 (Exp3)

    # MVE Planner (v4.6: per-agent coordinate descent + CRN)
    mve_samples = 50        # samples per agent (split across A candidates = 10 scenarios)
    mve_depth = 5           # rollout depth
    mve_temperature = 1.0   # softmax temperature for π_mve

    # Reward/Value scaling epsilon
    reward_scale_eps = 0.001

    # Exploration (epsilon-greedy for discrete actions)
    epsilon_init = 1.0          # [v4.6] restored: no PG, standard ε-greedy
    epsilon_min = 0.05
    epsilon_decay_steps = 28000  # [v4.6] restored: standard decay schedule

    # Evaluation
    evaluate_freq = 500    # evaluate every N steps
    evaluate_episodes = 30

    # Logging
    log_dir = './runs'
    save_dir = './checkpoints'

    # Device
    device = 'auto'         # 'auto', 'cpu', or 'cuda'

    # ── Exp2: Oracle-HyperMuZero ─────────────────────────────────
    hyper_hidden_dims = [256, 256]

    # ── Exp3: Infer-HyperMuZero ──────────────────────────────────
    gru_hidden_size = 128
    trajectory_window = 10   # sliding window for GRU input

    # ── v4.4: Target Network (EMA) ────────────────────────────────
    ema_tau = 0.99              # EMA coefficient: target = tau*target + (1-tau)*online

    # ── Adversarial Freeze (Phased Perspective Sampling) ────────
    freeze_enabled = True           # Total switch: enable/disable phased freezing
    freeze_warmup_steps = 4000      # Full agent training before freezing kicks in
    freeze_phase_steps = 4000     # Steps per phase (symmetric: Hunter phase = Prey phase)
    freeze_hunter_agents = [0, 1, 2]  # Agents active during Hunter phase
    freeze_prey_agents = [3]          # Agents active during Prey phase

    # ── Phase 5: ChunkedHMLP (hypnettorch) ───────────────────────
    chunk_alpha = 10            # chunk_size = alpha * hidden_size
    chunk_emb_size = 8          # per-chunk embedding dimension
    chunk_budget_factor = 1.0   # HyperNet params ≈ budget_factor * target params
    chunk_hyperfan_init = True  # use principled hyperfan initialization
