"""DualHyperNetwork v2 for Hyper-MuZero (Pkg-04 spec 01, Ch4.3).

v4 关键改动 (相对 v4.7 双路 (rule_emb, id_emb) + 单 forward):
    v4.7: __init__(rule_emb_dim, id_emb_dim, ...) + forward(rule_emb, id_emb)
    v4:   __init__(c_ctx_dim=16, ctx_aug_dim=80, ...) +
          forward_trans(c_ctx) -> theta_state       (客观通路, C1 物理转移上下文不变)
          forward_subjective(ctx_aug) -> (theta_rew, theta_pred)  (主观通路 per-agent)

保留 HyperNetMLP 结构不变 (L2 norm + learnable output_scale + small_init 稳定性技巧).

三个 HyperNetMLP 内部独立 (不共用 trunk):
    - hyper_trans: c_ctx (16) -> theta_state          <- 客观通路 (C1)
    - hyper_rew:   ctx_aug (80) -> theta_rew^i        <- 主观通路 per-agent (3 层加深)
    - hyper_pred:  ctx_aug (80) -> theta_pred^i       <- 主观通路 per-agent (可选 detach context, D5)

output_scale 三初值 (v4.7 经验保留, C8):
    - trans_output_scale_init = 0.01
    - rew_output_scale_init   = 0.1   <- v4.7 关键 (避免 init trap)
    - pred_output_scale_init  = 0.01
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from hyper_mve.utils.utils import orthogonal_init, small_init


def normalize_generated_output(raw, output_scale, norm_output, output_groups):
    """Apply HyperNet output normalization + learnable scale (shared helper).

    Used by both ``HyperNetMLP`` and ``SubjectiveHyperNet``'s heads so the math is
    defined once. Identical ops + ``1e-8`` epsilons to the original inline version.

    - ``norm_output=False``: ``raw * output_scale``.
    - ``output_groups is None``: whole-vector L2 norm (``||raw||=1``) then ``*scale``
      (FULL gen_scope legacy behavior — unchanged).
    - ``output_groups`` set: per-group RMS norm so each element's magnitude ≈ scale
      regardless of group dim (film_head / base_gen). Whole-vector L2 would pin a FiLM
      gamma element at ~scale/√dim (e.g. 0.1/√512 ≈ 4e-3 -> (1+γ)≈1 -> no modulation);
      RMS-normalizing the film segment separately keeps |γ| ≈ scale, and the weight
      segment starts near fan-in standard init.
    """
    if not norm_output:
        return raw * output_scale

    if output_groups is None:
        norms = torch.linalg.norm(raw, dim=-1, keepdim=True)
        return raw / (norms + 1e-8) * output_scale

    segs = []
    offset = 0
    for g in output_groups:
        seg = raw[:, offset:offset + g]
        rms = torch.sqrt(seg.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
        segs.append(seg / rms)
        offset += g
    return torch.cat(segs, dim=-1) * output_scale


class HyperNetMLP(nn.Module):
    """从 context embedding 生成 flat parameter vector (v4.7 保留, 结构不变).

    Architecture: context -> FC1 -> LN -> ReLU -> FC2 -> LN -> ReLU -> FC_out -> flat_params

    稳定性技巧 (spec 03 详述):
        - small_init(std=0.01) 在 output_layer 防初期权重爆炸
        - 可选 L2 normalize + learnable output_scale (norm_output=True): 固定方向分布,
          幅值 = output_scale, 解耦方向与幅值.
    """

    def __init__(self, input_dim, output_dim, hidden_dims=None, norm_output=True,
                 output_scale_init=0.01, output_groups=None):
        """
        Args:
            input_dim:         dimension of input context
            output_dim:        total number of parameters to generate
            hidden_dims:       list/tuple of hidden layer dimensions (default: (256, 256))
            norm_output:       whether to normalize the output (v4.0 stability trick)
            output_scale_init: [v4.7] initial value for learnable output_scale
                               (default 0.01; hyper_rew uses 0.1 to avoid initialization trap)
            output_groups:     [film_head] optional list[int] of contiguous segment sizes
                               (must sum to output_dim). When set, each segment is
                               RMS-normalized independently (per-element magnitude ≈
                               output_scale, dimension-independent) instead of the whole
                               vector being L2-normalized to unit norm. None (default) =
                               legacy whole-vector L2 norm (FULL gen_scope), unchanged.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = (256, 256)
        if output_groups is not None:
            assert sum(output_groups) == output_dim, (
                f"output_groups {output_groups} sum {sum(output_groups)} != "
                f"output_dim {output_dim}"
            )

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        self.trunk = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)

        # Initialize trunk with orthogonal, output with small weights
        for module in self.trunk:
            if isinstance(module, nn.Linear):
                orthogonal_init(module)
        small_init(self.output_layer, std=0.01)

        # [v4.0] L2 Norm + Scale stability trick
        self.norm_output = norm_output
        self.output_groups = output_groups
        self.output_scale = nn.Parameter(torch.tensor(float(output_scale_init)))

    def forward(self, context):
        """
        Args:
            context: (B, input_dim) -- context embedding
        Returns:
            flat_params: (B, output_dim) -- generated parameters
        """
        h = self.trunk(context)
        raw = self.output_layer(h)
        return normalize_generated_output(
            raw, self.output_scale, self.norm_output, self.output_groups
        )


class SubjectiveHyperNet(nn.Module):
    """Shared subjective generator (Idea 1): one trunk over ctx_aug -> two heads
    (theta_rew, theta_pred). Replaces the two independent hyper_rew/hyper_pred MLPs
    when ``share_subjective_trunk=True``. ``hyper_trans`` (objective) is unaffected.

    Each head keeps its OWN ``output_scale`` and ``output_groups`` so rew/pred retain
    their distinct init magnitudes + grouped-RMS norm. The shared trunk couples the
    two subjective representations (the intended effect).

    Detach (D5) under sharing: when ``detach_pred_context=True`` the pred head detaches
    the SHARED TRUNK OUTPUT (``h.detach()``), so value loss trains ONLY ``pred_head`` --
    not the shared trunk or context encoder. This is STRONGER than the unshared
    input-detach (which still trains hyper_pred's own trunk). When False, both heads +
    trunk train from both losses (the new ``duo_basegen`` preset uses False).
    """

    def __init__(
        self,
        ctx_aug_dim,
        rew_param_count,
        pred_param_count,
        hidden_dims=(256, 256, 256),
        norm_output=True,
        rew_output_scale_init=0.1,
        pred_output_scale_init=0.01,
        detach_pred_context=True,
        rew_output_groups=None,
        pred_output_groups=None,
    ):
        super().__init__()
        layers = []
        prev_dim = ctx_aug_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        self.trunk = nn.Sequential(*layers)
        self.rew_head = nn.Linear(prev_dim, rew_param_count)
        self.pred_head = nn.Linear(prev_dim, pred_param_count)

        # Same init scheme as HyperNetMLP: orthogonal trunk, small output heads.
        for module in self.trunk:
            if isinstance(module, nn.Linear):
                orthogonal_init(module)
        small_init(self.rew_head, std=0.01)
        small_init(self.pred_head, std=0.01)

        self.norm_output = norm_output
        self.detach_pred_context = detach_pred_context
        self.rew_output_groups = rew_output_groups
        self.pred_output_groups = pred_output_groups
        self.rew_output_scale = nn.Parameter(torch.tensor(float(rew_output_scale_init)))
        self.pred_output_scale = nn.Parameter(torch.tensor(float(pred_output_scale_init)))

    def forward(self, ctx_aug):
        """ctx_aug (B, ctx_aug_dim) -> (theta_rew, theta_pred)."""
        h = self.trunk(ctx_aug)
        theta_rew = normalize_generated_output(
            self.rew_head(h), self.rew_output_scale,
            self.norm_output, self.rew_output_groups,
        )
        h_pred = h.detach() if self.detach_pred_context else h
        theta_pred = normalize_generated_output(
            self.pred_head(h_pred), self.pred_output_scale,
            self.norm_output, self.pred_output_groups,
        )
        return theta_rew, theta_pred

    def rew_only(self, ctx_aug):
        """ctx_aug -> theta_rew (callable for reward_diversity_loss under sharing)."""
        return normalize_generated_output(
            self.rew_head(self.trunk(ctx_aug)), self.rew_output_scale,
            self.norm_output, self.rew_output_groups,
        )


class DualHyperNetwork(nn.Module):
    """v4 三路输入版 DualHyperNetwork (Ch4.3).

    v4 关键改动 (相对 v4.7):
        v4.7: __init__(rule_emb_dim, id_emb_dim, ...) + forward(rule_emb, id_emb)
        v4:   __init__(c_ctx_dim=16, ctx_aug_dim=80, ...) +
              forward_trans(c_ctx) -> theta_state +
              forward_subjective(ctx_aug) -> (theta_rew, theta_pred)
    """

    def __init__(
        self,
        c_ctx_dim,
        ctx_aug_dim,
        trans_param_count,
        rew_param_count,
        pred_param_count,
        hidden_dims=(256, 256),
        rew_hidden_dims=(256, 256, 256),
        norm_output=True,
        trans_output_scale_init=0.01,
        rew_output_scale_init=0.1,        # v4.7 关键
        pred_output_scale_init=0.01,
        detach_pred_context=True,         # D5: hyper_pred 输入 detach
        trans_output_groups=None,         # film_head: 分组 RMS 归一化的连续段大小
        rew_output_groups=None,
        pred_output_groups=None,
        share_subjective_trunk=False,     # Idea 1: 共享 hyper_rew/pred trunk
    ):
        """
        Args:
            c_ctx_dim:                 c_ctx 维度 (hyper_trans 输入, 默认 16)
            ctx_aug_dim:               ctx_aug 维度 (hyper_rew/pred 输入, 默认 80)
            trans_param_count:         FunctionalStateTransNet 参数总数
            rew_param_count:           FunctionalRewardHead 参数总数
            pred_param_count:          FunctionalPredictionNet 参数总数
            hidden_dims:               hyper_trans / hyper_pred 隐层
            rew_hidden_dims:           hyper_rew 隐层 (更深 3 层 for type 分化); None 退化为 hidden_dims
            norm_output:               L2 norm + scale (默认 True)
            trans/rew/pred_output_scale_init: output_scale 三初值 (C8)
            detach_pred_context:       D5; True 时 hyper_pred 接收 ctx_aug.detach()
        """
        super().__init__()

        self.c_ctx_dim = c_ctx_dim
        self.ctx_aug_dim = ctx_aug_dim
        self.detach_pred_context = detach_pred_context
        self.share_subjective_trunk = share_subjective_trunk

        rew_hdims = rew_hidden_dims if rew_hidden_dims is not None else hidden_dims

        # C1: hyper_trans 仅接 c_ctx_dim (16) -- 客观通路, 永不共享
        self.hyper_trans = HyperNetMLP(
            input_dim=c_ctx_dim,
            output_dim=trans_param_count,
            hidden_dims=hidden_dims,
            norm_output=norm_output,
            output_scale_init=trans_output_scale_init,
            output_groups=trans_output_groups,
        )

        if share_subjective_trunk:
            # Idea 1: 共享 trunk (深度 = rew_hdims) -> 两个 head (theta_rew/theta_pred).
            # 每个 head 保留各自 output_scale + output_groups.
            self.subjective = SubjectiveHyperNet(
                ctx_aug_dim=ctx_aug_dim,
                rew_param_count=rew_param_count,
                pred_param_count=pred_param_count,
                hidden_dims=rew_hdims,
                norm_output=norm_output,
                rew_output_scale_init=rew_output_scale_init,
                pred_output_scale_init=pred_output_scale_init,
                detach_pred_context=detach_pred_context,
                rew_output_groups=rew_output_groups,
                pred_output_groups=pred_output_groups,
            )
        else:
            # C2: hyper_rew 接完整 ctx_aug_dim (80), 更深 3 层
            self.hyper_rew = HyperNetMLP(
                input_dim=ctx_aug_dim,
                output_dim=rew_param_count,
                hidden_dims=rew_hdims,
                norm_output=norm_output,
                output_scale_init=rew_output_scale_init,   # 0.1 关键
                output_groups=rew_output_groups,
            )

            # C2: hyper_pred 同接 ctx_aug_dim (80)
            self.hyper_pred = HyperNetMLP(
                input_dim=ctx_aug_dim,
                output_dim=pred_param_count,
                hidden_dims=hidden_dims,
                norm_output=norm_output,
                output_scale_init=pred_output_scale_init,
                output_groups=pred_output_groups,
            )

        self.trans_param_count = trans_param_count
        self.rew_param_count = rew_param_count
        self.pred_param_count = pred_param_count

    def forward_trans(self, c_ctx):
        """C1: 仅接 c_ctx (B, 16) 生成 theta_state.

        改变 role / belief 输入时, 本方法输出不变 (物理转移上下文不变性).

        Args:
            c_ctx: (B, c_ctx_dim=16) float32
        Returns:
            theta_state: (B, trans_param_count) float32
        """
        assert c_ctx.shape[-1] == self.c_ctx_dim, (
            f"c_ctx last dim {c_ctx.shape[-1]} != c_ctx_dim {self.c_ctx_dim}"
        )
        return self.hyper_trans(c_ctx)

    def forward_subjective(self, ctx_aug):
        """C2: 接完整 80 维 ctx_aug 生成 (theta_rew, theta_pred).

        若 detach_pred_context=True (D5), hyper_pred 接收的 ctx_aug 被 .detach():
            - 防止 value loss 反向时扭曲 context_encoder 学习
            - reward loss 反向仍能驱动 context_encoder (经 hyper_rew 路径)

        Args:
            ctx_aug: (B, ctx_aug_dim=80) float32
        Returns:
            theta_rew:  (B, rew_param_count) float32
            theta_pred: (B, pred_param_count) float32
        """
        assert ctx_aug.shape[-1] == self.ctx_aug_dim, (
            f"ctx_aug last dim {ctx_aug.shape[-1]} != ctx_aug_dim {self.ctx_aug_dim}"
        )

        if self.share_subjective_trunk:
            # 共享 trunk: detach (D5) 在 trunk 输出处处理 (见 SubjectiveHyperNet.forward)
            return self.subjective(ctx_aug)

        theta_rew = self.hyper_rew(ctx_aug)

        # D5: hyper_pred 输入是否 detach (仅在传入 hyper_pred 那一行 detach, 不改原 ctx_aug)
        if self.detach_pred_context:
            theta_pred = self.hyper_pred(ctx_aug.detach())
        else:
            theta_pred = self.hyper_pred(ctx_aug)

        return theta_rew, theta_pred


def reward_diversity_loss(hyper_rew, ctx_aug_per_agent, target_cos=0.3, skip_pairs=None):
    """[v4.7 保留] theta_reward 多样性正则 (辅助, 默认禁用).

    按 D2: v4 type-aware 已通过 type_emb -> hyper_rew -> theta_rew^i 隐含实现.
    本 loss 作为辅助正则 (cfg.legacy.w_rew_diversity 控制权重, 默认 0.0),
    仅 Pkg-08 Ablation 启用做对照.

    注: share_subjective_trunk=True 时无独立 hyper_rew 模块, 传入
    model.hyper_net.subjective.rew_only 作为 hyper_rew callable 即可.

    Uses hinge loss: 仅当 cos_sim > target_cos 时惩罚, 充分分化后 loss 自然归零.

    Args:
        hyper_rew:         HyperNetMLP 实例
        ctx_aug_per_agent: (N, B, ctx_aug_dim) 每个 agent 的 ctx_aug 张量
        target_cos:        hinge 阈值
        skip_pairs:        跳过的 (i, j) 对 (如同 type agent 不需分化)
    Returns:
        loss: 标量 tensor (>= 0)
    """
    N, B, _ = ctx_aug_per_agent.shape
    thetas = [hyper_rew(ctx_aug_per_agent[i]) for i in range(N)]

    loss = torch.tensor(0.0, device=ctx_aug_per_agent.device)
    count = 0
    skip_set = set(skip_pairs) if skip_pairs else set()
    for i in range(N):
        for j in range(i + 1, N):
            if (i, j) in skip_set or (j, i) in skip_set:
                continue
            cos_sim = F.cosine_similarity(thetas[i], thetas[j], dim=-1)  # (B,)
            loss = loss + F.relu(cos_sim - target_cos).mean()
            count += 1

    return loss / max(count, 1)
