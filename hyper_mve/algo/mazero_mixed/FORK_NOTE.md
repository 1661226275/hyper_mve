# mazero_mixed — 混合博弈迁移用 MAZero 分叉

- 上游来源：`hyper_mve/comparison/vendor/MAZero/`（liuqh16/MAZero 官方克隆，ICLR 2024，GPL-3.0）。
  本目录为其**方法开发分叉**（fork-the-clone 决策，2026-07-16，见
  `~/.claude/plans/background-the-mid-term-defense-enchanted-bear.md`）。
- `comparison/vendor/MAZero/` 保持原样，作为协作版对比基线；一切迁移改动只发生在本目录。
- 许可证：本目录代码继承 GPL-3.0（LICENSE 随分叉保留）。
- 迁移阶段（报告三（1）四阶段）：
  1. RelationCommons Game 适配 + 未改动基座冒烟 ✅
     （2026-07-16：`config/relation/`，rel_duo_coop 冒烟 320 步，Test Mean Score
     0.81→12.6，无错误，ckpt 落盘；日志 `~/.claude/jobs/546e8c99/tmp/pristine_smoke.log`）
  2. ctree 向量化（逐智能体价值、向量回传、解耦选择）+ 全合作回归门槛 ✅
     （2026-07-16：cnode.h/.cpp 逐智能体 reward/pred_value/OS(λ) 统计 + 双 minmax 流；
     `select_mode` 0=联合（原语义）/1=解耦（逐智能体边际 UCB + 最大一致投影）；
     配置旗标 `--decoupled_selection`；逐智能体导出 get_*_vec 贯通 pxd/pyx/mcts_sampled；
     回归门槛 `test/test_ctree_regression.py` PASS（联合模式对 golden 完全一致，
     全合作下逐智能体根价值相等）；解耦模式冒烟 320 步 1.08→11.88 无错误。
     注意：修改 ctree 后必须重新 `build_ext`（CFLAGS 带 python+numpy include）。）
  3. 主观头/信念/超网络迁移 + 贝叶斯平均叶评估 + AWPO 逐智能体化 ⬜（进行中，分 4a/4b/4c）
     - 4a ✅完成（冒烟 PASS：rel_duo_coop 320 步 1.53→15.29，与向量化前基准相当，无错误）：
       全管线逐智能体化——env_wrapper 返回逐智能体奖励向量；
       model.py R/V 头输出 (B,N,S)；reanalyze 目标/根值/优势 (…,N)（优势逐智能体归一化）；
       train.py 逐智能体 AWPO（per_agent_log_prob × 自身优势，PG_type raw/sharp；none 分支保持联合形式）、
       优先级取智能体均值；config `_support_loss` 形状自适应；selfplay/test 日志取均值。
       门槛：rel_duo_coop 冒烟学习曲线与向量化前相当（前基准 0.8→12.6 / 解耦 1.1→11.9）。
     - 4b ✅代码完成（推理探针 PASS；rel_duo 冒烟进行中）：`config/relation/subjective_model.py`
       HyperMAMuZeroNet（直接 import 仓库 v5 模块，不复制）；FunctionalValueHead（film_head）；
       旗标 --subjective_model / --belief_point_estimate / --belief_{oracle,anneal}_steps /
       --belief_loss_coeff / --belief_grad_gating_steps；GameHistory 新增 beliefs/g_trues/
       belief_hiddens；selfplay oracle 环境 + GRU 维护 + 课程混合（oracle_blend_weight）；
       test.py 纯后验；reanalyze 两处按存储信念 set_belief + make_batch 附加三字段；
       update_weights 解包 + set_belief(step) + 单步截断信念 CE/多样性损失（train_logs belief_loss）。
       原设计草案：
       * v5 模块源：`hyper_mve/models/{belief_net,belief_encoder,_belief_obs_encoder,belief_losses,
         grad_gating,hyper_network,functional_nets,role_encoder,tri_context_encoder}.py` → 拷入
         `core/subjective/`；
       * ctx 流：真实步计算、搜索期冻结——selfplay/test 每 env 维护 BeliefNet GRU 隐状态；
         行向量从 obs 尾部 N-1 维提取（RelationObservationLayout）；模型加 `set_context(rows, belief)`
         （v5 6-API 模式），initial/recurrent_inference 期间头参数由 hyper(ctx) 生成；
       * 叶贝叶斯平均：在价值头内部枚举 G（|G|=5）：θ_val^{i,g}=hyper(ctx_i(g))，
         v̂_i=Σ_g b_i(g)·V_g —— C++ 层无需再改；点估计单头 = 消融旗标；
       * 信念监督：训练期 env 以 oracle_mode=True 提供 g 真值（仅入 info→GameHistory 新字段，
         永不进模型输入）；GameHistory 存每步 belief 后验 + g 真值 + GRU 隐状态；
         信念 CE 在 update_weights 内对采样 (obs_t, h_{t-1}) 做单步截断 BPTT，
         或对整局 obs 序列做 GRU 前向（v5 trainer 模式，T=100 可承受）——实现时二选一；
       * 课程/门控：三阶段示教-退火-纯推断 + belief_grad_gating 原样移植。
     - 4c ✅PASS（2026-07-16 晚）：rel_duo 全隐藏 5 构型族 + --subjective_model
       + --decoupled_selection，320 步 1.14→4.38→5.33→9.26（step300），无错误，
       belief_loss 正常记录（~1.44，短课程下符合预期）；阶段 3 全栈打通。
       日志 `~/.claude/jobs/546e8c99/tmp/subjective_smoke.log`。
  4. ExternalBaselineRunner 统一评估协议接入 ✅（2026-07-16 深夜）
     - `hyper_mve/baselines/external/mazero_mixed.py`（注册名 "mazero_mixed"——是方法不是基线）：
       train() 以 env_cfg_override 从 harness cfg 注入环境身份、程序化构造 args 驱动
       train_sync_serial（预算映射 total_transitions=total_env_steps、
       training_steps=env_steps/16）；evaluate() 消费 env_fn（无 oracle）执行
       分布式先验策略（信念 GRU + argmax），逐构型填 rel-v1 EvalReport +
       _eval_episode_returns；ckpt 存取 + param_count。
     - 契约测试 `tests/baselines/external/test_mazero_mixed_smoke.py` PASS（84s）。
     - 已披露偏差：train 不消费 env_fn（自建 oracle 环境=方法的 CTDE 设计）；
       评估为先验策略模式（搜索评估协议另行披露）。
     - 消融配套：叶评估点估计 = --belief_point_estimate ✅；协作完整性对照 =
       分叉不带 --subjective_model/--decoupled_selection（回归测试已证与原基座逐位一致），
       无需单独移植 pristine 克隆；条件化机制消融（MoE 路由/FiLM）另列任务。
