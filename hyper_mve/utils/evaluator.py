"""
Unified Evaluation Framework (v4.1)

Provides common utilities for evaluating all three experiment tracks:
  - Exp1 Baseline
  - Exp2 Oracle-HyperMuZero
  - Exp3 Infer-HyperMuZero

Each script defines its own `evaluate_fn(env, rule, num_episodes, ...)` that
returns an EvalResult dict. This module provides:
  - run_eval_suite(): loop over test_rules and call evaluate_fn
  - log_results(): write to TensorBoard
  - print_summary(): print Agent-wise comparison table
"""
import numpy as np


# Default test rules for evaluation
DEFAULT_TEST_RULES = [0.0, 0.5, 1.0]

# Agent role names for pretty-printing
AGENT_NAMES = ['Hunter0', 'Hunter1', 'Agent2(Var)', 'Prey']


def run_eval_suite(evaluate_fn, env, test_rules=None, num_episodes=10, **kwargs):
    """
    Run evaluation across multiple rules.

    Args:
        evaluate_fn: Callable(env, rule, num_episodes, **kwargs) -> EvalResult
            Each experiment script provides its own evaluate_fn.
        env: NonStationaryMultiAgentEnv instance.
        test_rules: List of rule values to test. Default [0.0, 0.5, 1.0].
        num_episodes: Number of episodes per rule.
        **kwargs: Extra arguments forwarded to evaluate_fn
                  (e.g., model, device, config for different experiment types).

    Returns:
        dict: {rule_val: EvalResult, ...}
            EvalResult = {
                'mean_rewards': np.array([r0, r1, r2, r3]),  # per-agent
                'mean_total_reward': float,
                'mean_ep_length': float,
            }
    """
    if test_rules is None:
        test_rules = DEFAULT_TEST_RULES

    all_results = {}
    for rule_val in test_rules:
        result = evaluate_fn(env, rule=rule_val, num_episodes=num_episodes, **kwargs)
        all_results[rule_val] = result

    return all_results


def log_results(writer, results, step, prefix='eval'):
    """
    Log evaluation results to TensorBoard.

    Args:
        writer: SummaryWriter instance.
        results: dict from run_eval_suite, {rule_val: EvalResult}.
        step: Global training step.
        prefix: Tag prefix for TensorBoard. Default 'eval'.
    """
    for rule_val, res in results.items():
        rule_tag = f'rule_{rule_val:.1f}'

        # Per-agent rewards
        mean_rewards = res['mean_rewards']
        for i, agent_name in enumerate(AGENT_NAMES):
            writer.add_scalar(
                f'{prefix}/{rule_tag}/{agent_name}_reward',
                mean_rewards[i], step
            )

        # Aggregated metrics
        writer.add_scalar(
            f'{prefix}/{rule_tag}/total_reward',
            res['mean_total_reward'], step
        )
        writer.add_scalar(
            f'{prefix}/{rule_tag}/ep_length',
            res['mean_ep_length'], step
        )


def print_summary(results, step=None):
    """
    Print a formatted Agent-wise comparison table.

    Example output:
    ┌──────────────────────────────────────────────────────────────┐
    │  Eval @ step 10000                                          │
    ├──────────┬──────────┬──────────┬────────────┬───────┬───────┤
    │ Rule     │ Hunter0  │ Hunter1  │ Agent2(Var)│ Prey  │ Total │
    ├──────────┼──────────┼──────────┼────────────┼───────┼───────┤
    │ 0.0      │  -1.23   │  -1.45   │   2.34     │ -0.56 │ -0.90 │
    │ 0.5      │  ...     │  ...     │   ...      │ ...   │ ...   │
    │ 1.0      │  ...     │  ...     │   ...      │ ...   │ ...   │
    └──────────┴──────────┴──────────┴────────────┴───────┴───────┘

    Args:
        results: dict from run_eval_suite, {rule_val: EvalResult}.
        step: Optional training step for display.
    """
    header = f'  Eval' + (f' @ step {step}' if step is not None else '')
    sep = '=' * 72

    print(f'\n{sep}')
    print(header)
    print(sep)

    # Table header
    col_fmt = '{:<8s}' + '{:>10s}' * (len(AGENT_NAMES) + 2)
    print(col_fmt.format('Rule', *AGENT_NAMES, 'Total', 'EpLen'))
    print('-' * 72)

    # Table rows
    for rule_val in sorted(results.keys()):
        res = results[rule_val]
        mean_rewards = res['mean_rewards']
        row_vals = [f'{r:>10.2f}' for r in mean_rewards]
        row_vals.append(f'{res["mean_total_reward"]:>10.2f}')
        row_vals.append(f'{res["mean_ep_length"]:>10.1f}')
        print(f'{rule_val:<8.1f}' + ''.join(row_vals))

    print(sep + '\n')


def make_eval_result(episode_rewards_n, episode_lengths):
    """
    Construct a standardized EvalResult dict from raw episode data.

    Args:
        episode_rewards_n: list of np.array, each shape (num_agents,).
            Per-episode cumulative rewards for each agent.
        episode_lengths: list of int, episode lengths.

    Returns:
        EvalResult dict:
        {
            'mean_rewards': np.array([r0, r1, r2, r3]),
            'mean_total_reward': float,
            'mean_ep_length': float,
        }
    """
    rewards_array = np.array(episode_rewards_n)  # (num_episodes, num_agents)
    mean_rewards = rewards_array.mean(axis=0)     # (num_agents,)
    return {
        'mean_rewards': mean_rewards,
        'mean_total_reward': float(mean_rewards.sum()),
        'mean_ep_length': float(np.mean(episode_lengths)),
    }
