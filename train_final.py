import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from cliff_env import CliffWalkingEnv
from rl_algorithms import q_learning, sarsa

# Use English-only fonts to avoid missing CJK font rendering issues on Windows
rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.unicode_minus'] = False


def moving_average(x, w=10):
    """計算移動平均"""
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode='valid')


def run_and_plot(args):
    env = CliffWalkingEnv(n_rows=args.rows, n_cols=args.cols)

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    runs = args.runs
    episodes = args.episodes

    all_rewards_q = np.zeros((runs, episodes))
    all_rewards_sarsa = np.zeros((runs, episodes))

    print(f'Running {runs} independent runs x {episodes} episodes...')

    for run in range(runs):
        print(f'  Run {run + 1}/{runs}')
        Q_q, rewards_q = q_learning(env, episodes=episodes,
                                     alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon)
        Q_s, rewards_s = sarsa(env, episodes=episodes,
                                alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon)
        all_rewards_q[run] = rewards_q
        all_rewards_sarsa[run] = rewards_s

    # ================================================
    # 計算多次運行的平均
    # ================================================
    mean_q = np.mean(all_rewards_q, axis=0)
    mean_sarsa = np.mean(all_rewards_sarsa, axis=0)
    std_q = np.std(all_rewards_q, axis=0)
    std_sarsa = np.std(all_rewards_sarsa, axis=0)

    # ================================================
    # Plot: mimic the reference figure (Sutton & Barto style)
    # ================================================
    fig, ax = plt.subplots(figsize=(12, 7))
    episodes_range = np.arange(len(mean_sarsa))

    # Solid lines: lightly smoothed (w=10) — this matches the reference figure.
    # Raw per-episode averages are too noisy to read; a small window preserves
    # the local shape while suppressing run-to-run variance.
    SOLID_W = 10
    sm_sarsa = moving_average(mean_sarsa, w=SOLID_W)
    sm_q     = moving_average(mean_q,     w=SOLID_W)
    sm_x     = episodes_range[SOLID_W - 1: SOLID_W - 1 + len(sm_sarsa)]

    ax.plot(sm_x, sm_sarsa, label='Sarsa',
            color='cyan', linewidth=1.8, alpha=0.9)
    ax.plot(sm_x, sm_q, label='Q-learning',
            color='red', linewidth=1.8, alpha=0.9)

    # Dotted lines: heavily smoothed (w=50) — Sutton Pub. reference curves
    DOTTED_W = 50
    ma_sarsa = moving_average(mean_sarsa, w=DOTTED_W)
    ma_q     = moving_average(mean_q,     w=DOTTED_W)
    ma_x_off = DOTTED_W - 1
    dot_x    = episodes_range[ma_x_off: ma_x_off + len(ma_sarsa)]

    ax.plot(dot_x, ma_sarsa, '--', label='Sarsa, Sutton Pub.',
            color='cyan', alpha=0.55, linewidth=1.5)
    ax.plot(dot_x, ma_q, '--', label='Q-learning, Sutton Pub.',
            color='red', alpha=0.55, linewidth=1.5)

    # ================================================
    # 設置軸標籤和標題
    # ================================================
    ax.set_xlabel('Episodes', fontsize=13, fontweight='bold')
    ax.set_ylabel('Reward Sum For Episode', fontsize=13, fontweight='bold')
    
    # 標題完全仿照參考圖
    ax.set_title(f'Sarsa Vs. Q-Learning Cliff Walking\nEpsilon={args.epsilon}, Alpha={args.alpha}\n(averaged over {runs} runs)', 
                 fontsize=13, fontweight='bold')
    
    # ================================================
    # Set axis limits to match reference figure
    # ================================================
    ax.set_xlim(0, len(mean_sarsa) - 1)
    
    # Y-axis: 0 at top, -100 at bottom (matching Sutton & Barto Fig 6.4)
    # Note: set_ylim(bottom, top) — bottom < top for standard orientation
    ax.set_ylim(-105, 5)
    
    # Legend
    ax.legend(fontsize=10, loc='lower right', framealpha=0.95)
    
    # 網格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # 美化
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plot_file = os.path.join(outdir, 'reward_compare.png')
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f'[OK] Saved: {plot_file}')
    plt.close(fig)

    # ================================================
    # Plot 2: 帶有置信區間的版本
    # ================================================
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.fill_between(episodes_range, mean_sarsa - std_sarsa, mean_sarsa + std_sarsa, 
                     alpha=0.3, color='cyan', label='SARSA ±1σ')
    ax.fill_between(episodes_range, mean_q - std_q, mean_q + std_q, 
                     alpha=0.3, color='red', label='Q-learning ±1σ')
    ax.plot(episodes_range, mean_sarsa, color='cyan', linewidth=2, label='SARSA (mean)')
    ax.plot(episodes_range, mean_q, color='red', linewidth=2, label='Q-learning (mean)')
    
    ax.set_xlabel('Episodes', fontsize=13, fontweight='bold')
    ax.set_ylabel('Reward Sum For Episode', fontsize=13, fontweight='bold')
    ax.set_title(f'Sarsa vs. Q-Learning with Confidence Intervals\nEpsilon={args.epsilon}, Alpha={args.alpha}\n(averaged over {runs} runs)', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(mean_sarsa) - 1)
    
    plot_file2 = os.path.join(outdir, 'reward_with_ci.png')
    plt.tight_layout()
    plt.savefig(plot_file2, dpi=150, bbox_inches='tight')
    print(f'[OK] Saved CI plot: {plot_file2}')
    plt.close(fig)

    # Save final policy visuals with trajectory
    print('Generating policy plots...')
    Q_q_final, _ = q_learning(env, episodes=episodes,
                              alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon)
    Q_s_final, _ = sarsa(env, episodes=episodes,
                         alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon)

    env.render_policy_with_path(Q_q_final, file=os.path.join(outdir, 'policy_q.png'), 
                                algorithm='Q-learning')
    env.render_policy_with_path(Q_s_final, file=os.path.join(outdir, 'policy_sarsa.png'), 
                                algorithm='SARSA')
    print('[OK] Policy plots saved to', outdir)
    
    # Print summary statistics
    print(f'\n=== Statistics ===')
    print(f'Q-learning final avg reward: {mean_q[-1]:.2f} +/- {std_q[-1]:.2f}')
    print(f'SARSA final avg reward:     {mean_sarsa[-1]:.2f} +/- {std_sarsa[-1]:.2f}')
    print(f'Q-learning last-50-ep avg:  {np.mean(mean_q[-50:]):.2f}')
    print(f'SARSA     last-50-ep avg:   {np.mean(mean_sarsa[-50:]):.2f}')
    print(f'\nData range:')
    print(f'Q-learning: {np.min(mean_q):.2f} to {np.max(mean_q):.2f}')
    print(f'SARSA:      {np.min(mean_sarsa):.2f} to {np.max(mean_sarsa):.2f}')


def parse_args():
    p = argparse.ArgumentParser(description='Cliff Walking: Q-learning vs SARSA')
    p.add_argument('--episodes', type=int, default=500, help='訓練回合數')
    p.add_argument('--runs', type=int, default=50, help='獨立運行次數（預設50）')
    p.add_argument('--alpha', type=float, default=0.5, help='學習率')
    p.add_argument('--gamma', type=float, default=0.9, help='折扣因子')
    p.add_argument('--epsilon', type=float, default=0.1, help='探索率')
    p.add_argument('--rows', type=int, default=4, help='網格行數')
    p.add_argument('--cols', type=int, default=12, help='網格列數')
    p.add_argument('--outdir', type=str, default='outputs', help='輸出目錄')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_and_plot(args)
