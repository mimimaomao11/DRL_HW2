import os
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from flask import Flask, render_template, request, jsonify
import threading

from cliff_env import CliffWalkingEnv
from rl_algorithms import q_learning, sarsa

rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

app = Flask(__name__)

# Global state
training_progress = {"status": "idle", "progress": 0, "results": None}


def moving_average(x, w=10):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode='valid')


def train_models(episodes, runs, alpha, gamma, epsilon, rows, cols):
    """Train both models and generate plots."""
    global training_progress
    
    training_progress["status"] = "running"
    training_progress["progress"] = 0
    
    env = CliffWalkingEnv(n_rows=rows, n_cols=cols)
    
    all_rewards_q = np.zeros((runs, episodes))
    all_rewards_sarsa = np.zeros((runs, episodes))
    
    for run in range(runs):
        Q_q, rewards_q = q_learning(env, episodes=episodes,
                                     alpha=alpha, gamma=gamma, epsilon=epsilon)
        Q_s, rewards_s = sarsa(env, episodes=episodes,
                                alpha=alpha, gamma=gamma, epsilon=epsilon)
        all_rewards_q[run] = rewards_q
        all_rewards_sarsa[run] = rewards_s
        training_progress["progress"] = int((run + 1) / runs * 100)
    
    mean_q = np.mean(all_rewards_q, axis=0)
    mean_sarsa = np.mean(all_rewards_sarsa, axis=0)
    std_q = np.std(all_rewards_q, axis=0)
    std_sarsa = np.std(all_rewards_sarsa, axis=0)
    
    # Generate plot 1: Reward curves
    fig, ax = plt.subplots(figsize=(10, 6))
    episodes_range = np.arange(len(mean_sarsa))
    
    ax.plot(episodes_range, mean_sarsa, label='SARSA', color='teal', linewidth=2, alpha=0.7)
    ax.plot(episodes_range, mean_q, label='Q-learning', color='red', linewidth=2, alpha=0.7)
    
    ma_sarsa = moving_average(mean_sarsa, w=20)
    ma_q = moving_average(mean_q, w=20)
    ax.plot(np.arange(len(ma_sarsa)), ma_sarsa, '--', label='SARSA (MA)', 
            color='teal', alpha=0.5, linewidth=1.5)
    ax.plot(np.arange(len(ma_q)), ma_q, '--', label='Q-learning (MA)', 
            color='red', alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel('Episodes', fontsize=12)
    ax.set_ylabel('Reward Sum For Episode', fontsize=12)
    ax.set_title(f'Sarsa vs. Q-Learning\nEpsilon={epsilon}, Alpha={alpha}', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    buf1 = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf1, format='png', dpi=100)
    buf1.seek(0)
    plot1_b64 = base64.b64encode(buf1.getvalue()).decode()
    plt.close(fig)
    
    # Generate plot 2: With confidence intervals
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(episodes_range, mean_sarsa - std_sarsa, mean_sarsa + std_sarsa, 
                     alpha=0.3, color='teal', label='SARSA ±1σ')
    ax.fill_between(episodes_range, mean_q - std_q, mean_q + std_q, 
                     alpha=0.3, color='red', label='Q-learning ±1σ')
    ax.plot(episodes_range, mean_sarsa, color='teal', linewidth=2.5, label='SARSA (mean)')
    ax.plot(episodes_range, mean_q, color='red', linewidth=2.5, label='Q-learning (mean)')
    
    ax.set_xlabel('Episodes', fontsize=12)
    ax.set_ylabel('Reward Sum For Episode', fontsize=12)
    ax.set_title(f'Sarsa vs. Q-Learning with Confidence Intervals\nEpsilon={epsilon}, Alpha={alpha}', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    buf2 = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf2, format='png', dpi=100)
    buf2.seek(0)
    plot2_b64 = base64.b64encode(buf2.getvalue()).decode()
    plt.close(fig)
    
    # Generate policy visualizations
    Q_q_final, _ = q_learning(env, episodes=episodes,
                              alpha=alpha, gamma=gamma, epsilon=epsilon)
    Q_s_final, _ = sarsa(env, episodes=episodes,
                         alpha=alpha, gamma=gamma, epsilon=epsilon)
    
    arrow = {0: (0, 0.4), 1: (0.4, 0), 2: (0, -0.4), 3: (-0.4, 0)}
    
    # Q-learning policy
    policy_q = np.argmax(Q_q_final, axis=1).reshape(rows, cols)
    fig, ax = plt.subplots(figsize=(cols / 2, rows / 2))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.invert_yaxis()
    for r in range(rows):
        for c in range(cols):
            if (r, c) == env.start:
                ax.text(c + 0.5, r + 0.5, 'S', ha='center', va='center', fontsize=10, weight='bold')
            elif (r, c) == env.goal:
                ax.text(c + 0.5, r + 0.5, 'G', ha='center', va='center', fontsize=10, weight='bold')
            elif (r, c) in env.cliff:
                ax.add_patch(plt.Rectangle((c, r), 1, 1, color='lightblue'))
            a = policy_q[r, c]
            dx, dy = arrow[a]
            ax.arrow(c + 0.5 - dx / 2, r + 0.5 - dy / 2, dx, dy,
                     head_width=0.15, head_length=0.15, fc='k', ec='k')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Q-learning Policy')
    buf_q = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf_q, format='png', dpi=100)
    buf_q.seek(0)
    policy_q_b64 = base64.b64encode(buf_q.getvalue()).decode()
    plt.close(fig)
    
    # SARSA policy
    policy_s = np.argmax(Q_s_final, axis=1).reshape(rows, cols)
    fig, ax = plt.subplots(figsize=(cols / 2, rows / 2))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.invert_yaxis()
    for r in range(rows):
        for c in range(cols):
            if (r, c) == env.start:
                ax.text(c + 0.5, r + 0.5, 'S', ha='center', va='center', fontsize=10, weight='bold')
            elif (r, c) == env.goal:
                ax.text(c + 0.5, r + 0.5, 'G', ha='center', va='center', fontsize=10, weight='bold')
            elif (r, c) in env.cliff:
                ax.add_patch(plt.Rectangle((c, r), 1, 1, color='lightblue'))
            a = policy_s[r, c]
            dx, dy = arrow[a]
            ax.arrow(c + 0.5 - dx / 2, r + 0.5 - dy / 2, dx, dy,
                     head_width=0.15, head_length=0.15, fc='k', ec='k')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('SARSA Policy')
    buf_s = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf_s, format='png', dpi=100)
    buf_s.seek(0)
    policy_s_b64 = base64.b64encode(buf_s.getvalue()).decode()
    plt.close(fig)
    
    stats = {
        "q_final": float(mean_q[-1]),
        "q_std": float(std_q[-1]),
        "sarsa_final": float(mean_sarsa[-1]),
        "sarsa_std": float(std_sarsa[-1]),
        "q_last50": float(np.mean(mean_q[-50:])),
        "sarsa_last50": float(np.mean(mean_sarsa[-50:])),
    }
    
    training_progress["status"] = "complete"
    training_progress["progress"] = 100
    training_progress["results"] = {
        "plot1": plot1_b64,
        "plot2": plot2_b64,
        "policy_q": policy_q_b64,
        "policy_s": policy_s_b64,
        "stats": stats
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/train', methods=['POST'])
def train():
    data = request.json
    episodes = data.get('episodes', 500)
    runs = data.get('runs', 20)
    alpha = data.get('alpha', 0.5)
    gamma = data.get('gamma', 0.9)
    epsilon = data.get('epsilon', 0.1)
    rows = data.get('rows', 4)
    cols = data.get('cols', 12)
    
    # Run training in background
    thread = threading.Thread(target=train_models, args=(episodes, runs, alpha, gamma, epsilon, rows, cols))
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "started"})


@app.route('/progress')
def progress():
    return jsonify(training_progress)


@app.route('/results')
def results():
    if training_progress["results"]:
        return jsonify(training_progress["results"])
    return jsonify({"error": "No results yet"}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
