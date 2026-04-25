import numpy as np


def epsilon_greedy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(Q.shape[1])
    return int(np.argmax(Q[state]))


def q_learning(env, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1, max_steps=10000):
    """
    Q-learning algorithm (Off-policy).

    Update: Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

    Args:
        env: CliffWalkingEnv instance
        episodes (int): number of training episodes
        alpha (float): learning rate
        gamma (float): discount factor
        epsilon (float): epsilon-greedy exploration rate
        max_steps (int): max steps per episode to prevent infinite loops

    Returns:
        Q (np.ndarray): learned Q-table, shape (n_states, n_actions)
        rewards (list[float]): total reward per episode
    """
    Q = np.zeros((env.n_states, env.n_actions))
    rewards = []

    for ep in range(episodes):
        s = env.reset()
        total_r = 0
        done = False
        step = 0
        while not done and step < max_steps:
            a = epsilon_greedy(Q, s, epsilon)
            s2, r, done, _ = env.step(a)
            # Off-policy: use best possible next action regardless of what was taken
            Q[s, a] += alpha * (r + gamma * np.max(Q[s2]) - Q[s, a])
            s = s2
            total_r += r
            step += 1
        rewards.append(total_r)
    return Q, rewards


def sarsa(env, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1, max_steps=10000):
    """
    SARSA algorithm (On-policy).

    Update: Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
    where a' is the actual action chosen by epsilon-greedy from s'.

    Args:
        env: CliffWalkingEnv instance
        episodes (int): number of training episodes
        alpha (float): learning rate
        gamma (float): discount factor
        epsilon (float): epsilon-greedy exploration rate
        max_steps (int): max steps per episode to prevent infinite loops

    Returns:
        Q (np.ndarray): learned Q-table, shape (n_states, n_actions)
        rewards (list[float]): total reward per episode
    """
    Q = np.zeros((env.n_states, env.n_actions))
    rewards = []

    for ep in range(episodes):
        s = env.reset()
        a = epsilon_greedy(Q, s, epsilon)
        total_r = 0
        done = False
        step = 0
        while not done and step < max_steps:
            s2, r, done, _ = env.step(a)
            a2 = epsilon_greedy(Q, s2, epsilon)
            # On-policy: update using the actual next action a2 chosen by epsilon-greedy
            Q[s, a] += alpha * (r + gamma * Q[s2, a2] - Q[s, a])
            s, a = s2, a2
            total_r += r
            step += 1
        rewards.append(total_r)
    return Q, rewards
