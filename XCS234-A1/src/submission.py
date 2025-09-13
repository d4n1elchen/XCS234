### MDP Value Iteration and Policy Iteration

import sys

import numpy as np

from riverswim import RiverSwim
from store import StoreKeeper

np.set_printoptions(precision=3)


def bellman_backup(state, action, R, T, gamma, V):
    """
    Perform a single Bellman backup.

    Parameters
    ----------
    state: int
    action: int
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    V: np.array (num_states)

    Returns
    -------
    backup_val: float
    """
    backup_val = None
    ############################
    ### START CODE HERE ###
    backup_val = R[state, action] + gamma * np.dot(T[state, action], V)
    ### END CODE HERE ###
    ############################

    return backup_val


def policy_evaluation(policy, R, T, gamma, tol=1e-3):
    """
    Compute the value function induced by a given policy for the input MDP
    Parameters
    ----------
    policy: np.array (num_states)
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    tol: float

    Returns
    -------
    value_function: np.array (num_states)
    """
    num_states, _ = R.shape
    value_function = None

    ############################
    ### START CODE HERE ###
    value_function = np.zeros(num_states)

    while True:
        V_next = np.zeros(num_states)
        for state in range(num_states):
            action = int(policy[state])
            V_next[state] = bellman_backup(state, action, R, T, gamma, value_function)
        max_diff = np.max(np.abs(value_function - V_next))
        value_function = V_next

        if max_diff < tol:
            break
    ### END CODE HERE ###
    ############################
    return value_function


def policy_improvement(R, T, V_policy, gamma):
    """
    Given the value function induced by a given policy, perform policy improvement
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    V_policy: np.array (num_states)
    gamma: float

    Returns
    -------
    new_policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    new_policy = None

    ############################
    ### START CODE HERE ###
    policy = np.zeros(num_states, dtype=int)

    for state in range(num_states):
        Q = np.zeros(num_actions)
        for action in range(num_actions):
            Q[action] = bellman_backup(state, action, R, T, gamma, V_policy)
        policy[state] = np.argmax(Q)

    new_policy = policy
    ### END CODE HERE ###
    ############################
    return new_policy


def policy_iteration(R, T, gamma, tol=1e-3):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)

    Returns
    -------
    V_policy: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    V_policy = None
    policy = None
    ############################
    ### START CODE HERE ###
    policy = np.random.choice(num_actions, size=num_states)

    while True:
        V_policy = policy_evaluation(policy, R, T, gamma, tol)
        new_policy = policy_improvement(R, T, V_policy, gamma)
        if np.array_equal(policy, new_policy):
            break
        policy = new_policy
    ### END CODE HERE ###
    ############################
    return V_policy, policy


def value_iteration(R, T, gamma, tol=1e-3):
    """Runs value iteration.
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)

    Returns
    -------
    value_function: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    value_function = None
    policy = None
    ############################
    ### START CODE HERE ###
    value_function = np.zeros(num_states)

    while True:
        V_next = np.zeros(num_states)

        for state in range(num_states):
            Q = np.zeros(num_actions)
            for action in range(num_actions):
                Q[action] = bellman_backup(state, action, R, T, gamma, value_function)
            V_next[state] = np.max(Q)

        max_diff = np.max(np.abs(value_function - V_next))
        value_function = V_next

        if max_diff < tol:
            break

    policy = policy_improvement(R, T, value_function, gamma)
    ### END CODE HERE ###
    ############################
    return value_function, policy


# Edit below to run policy and value iteration on different configurations
# You may change the parameters in the functions below
if __name__ == "__main__":
    SEED = 1234

    RIVER_CURRENT = "WEAK"
    assert RIVER_CURRENT in ["WEAK", "MEDIUM", "STRONG"]
    env = RiverSwim(RIVER_CURRENT, SEED)

    R, T = env.get_model()
    discount_factor = float(sys.argv[1]) if len(sys.argv) > 1 else 0.99

    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)

    V_pi, policy_pi = policy_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_pi)
    print([["L", "R"][a] for a in policy_pi])

    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)

    V_vi, policy_vi = value_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_vi)
    print([["L", "R"][a] for a in policy_vi])
