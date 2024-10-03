import numpy as np

from dynamic_programming.grid_world_env import GridWorldEnv
from dynamic_programming.mdp import MDP
from dynamic_programming.stochastic_grid_word_env import StochasticGridWorldEnv

# Exercice 2: Résolution du MDP
# -----------------------------
# Ecrire une fonction qui calcule la valeur de chaque état du MDP, en
# utilisant la programmation dynamique.
# L'algorithme de programmation dynamique est le suivant:
#   - Initialiser la valeur de chaque état à 0
#   - Tant que la valeur de chaque état n'a pas convergé:
#       - Pour chaque état:
#           - Estimer la fonction de valeur de chaque état
#           - Choisir l'action qui maximise la valeur
#           - Mettre à jour la valeur de l'état
#
# Indice: la fonction doit être itérative.


def mdp_value_iteration(mdp: MDP, max_iter: int = 1000, gamma=1.0) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration":
    https://en.wikipedia.org/wiki/Markov_decision_process#Value_iteration
    """
    values = np.zeros(mdp.observation_space.n)
    # BEGIN SOLUTION
    for _ in range(max_iter):
        prev_val = values.copy()
        delta = 0
        for state in range(mdp.observation_space.n):
            mdp.initial_state = state
            delta = value_iteration_per_state_mdp(mdp, values, gamma, prev_val, delta, state=state)
            #delta = value_iteration_per_state(mdp, values, gamma, prev_val, delta, state=state)
        if delta < 1e-6:
            break
    return values
    # END SOLUTION
    return values

def value_iteration_per_state_mdp(env, values, gamma, prev_val, delta, state):
    values[state] = float("-inf")
    for action in range(env.action_space.n):
        next_state_data = env.P[state][action]
        next_state, reward, done = next_state_data

        current_sum = reward + gamma * prev_val[next_state]
        values[state] = max(values[state], current_sum)
    
    delta = max(delta, np.abs(values[state] - prev_val[state]))
    return delta


def grid_world_value_iteration(
    env: GridWorldEnv,
    max_iter: int = 1000,
    gamma=1.0,
    theta=1e-5,
) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration".
    theta est le seuil de convergence (différence maximale entre deux itérations).
    """
    values = np.zeros((4, 4))
    # BEGIN SOLUTION
    for i in range(max_iter):
        prev_values = np.copy(values)
        delta = 0  
        # pour chaque cases de la grille
        for row in range(env.height):
            for col in range(env.width):
                env.set_state(row, col)
                if env.grid[row, col] in {"W", "P", "N"}:
                    continue # Skip

                max_value = float("-inf")
                
                for action in range(env.action_space.n):
                    next_state, reward, _, _ = env.step(action, make_move=False)
                    next_row, next_col = next_state
                    value = reward + gamma * prev_values[next_row, next_col]
                    max_value = max(max_value, value)
                values[row, col] = max_value
                delta = max(delta, np.abs(prev_values[row, col] - values[row, col]))

        if delta < theta:
            print(f"Convergence à la {i} ème itération.")
            break
    
    return values
    # END SOLUTION


def value_iteration_per_state(env, values, gamma, prev_val, delta):
    row, col = env.current_position
    values[row, col] = float("-inf")
    for action in range(env.action_space.n):
        next_states = env.get_next_states(action=action)
        current_sum = 0
        for next_state, reward, probability, _, _ in next_states:
            # print((row, col), next_state, reward, probability)
            next_row, next_col = next_state
            current_sum += (
                probability
                * env.moving_prob[row, col, action]
                * (reward + gamma * prev_val[next_row, next_col])
            )
        values[row, col] = max(values[row, col], current_sum)
    delta = max(delta, np.abs(values[row, col] - prev_val[row, col]))

    return delta


def stochastic_grid_world_value_iteration(
    env: StochasticGridWorldEnv,
    max_iter: int = 1000,
    gamma: float = 1.0,
    theta: float = 1e-5,
) -> np.ndarray:
    values = np.zeros((env.height, env.width))
    # 
    for _ in range(max_iter):
        delta = 0
        new_values = values.copy()
        
        for row in range(env.height):
            for col in range(env.width):
                if env.grid[row, col] in {"W", "P", "N"}:
                    continue
                
                old_value = values[row, col]
                env.set_state(row, col)
                
                action_values = []
                for action in range(env.action_space.n):
                    next_states = env.get_next_states(action)
                    action_value = 0
                    for next_state, reward, prob, is_done, _ in next_states:
                        next_row, next_col = next_state
                        action_value += prob * (reward + gamma * values[next_row, next_col])
                    
                    action_values.append(action_value)
                new_values[row, col] = max(action_values)
                delta = max(delta, abs(old_value - new_values[row, col]))
        values = new_values
        # theta x 10 pour que ça fasse 10^-6
        if delta < (theta / 10):
            break
    
    return values