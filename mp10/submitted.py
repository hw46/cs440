'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

epsilon = 1e-3

def compute_transition(model):
    '''
    Parameters:
    model - the MDP model returned by load_MDP()
    
    Output:
    P - An M x N x 4 x M x N numpy array. P[r, c, a, r', c'] is the probability that the agent will move from cell (r, c) to (r', c') if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
    '''
    M, N = model.M, model.N
    P = np.zeros((M, N, 4, M, N))  # Initialize the transition probability matrix

    # Define action mappings (left, up, right, down)
    actions = [(0, -1), (-1, 0), (0, 1), (1, 0)]

    for r in range(M):
        for c in range(N):
            if model.TS[r, c]:  # If it's a terminal state, no outgoing transitions
                continue
            for a, (dr, dc) in enumerate(actions):
                # Target positions for the intended action and deviations
                positions = [(r + dr, c + dc), (r + actions[(a - 1) % 4][0], c + actions[(a - 1) % 4][1]), (r + actions[(a + 1) % 4][0], c + actions[(a + 1) % 4][1])]
                for idx, (nr, nc) in enumerate(positions):
                    if 0 <= nr < M and 0 <= nc < N and not model.W[nr, nc]:  # Check for valid and non-wall positions
                        P[r, c, a, nr, nc] += model.D[r, c, idx]
                    else:
                        P[r, c, a, r, c] += model.D[r, c, idx]  # Bump back to the original position on invalid move

    return P

def compute_utility(model, U_current, P):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()
    U_current - The current utility function, which is an M x N array
    P - The precomputed transition matrix returned by compute_transition()

    Output:
    U_next - The updated utility function, which is an M x N array
    '''
    M, N = model.M, model.N
    U_next = np.zeros_like(U_current)

    for r in range(M):
        for c in range(N):
            if model.TS[r, c]:  # Handle terminal states differently if necessary
                U_next[r, c] = model.R[r, c]  # If terminal states' utilities don't change, consider commenting this out
            else:
                # Calculate utility for each action and take max
                action_values = []
                for a in range(4):  # Assuming four possible actions: left, up, right, down
                    expected_utility = np.sum(P[r, c, a] * U_current)
                    action_values.append(expected_utility)
                U_next[r, c] = model.R[r, c] + model.gamma * max(action_values)
    
    return U_next

def value_iterate(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()

    Output:
    U - The utility function, which is an M x N array
    '''
    M, N = model.M, model.N
    U = np.zeros((M, N))  # Initialize utility values, assuming all start at zero
    P = compute_transition(model)  # Precompute transition probabilities

    iteration_count = 0
    while True:
        U_next = compute_utility(model, U, P)
        max_diff = np.max(np.abs(U_next - U))
        if max_diff < epsilon:
            break
        U = U_next
        iteration_count += 1
        if iteration_count >= 100:  # To avoid infinite loops in case of non-convergence
            break
    
    return U

def policy_evaluation(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP();
    
    Output:
    U - The converged utility function, which is an M x N array
    '''
    M, N = model.M, model.N
    U = np.zeros((M, N))  # Initialize utility values

    while True:
        U_next = np.zeros_like(U)
        for r in range(M):
            for c in range(N):
                U_next[r, c] = model.R[r, c] + model.gamma * np.sum(
                    model.FP[r, c, :, :] * U
                )

        if np.max(np.abs(U_next - U)) < epsilon:
            break
        U = U_next

    return U
