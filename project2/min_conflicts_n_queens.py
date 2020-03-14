import numpy as np
### WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS


def min_conflicts_n_queens(initialization: list) -> (list, int):
    """
    Solve the N-queens problem with no conflicts (i.e. each row, column, and diagonal contains at most 1 queen).
    Given an initialization for the N-queens problem, which may contain conflicts, this function uses the min-conflicts
    heuristic(see AIMA, pg. 221) to produce a conflict-free solution.

    Be sure to break 'ties' (in terms of minimial conflicts produced by a placement in a row) randomly.
    You should have a hard limit of 1000 steps, as your algorithm should be able to find a solution in far fewer (this
    is assuming you implemented initialize_greedy_n_queens.py correctly).

    Return the solution and the number of steps taken as a tuple. You will only be graded on the solution, but the
    number of steps is useful for your debugging and learning. If this algorithm and your initialization algorithm are
    implemented correctly, you should only take an average of 50 steps for values of N up to 1e6.

    As usual, do not change the import statements at the top of the file. You may import your initialize_greedy_n_queens
    function for testing on your machine, but it will be removed on the autograder (our test script will import both of
    your functions).

    On failure to find a solution after 1000 steps, return the tuple ([], -1).

    :param initialization: numpy array of shape (N,) where the i-th entry is the row of the queen in the ith column (may
                           contain conflicts)

    :return: solution - numpy array of shape (N,) containing a-conflict free assignment of queens (i-th entry represents
    the row of the i-th column, indexed from 0 to N-1)
             num_steps - number of steps (i.e. reassignment of 1 queen's position) required to find the solution.
    """

    N = len(initialization)
    solution = initialization.copy()
    num_steps = 0
    max_steps = 1000

    # build a conflict map of 3 parts: horizontal, posdiag, negdiag
    horizontal = np.zeros(N)
    posdiag = np.zeros(2*N-1)   # there are 2N-1 diagonals in a NxN board
    negdiag = np.zeros(2*N-1)
    for col in range(N):
        Q_row = solution[col]
        horizontal[Q_row] += 1
        Q_posdiag = N - 1 + (col - Q_row)
        posdiag[Q_posdiag] += 1
        Q_negdiag = col + Q_row
        negdiag[Q_negdiag] += 1

    for idx in range(max_steps):
        ## YOUR CODE HERE
        if check(horizontal, posdiag, negdiag) == True:
            return solution, num_steps

        # pick a random col and find out the conflict situation on this col
        rand_col = np.random.choice(solution.shape[0])
        old_row = solution[rand_col]
        col_conflict = count_col_conflict(N, horizontal, posdiag, negdiag, rand_col)
        col_conflict[old_row] -= 3      # counted extra 3 times where Q is placed

        if col_conflict[old_row] == 0:
            pass
        else:
            # find out the min row on this column, randomly break ties
            min_row = np.argwhere(col_conflict == col_conflict.min())
            rand_idx = np.random.choice(min_row.shape[0])
            new_row = min_row[rand_idx]

            # update: solution, horizontal, posdiag, negdiag
            solution[rand_col] = new_row
            horizontal[old_row] -= 1
            horizontal[new_row] += 1
            posdiag[N-1+(rand_col-old_row)] -= 1
            posdiag[N-1+(rand_col-new_row)] += 1
            negdiag[rand_col+old_row] -= 1
            negdiag[rand_col+new_row] += 1

        num_steps += 1

    return [], -1


def count_col_conflict(N, horizontal, posdiag, negdiag, col):
    col_conflict = np.zeros(N)
    for row in range(N):
        horiz_conf = horizontal[row]
        posdiag_conf = posdiag[N-1+(col-row)]
        negdiag_conf = negdiag[col+row]
        col_conflict[row] = horiz_conf + posdiag_conf + negdiag_conf
    return col_conflict


def check(horizontal, posdiag, negdiag):
    if (max(horizontal) == 1) and (max(posdiag) == 1) and (max(negdiag) == 1):
        return True
    else:
        return False


if __name__ == '__main__':
    # Test your code here!
    from initialize_greedy_n_queens import initialize_greedy_n_queens
    from support import plot_n_queens_solution

    N = 10
    # Use this after implementing initialize_greedy_n_queens.py
    assignment_initial = initialize_greedy_n_queens(N)
    # Plot the initial greedy assignment
    plot_n_queens_solution(assignment_initial)

    assignment_solved, n_steps = min_conflicts_n_queens(assignment_initial)
    # Plot the solution produced by your algorithm
    plot_n_queens_solution(assignment_solved)
