import numpy as np
import support
### WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS


def initialize_greedy_n_queens(N: int) -> list:
    """
    This function takes an integer N and produces an initial assignment that greedily (in terms of minimizing conflicts)
    assigns the row for each successive column. Note that if placing the i-th column's queen in multiple row positions j
    produces the same minimal number of conflicts, then you must break the tie RANDOMLY! This strongly affects the
    algorithm's performance!

    Example:
    Input N = 4 might produce greedy_init = np.array([0, 3, 1, 2]), which represents the following "chessboard":

     _ _ _ _
    |Q|_|_|_|
    |_|_|Q|_|
    |_|_|_|Q|
    |_|Q|_|_|

    which has one diagonal conflict between its two rightmost columns.

    You many only use numpy, which is imported as np, for this question. Access all functions needed via this name (np)
    as any additional import statements will be removed by the autograder.

    :param N: integer representing the size of the NxN chessboard
    :return: numpy array of shape (N,) containing an initial solution using greedy min-conflicts (this may contain
    conflicts). The i-th entry's value j represents the row  given as 0 <= j < N.
    """
    greedy_init = np.zeros(N)
    # First queen goes in a random spot
    greedy_init[0] = int(np.random.randint(0, N))

    ### YOUR CODE GOES HERE
    first_row = int(greedy_init[0])
    horizontal_Q = np.zeros(N)
    posdiag_Q = np.zeros(N)
    negdiag_Q = np.zeros(N)
    horizontal_Q[first_row] += 1
    if first_row + 1 < N:
        posdiag_Q[first_row+1] += 1
    if first_row - 1 >= 0:
        negdiag_Q[first_row-1] += 1
    # print('first_row: ', first_row)
    # print('horizon: ', horizontal_Q)
    # print('posdiag: ', posdiag_Q)
    # print('negdiag: ', negdiag_Q)
    for col in range(1, N):
        overall_Q = horizontal_Q + posdiag_Q + negdiag_Q
        # print('overall: ', overall_Q)
        # print('')
        min_index = np.argwhere(overall_Q == overall_Q.min())
        rand_index = np.random.choice(min_index.shape[0])
        row_num = min_index[rand_index][0]
        greedy_init[col] = int(row_num)
        # print('row_num: ', row_num)

        posdiag_Q = np.roll(posdiag_Q, 1)
        posdiag_Q[0] = 0
        negdiag_Q = np.roll(negdiag_Q, -1)
        negdiag_Q[-1] = 0
        horizontal_Q[row_num] += 1
        if row_num + 1 < N:
            posdiag_Q[row_num + 1] += 1
        if row_num - 1 >= 0:
            negdiag_Q[row_num - 1] += 1
        # print('horizon: ', horizontal_Q)
        # print('posdiag: ', posdiag_Q)
        # print('negdiag: ', negdiag_Q)

    return greedy_init.astype(int)


def no_longer_avail(board, row_num, col_num):
    board[row_num][1:] = -1             # this row is no longer available
    board[:row_num, col_num] = -1
    board[(row_num+1):, col_num] = -1   # this col is no longer available
    N = board.shape[0]
    for i in range(N):
        if (row_num-i>0) and (col_num-i>0):
            board[row_num-i][col_num-i] = -1
        else:
            break
    for i in range(N):
        if (row_num+i<N) and (col_num+i<N):
            board[row_num + i][col_num + i] = -1
        else:
            break
    return board


if __name__ == '__main__':
    # You can test your code here
    greedy_init = initialize_greedy_n_queens(5)
    print(greedy_init)
    support.plot_n_queens_solution(greedy_init)