import queue
import numpy as np
import matplotlib.pyplot as plt
from search_problems import Node, GridSearchProblem, get_random_grid_problem


def a_star_search(problem):
    """
    Uses the A* algorithm to solve an instance of GridSearchProblem. Use the methods of GridSearchProblem along with
    structures and functions from the allowed imports (see above) to implement A*.

    :param problem: an instance of GridSearchProblem to solve
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by your search
             max_frontier_size: maximum frontier size during search
    """
    ####
    #   COMPLETE THIS CODE
    ####
    num_nodes_expanded = 0
    max_frontier_size = 0
    path = []

    past = {}
    frontier = {}
    cur_state = problem.init_state
    # create init node
    action_list = problem.get_actions(cur_state)
    init_node = Node(None, cur_state, action_list, 1)
    cur_node_q = queue.PriorityQueue()
    cur_node_q.put((problem.heuristic(init_node.state), init_node))
    cur_node = init_node

    while True:
        if cur_node_q.empty():  # failure to find a path
            return [], num_nodes_expanded, max_frontier_size

        cur_f, cur_node = cur_node_q.get()
        if problem.goal_test(cur_node.state):
            # backtrack to find the shortest path
            path = problem.trace_path(cur_node, problem.init_state)
            return path, num_nodes_expanded, max_frontier_size

        past.update({str(cur_node.state): cur_node})
        for action in problem.get_actions(cur_node.state):
            child_node = problem.get_child_node(cur_node, action)
            num_nodes_expanded += 1
            if (past.get(str(child_node.state)) is None) and (frontier.get(str(child_node.state)) is None):
                h_cost = problem.heuristic(child_node.state)
                cur_node_q.put((child_node.path_cost + h_cost, child_node))
                frontier.update({str(child_node.state): child_node})
            elif frontier.get(str(child_node.state)) is not None:
                if child_node.path_cost < frontier.get(str(child_node.state)).path_cost:
                    frontier.update({str(child_node.state): child_node})


def search_phase_transition():
    """
    Simply fill in the prob. of occupancy values for the 'phase transition' and peak nodes expanded within 0.05. You do
    NOT need to submit your code that determines the values here: that should be computed on your own machine. Simply
    fill in the values!

    :return: tuple containing (transition_start_probability, transition_end_probability, peak_probability)
    """
    ####
    #   REPLACE THESE VALUES
    ####
    transition_start_probability = 0.3
    transition_end_probability = 0.5
    peak_nodes_expanded_probability = 0.4
    return transition_start_probability, transition_end_probability, peak_nodes_expanded_probability


if __name__ == '__main__':
    # Test your code here!
    # Create a random instance of GridSearchProblem
    p_occ = 0.3
    M = 100
    N = 100
    # problem = get_random_grid_problem(p_occ, M, N)
    # # Solve it
    # path, num_nodes_expanded, max_frontier_size = a_star_search(problem)
    #
    # # Check the result
    # correct = problem.check_solution(path)
    # print("Solution is correct: {:}".format(correct))
    # # Plot the result
    # problem.plot_solution(path)

    # Experiment and compare with BFS
    x = []
    solvability = []
    effort = []
    for i in np.arange(0.1, 0.95, 0.05):
        x.append(i)
        node_count = 0
        solved = 0
        for j in range(100):
            problem = get_random_grid_problem(i, M, N)
            path, num_nodes_expanded, max_frontier_size = a_star_search(problem)
            node_count += num_nodes_expanded
            if path != []:
                solved = solved + 1
        effort.append(node_count / 100)
        solvability.append(solved / 100)

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(x, solvability)
    ax[1].plot(x, effort)
    plt.show()