import queue
import numpy as np
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
    while not problem.goal_test(cur_node.state):
        cur_f, cur_node = cur_node_q.get()
        cur_h = problem.heuristic(cur_node.state)
        past.update({str(cur_node.state): cur_node})
        for action in cur_node.action:
            child_state = problem.transition(cur_node.state, action)
            if (past.get(str(child_state)) is None) and (frontier.get(str(child_state)) is None):
                # create child node
                child_node = Node(cur_node, child_state, problem.get_actions(child_state), 1)
                h_cost = problem.heuristic(child_state)
                g_cost = cur_f - cur_h
                cur_node_q.put((h_cost + g_cost + 1, child_node))
                frontier.update({str(child_state): child_node})
                # if cur_node_q.qsize() > max_frontier_size:
                #     max_frontier_size = cur_node_q.qsize()
        num_nodes_expanded += 1

    # backtrack to find the shortest path
    path = problem.trace_path(cur_node, problem.init_state)

    return path, num_nodes_expanded, max_frontier_size


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
    transition_start_probability = -1.0
    transition_end_probability = -1.0
    peak_nodes_expanded_probability = -1.0
    return transition_start_probability, transition_end_probability, peak_nodes_expanded_probability


if __name__ == '__main__':
    # Test your code here!
    # Create a random instance of GridSearchProblem
    p_occ = 0.25
    M = 30
    N = 30
    problem = get_random_grid_problem(p_occ, M, N)
    # Solve it
    path, num_nodes_expanded, max_frontier_size = a_star_search(problem)
    # Check the result
    correct = problem.check_solution(path)
    print("Solution is correct: {:}".format(correct))
    # Plot the result
    problem.plot_solution(path)

    # Experiment and compare with BFS