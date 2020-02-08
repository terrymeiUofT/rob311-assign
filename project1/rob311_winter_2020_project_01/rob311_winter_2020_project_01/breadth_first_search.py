from collections import deque
import numpy as np
from search_problems import Node, GraphSearchProblem


def breadth_first_search(problem):
    """
    Implement a simple breadth-first search algorithm that takes instances of SimpleSearchProblem (or its derived
    classes) and provides a valid and optimal path from the initial state to the goal state. Useful for testing your
    bidirectional and A* search algorithms.

    :param problem: instance of SimpleSearchProblem
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by your search
             max_frontier_size: maximum frontier size during search
    """
    ####
    #   COMPLETE THIS CODE
    ####
    max_frontier_size = 0
    num_nodes_expanded = 0
    path = []

    past = {}
    queue = {}
    cur_state = problem.init_state
    # create init node
    action_list = problem.get_actions(cur_state)
    init_node = Node(None, cur_state, action_list, 1)
    cur_node_q = [init_node]
    cur_node = cur_node_q[0]
    while not problem.goal_test(cur_node.state):
        cur_node = cur_node_q.pop(0)
        past.update({str(cur_node.state): cur_node})
        for action in cur_node.action:
            child_state = problem.transition(cur_node.state, action)
            if (past.get(str(child_state)) is None) and (queue.get(str(child_state)) is None):
                # create child node
                child_node = Node(cur_node, child_state, problem.get_actions(child_state), 1)
                cur_node_q.append(child_node)
                queue.update({str(child_state): child_node})
                # if len(cur_node_q) > max_frontier_size:
                #     max_frontier_size = len(cur_node_q)
        num_nodes_expanded += 1

    # backtrack to find the shortest path
    path = problem.trace_path(cur_node, problem.init_state)

    return path, num_nodes_expanded, max_frontier_size


if __name__ == '__main__':
    # Simple example
    goal_states = [0]
    init_state = 9
    V = np.arange(0, 10)
    E = np.array([[0, 1],
                  [1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5],
                  [5, 6],
                  [6, 7],
                  [7, 8],
                  [8, 9],
                  [0, 6],
                  [1, 7],
                  [2, 5],
                  [9, 4]])
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)
    print('number of nodes expanded: ', num_nodes_expanded)
    print('max frontier size: ', max_frontier_size)

    # Use stanford_large_network_facebook_combined.txt to make your own test instances
    E = np.loadtxt('./stanford_large_network_facebook_combined.txt', dtype=int)
    V = np.unique(E)
    goal_states = [349]
    init_state = 0
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)
    print('number of nodes expanded: ', num_nodes_expanded)
    print('max frontier size: ', max_frontier_size)

    # twitter dataset
    E_twitter = np.load('twitter_edges_project_01.npy')
    V_twitter = np.unique(E_twitter)
    twitter_problem = GraphSearchProblem([59999], 0, V_twitter, E_twitter)
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(twitter_problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)
    print('number of nodes expanded: ', num_nodes_expanded)
    print('max frontier size: ', max_frontier_size)