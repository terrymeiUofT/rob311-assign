from collections import deque
import numpy as np
from search_problems import Node, GraphSearchProblem


def overlap(past, past_r):
    for state in past:
        if str(state) in past_r:
            return [True, past[state], past_r[state]]
    return [False, None, None]


def bidirectional_search(problem):
    """
        Implement a bidirectional search algorithm that takes instances of SimpleSearchProblem (or its derived
        classes) and provides a valid and optimal path from the initial state to the goal state.

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

    past_r = {}
    queue_r = {}
    cur_state_r = problem.goal_states[0]
    # create init node
    action_list_r = problem.get_actions(cur_state_r)
    init_node_r = Node(None, cur_state_r, action_list_r, 1)
    cur_node_q_r = [init_node_r]

    overlapped = False
    while not overlapped:
        cur_node = cur_node_q.pop(0)
        past.update({str(cur_node.state): cur_node})
        for action in cur_node.action:
            child_state = problem.transition(cur_node.state, action)
            # create child node
            child_node = Node(cur_node, child_state, problem.get_actions(child_state), 1)
            if (str(child_state) not in past) and (str(child_state) not in queue):
                cur_node_q.append(child_node)
                queue.update({str(child_state): child_node})
                if len(cur_node_q) > max_frontier_size:
                    max_frontier_size = len(cur_node_q)
        num_nodes_expanded += 1

        cur_node_r = cur_node_q_r.pop(0)
        past_r.update({str(cur_node_r.state): cur_node_r})
        for action in cur_node_r.action:
            child_state_r = problem.transition(cur_node_r.state, action)
            # create child node
            child_node_r = Node(cur_node_r, child_state_r, problem.get_actions(child_state_r), 1)
            if (str(child_state_r) not in past_r) and (str(child_state_r) not in queue_r):
                cur_node_q_r.append(child_node_r)
                queue_r.update({str(child_state_r): child_node_r})
                if len(cur_node_q_r) > max_frontier_size:
                    max_frontier_size = len(cur_node_q_r)
        num_nodes_expanded += 1
        overlapped, mid_node, mid_node_r = overlap(past, past_r)

    # find the middle node and half of the path
    path.append(mid_node.state)
    root_node = mid_node
    while (root_node.state != problem.init_state):
        path.append(root_node.parent.state)
        root_node = root_node.parent
    path.reverse()

    # find the other half of the path
    path_r = []
    root_node = mid_node_r
    while (root_node.state != problem.goal_states[0]):
        path_r.append(root_node.parent.state)
        root_node = root_node.parent
    path += path_r

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
    path, num_nodes_expanded, max_frontier_size = bidirectional_search(problem)
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
    path, num_nodes_expanded, max_frontier_size = bidirectional_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)
    print('number of nodes expanded: ', num_nodes_expanded)
    print('max frontier size: ', max_frontier_size)

    # Be sure to compare with breadth_first_search!