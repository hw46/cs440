from collections import deque
from queue import PriorityQueue

# submitted.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# submitted should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi)


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement bfs function
    
    start = maze.start
    goal = maze.waypoints[0]
    queue = deque([(start, [start])])
    visited = set()
    visited.add(start)

    while queue:
        current_position, path_to_current = queue.popleft()
        if current_position == goal:
            return path_to_current
        for neighbor in maze.neighbors_all(*current_position):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path_to_current + [neighbor]))
    
    return []

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement astar_single
    
    start = maze.start
    goal = maze.waypoints[0]
    frontier = PriorityQueue()
    frontier.put((0 + max(abs(start[0] - goal[0]), abs(start[1] - goal[1])), 0, start, [start]))
    g_score = {start: 0}

    while not frontier.empty():
        _, current_g, current_position, path_to_current = frontier.get()
        if current_position == goal:
            return path_to_current
        for neighbor in maze.neighbors_all(*current_position):
            tentative_g_score = current_g + 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + max(abs(neighbor[0] - goal[0]), abs(neighbor[1] - goal[1]))
                frontier.put((f_score, tentative_g_score, neighbor, path_to_current + [neighbor]))
                
    return []

# This function is for Extra Credits, please begin this part after finishing previous two functions
def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start = maze.start
    waypoints = set(maze.waypoints)

    def heuristic(current_position, remaining_waypoints):
        if not remaining_waypoints:
            return 0
        nearest_wp_dist = min(chebyshev_distance(current_position, wp) for wp in remaining_waypoints)
        mst_cost = mst(remaining_waypoints | {current_position})
        return nearest_wp_dist + mst_cost

    def chebyshev_distance(a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def mst(waypoints):
        if not waypoints:
            return 0
        waypoints = list(waypoints)
        in_mst = [False] * len(waypoints)
        edge_cost = [float('inf')] * len(waypoints)
        edge_cost[0] = 0
        total_cost = 0
        for _ in range(len(waypoints)):
            min_cost = float('inf')
            for i in range(len(waypoints)):
                if not in_mst[i] and edge_cost[i] < min_cost:
                    min_cost = edge_cost[i]
                    min_index = i
            in_mst[min_index] = True
            total_cost += min_cost
            for i in range(len(waypoints)):
                if not in_mst[i]:
                    distance = chebyshev_distance(waypoints[min_index], waypoints[i])
                    if distance < edge_cost[i]:
                        edge_cost[i] = distance
        return total_cost

    frontier = PriorityQueue()
    frontier.put((0, start, frozenset(), [start]))
    visited = set()

    while not frontier.empty():
        current_f, current_position, visited_waypoints, path = frontier.get()
        if visited_waypoints == waypoints:
            return path
        if (current_position, visited_waypoints) in visited:
            continue
        visited.add((current_position, visited_waypoints))
        for next_position in maze.neighbors_all(*current_position):
            next_visited_waypoints = visited_waypoints | ({next_position} if next_position in waypoints else set())
            if (next_position, next_visited_waypoints) not in visited:
                g = len(path)
                h = heuristic(next_position, waypoints - next_visited_waypoints)
                frontier.put((g + h, next_position, next_visited_waypoints, path + [next_position]))

    return []