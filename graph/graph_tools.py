from __future__ import print_function
# Author: Kyle Kastner
# License: BSD 3-Clause

# Using code modified from the following authors, collected in one place
# http://www.gilles-bertrand.com/2014/03/dijkstra-algorithm-python-example-source-code-shortest-path.html
# http://eddmann.com/posts/depth-first-search-and-breadth-first-search-in-python/
# https://gist.github.com/joninvski/701720https://gist.github.com/joninvski/701720
# https://jlmedina123.wordpress.com/2014/05/17/floyd-warshall-algorithm-in-python/
# http://code.activestate.com/recipes/119466-dijkstras-algorithm-for-shortest-paths/
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import subprocess
import os


def pwrap(args, shell=False):
    p = subprocess.Popen(args, shell=shell, stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
    return p

# Print output
# http://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
def execute(cmd, shell=False):
    popen = pwrap(cmd, shell=shell)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line

    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def pe(cmd, shell=False):
    """
    Print and execute command on system
    """
    for line in execute(cmd, shell=shell):
        print(line, end="")


def _paths(graph, start, end, pop):
    # dfs or bfs depending on pop
    q = [(start, [start])]
    while q:
        if pop is None:
            (vertex, path) = q.pop()
        else:
            (vertex, path) = q.pop(0)
        for nx in set(graph[vertex].keys()) - set(path):
            if nx == end:
                yield path + [nx]
            else:
                q.append((nx, path + [nx]))


def dfs_paths(graph, start, end):
    return _paths(graph, start, end, None)


def bfs_paths(graph, start, end):
    return _paths(graph, start, end, 0)


def initialize_bf(graph, source):
    d = {} # Stands for destination
    p = {} # Stands for predecessor
    for node in graph:
        d[node] = float("inf")
        p[node] = None
    d[source] = 0
    return d, p


def relax_bf(node, neighbour, graph, d, p):
    if d[neighbour] > d[node] + graph[node][neighbour]:
        d[neighbour] = d[node] + graph[node][neighbour]
        p[neighbour] = node


def bellman_ford_paths(graph, source):
    # returns distances and paths
    d, p = initialize_bf(graph, source)
    for i in range(len(graph)-1):
        for u in graph:
            for v in graph[u]:
                relax_bf(u, v, graph, d, p)

    # Check for negative-weight cycles
    for u in graph:
        for v in graph[u]:
            assert d[v] <= d[u] + graph[u][v]
    return d, p


def floyd_warshall_paths(graph):
    # returns distances and paths
    # Initialize dist and pred:
    # copy graph into dist, but add infinite where there is
    # no edge, and 0 in the diagonal
    dist = {}
    pred = {}
    for u in graph:
        dist[u] = {}
        pred[u] = {}
        for v in graph:
            dist[u][v] = float("inf")
            pred[u][v] = -1
        dist[u][u] = 0
        for neighbor in graph[u]:
            dist[u][neighbor] = graph[u][neighbor]
            pred[u][neighbor] = u

    for t in graph:
        # given dist u to v, check if path u - t - v is shorter
        for u in graph:
            for v in graph:
                newdist = dist[u][t] + dist[t][v]
                if newdist < dist[u][v]:
                    dist[u][v] = newdist
                    pred[u][v] = pred[t][v] # route new path through t
    return dist, pred


def dijkstra_path(graph, start, end, visited=[], distances={}, predecessors={}):
    """Find the shortest path between start and end nodes in a graph"""
    # we've found our end node, now find the path to it, and return
    if start == end:
        path = []
        while end != None:
            path.append(end)
            end = predecessors.get(end, None)
        return distances[start], path[::-1]
    # detect if it's the first time through, set current distance to zero
    if not visited:
        distances[start] = 0

    # process neighbors as per algorithm, keep track of predecessors
    for neighbor in graph[start]:
        if neighbor not in visited:
            neighbordist = distances.get(neighbor, float("inf"))
            tentativedist = distances[start] + graph[start][neighbor]
            if tentativedist < neighbordist:
                distances[neighbor] = tentativedist
                predecessors[neighbor] = start
    # neighbors processed, now mark the current node as visited
    visited.append(start)
    # finds the closest unvisited node to the start
    unvisiteds = dict((k, distances.get(k, float("inf"))) for k in graph if k not in visited)
    closestnode = min(unvisiteds, key=unvisiteds.get)
    # now we can take the closest node and recurse, making it current
    return dijkstra_path(graph, closestnode, end, visited, distances, predecessors)


def graphviz_plot(graph, fname="tmp_dotgraph.dot", show=True):
    if os.path.exists(fname):
        print("WARNING: Overwriting existing file {} for new plots".format(fname))
    f = open(fname,'w')
    f.writelines('digraph G {\nnode [width=.3,height=.3,shape=octagon,style=filled,color=skyblue];\noverlap="false";\nrankdir="LR";\n')
    for i in graph:
        for j in graph[i]:
            s= '      '+ i
            s +=  ' -> ' +  j + ' [label="' + str(graph[i][j]) + '"]'
            s+=';\n'
            f.writelines(s)
    f.writelines('}')
    f.close()
    graphname = fname.split(".")[0] + ".png"
    pe(["dot", "-Tpng", fname, "-o", graphname])

    if show:
        plt.imshow(mpimg.imread(graphname))
        plt.show()


def test_graph_tools():
    graph = {'s': {'a': 2, 'b': 1},
             'a': {'s': 3, 'b': 4, 'c': 8},
             'b': {'s': 4, 'a': 2, 'd': 2},
             'c': {'a': 2, 'd': 7, 't': 4},
             'd': {'b': 1, 'c': 11, 't': 5},
             't': {'c': 4, 'd': 5}}

    print([p for p in bfs_paths(graph, 'a', 't')])
    print([p for p in dfs_paths(graph, 'a', 't')])
    print(dijkstra_path(graph, 'a', 't'))
    print(floyd_warshall_paths(graph))
    print(bellman_ford_paths(graph, 'a'))
    graphviz_plot(graph)


if __name__ == "__main__":
   test_graph_tools()
