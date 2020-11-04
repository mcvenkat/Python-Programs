import networkx as nx
from matplotlib import pyplot as plt

graph = nx.DiGraph()
graph.add_edges_from([("root", "a"), ("a", "b"), ("a", "e"), ("b", "c"), ("b", "d"), ("d", "e")])

graph.nodes() # => NodeView(('root', 'a', 'b', 'e', 'c', 'd'))

nx.shortest_path(graph, 'root', 'e') # => ['root', 'a', 'e']

nx.dag_longest_path(graph) # => ['root', 'a', 'b', 'd', 'e']

list(nx.topological_sort(graph)) # => ['root', 'a', 'b', 'd', 'e', 'c']

nx.is_directed(graph)

nx.is_directed_acyclic_graph(graph)
