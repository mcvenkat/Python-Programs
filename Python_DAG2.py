import networkx as nx
from matplotlib import pyplot as plt

g1 = nx.DiGraph()
g1.add_edges_from([("root", "a"), ("a", "b"), ("a", "e"), ("b", "c"), ("b", "d"), ("d", "e"), ("e", "f")])
plt.tight_layout()
nx.draw_networkx(g1, arrows=True)
plt.savefig("g1.jpg", format="JPG")
# tell matplotlib you're done with the plot: https://stackoverflow.com/questions/741877/how-do-i-tell-matplotlib-that-i-am-done-with-a-plot
plt.clf()
