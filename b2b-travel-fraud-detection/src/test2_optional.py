import networkx as nx

G = nx.read_edgelist("/Users/shashwata/Documents/Data Science Projects/B2B Travel Fraud/VoyageHack/Data/graph/edges.txt", nodetype=int)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Min degree:", min(dict(G.degree()).values()))
