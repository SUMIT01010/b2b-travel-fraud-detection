import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

# Load the DataFrame from the pickle file
df = pd.read_pickle("/Users/shashwata/Documents/Data Science Projects/B2B Travel Fraud/VoyageHack/Data/processed/merged.pkl")

# Initiating node registry
node_map = {}
node_type = {}
current = 0

def register(node, ntype):
    global current
    if node not in node_map:
        node_map[node] = current
        node_type[current] = ntype
        current += 1

# Registering all nodes
print("Registering nodes...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    register(f"A_{row.agency_id}", "agency")
    register(f"U_{row.user_id}", "user")
    register(f"D_{row.device_id}", "device")
    register(f"I_{row.ip_id}", "ip")
    register(f"B_{row.booking_id}", "booking")

print("Total nodes:", len(node_map))

# Build Edge List
edges = []

print("Building edges...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    b = node_map[f"B_{row.booking_id}"]

    edges.append((node_map[f"A_{row.agency_id}"], b))
    edges.append((node_map[f"U_{row.user_id}"], b))
    edges.append((node_map[f"D_{row.device_id}"], b))
    edges.append((node_map[f"I_{row.ip_id}"], b))

# Making Undirected Graph
edges = edges + [(j, i) for i, j in edges]

# Saving Edges
edges = np.array(edges)
np.savetxt("/Users/shashwata/Documents/Data Science Projects/B2B Travel Fraud/VoyageHack/Data/graph/edges.txt", edges, fmt="%d")

# Saving Mappings
with open("/Users/shashwata/Documents/Data Science Projects/B2B Travel Fraud/VoyageHack/Data/graph/node_map.pkl", "wb") as f:
    pickle.dump(node_map, f)

with open("/Users/shashwata/Documents/Data Science Projects/B2B Travel Fraud/VoyageHack/Data/graph/node_type.pkl", "wb") as f:
    pickle.dump(node_type, f)


# Sanity Checks
print("Total edges:", len(edges))

from collections import Counter
print("Node type counts:")
print(Counter(node_type.values()))



