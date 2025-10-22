# %%
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from networkx.algorithms.community import louvain_communities
import numpy as np

# %%
# load edge list
df = pd.read_csv('edges.csv')

# build weighted graph
G = nx.Graph()
edges = [(row['concept_A'], row['concept_B'], row['n_people']) for _, row in df.iterrows()]
G.add_weighted_edges_from(edges)

# %%
# basic graph statistics
print(f"Total conditions: {G.number_of_nodes()}")
print(f"Total co-occurences: {G.number_of_edges()}")
print(f"Network density: {nx.density(G):.4f}")
print(f"Is connected: {nx.is_connected(G)}")

# %% [markdown]
# <h1>1. Centrality analysis  </h1>
# Analyze nodes by weighted degree centrality and betweenness centrality. Nodes with high degree co-occur with many other conditions, especially when they co-occur in a larger number of patients. Nodes with high betweenness serve as bridges between different co-occurrences, indicating that their development may be associated with a new set of potential co-occurrences. 

# %%
# compute node centralities
degree_centrality = dict(G.degree(weight='weight'))
betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
betweenness_centrality = nx.betweenness_centrality(G, weight='weight')

# %%
# show top 20 diseases by degree centrality
print("20 conditions with most widespread co-occurrence:\n")
sorted_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
for i, (node, cent) in enumerate(sorted_degree[:20], 1):
    degree = G.degree(node)
    print(f"{i:2d}. {node:50s} | Weighted Centrality: {cent} | # Connections: {degree}")

# %%
print("20 conditions that most strongly bridge different comorbidity groups)\n")
sorted_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
for i, (node, cent) in enumerate(sorted_betweenness[:20], 1):
    print(f"{i:2d}. {node:50s} | Centrality: {cent:.4f}")

# %% [markdown]
# <h1>2. First-level community analysis</h1>

# %%
# apply Louvain community detection algorithm
# search a range of resolution values to find best modularity
best_R = 0.0
best_mod = -1.0
for R in np.arange(0.0, 3.1, 0.1):
    comms = louvain_communities(G, weight='weight', seed = 42, resolution=R)
    mod = nx.algorithms.community.modularity(G, comms, weight='weight')
    if mod > best_mod:
        best_mod = mod
        best_R = R

# apply most optimal resolution to get main communities
main_communities = louvain_communities(G, weight='weight', seed=42, resolution=best_R)

# create map of condition node to main community
main_community_map = {}
for idx, community in enumerate(main_communities):
    for node in community:
        main_community_map[node] = idx

# %%
# calculate modularity of partition
modularity = nx.algorithms.community.modularity(G, main_communities, weight='weight')
print(f"Number of main communities detected: {len(main_communities)}")
print(f"Modularity score: {modularity:.4f}\n")

main_community_sizes = sorted([(i, len(comm)) for i, comm in enumerate(main_communities)], 
                               key=lambda x: x[1], reverse=True)
print("Main community sizes:")
for comm_id, size in main_community_sizes:
    print(f"  Community {comm_id}: {size} conditions")

# %% [markdown]
# <h1>3. Sub-communities within main communities</h1>

# %%
# functionalize subcommunity processing to be used in the cell below
def process_subcommunity(nodes, subcom_id):
    subcom_nodes = list(nodes)
    # assign nodes to the subcommunity map
    for node in subcom_nodes:
        subcommunity_map[node] = subcom_id

    # compute average centralities
    subcom_degree_avg = np.mean([degree_centrality[n] for n in subcom_nodes])
    subcom_between_avg = np.mean([betweenness_centrality[n] for n in subcom_nodes])

    # nodes will be sorted by degree centrality
    sorted_nodes = sorted(subcom_nodes, key=lambda x: degree_centrality[x], reverse=True)

    # print summary
    print(f"  Sub-community {subcom_id} ({len(subcom_nodes)} nodes):")
    print(f"    Avg degree centrality: {subcom_degree_avg:.4f}")
    print(f"    Avg betweenness centrality: {subcom_between_avg:.4f}")
    # show up to 10 top nodes
    print("    Top representative nodes:")
    top_nodes = []
    i = 0
    while i < min(10, len(sorted_nodes)):
        print(f"      - {sorted_nodes[i]}")
        top_nodes.append(sorted_nodes[i])
        i += 1

    # append structured info
    subcommunity_info.append({
        'main_comm': main_comm_id,
        'sub_comm': subcom_id,
        'size': len(subcom_nodes),
        'top_nodes': top_nodes,
        'avg_degree': subcom_degree_avg,
        'avg_betweenness': subcom_between_avg
    })

    print()

# %%
# prepare to store sub-community information
subcommunity_map = {}
subcommunity_counter = 0
subcommunity_info = []

# iterate over main communities
for main_comm_id, size in main_community_sizes:
    if size < 20:  # don't subdivide very small communities
        print(f"Main Community {main_comm_id} ({size} nodes): Too small to subdivide\n")
        # process the whole community as one sub-community x.0
        nodes = main_communities[main_comm_id]
        process_subcommunity(nodes, f"{main_comm_id}.0")
        continue
    
    print(f"Main Community {main_comm_id} ({size} nodes):")
    print("-" * 70)
    
    # extract subgraph for this community
    subgraph = G.subgraph(main_communities[main_comm_id]).copy()
    
    # search a range of resolution values to find best modularity
    best_R = 1.0
    best_mod = -1.0
    for R in np.arange(1, 3.1, 0.1):
        comms = louvain_communities(subgraph, weight='weight', seed=42, resolution=R)
        mod = nx.algorithms.community.modularity(subgraph, comms, weight='weight')
        if mod > best_mod:
            best_mod = mod
            best_R = R

    # detect sub-communities with higher resolution for finer granularity
    sub_communities = louvain_communities(subgraph, weight='weight', seed=42, resolution=best_R)
    sub_modularity = nx.algorithms.community.modularity(subgraph, sub_communities, weight='weight')
    
    print(f"  Found {len(sub_communities)} sub-communities (modularity: {sub_modularity:.3f})\n")
    
    # sort sub-communities by size
    sorted_subcoms = sorted(enumerate(sub_communities), key=lambda x: len(x[1]), reverse=True)
    
    # iterate over sub-communities
    for sub_idx, (original_idx, subcom) in enumerate(sorted_subcoms):
        subcom_id = f"{main_comm_id}.{sub_idx}"
        process_subcommunity(subcom, subcom_id)

# %% [markdown]
# <h1>4. Graph plot</h1>

# %%
# Add node attributes
nx.set_node_attributes(G, degree_centrality, 'degree_centrality')
nx.set_node_attributes(G, betweenness_centrality, 'betweenness_centrality')
nx.set_node_attributes(G, main_community_map, 'main_community')
nx.set_node_attributes(G, subcommunity_map, 'subcommunity')

# Layout
pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)

# Prepare edge traces
edge_x, edge_y = [], []
for u, v in G.edges():
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#ddd'),
    hoverinfo='none',
    mode='lines'
)

# Node coordinates and attributes
node_x, node_y = [], []
node_text = []
node_color = []
node_size = []

# Extended color palette for main communities
palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    
    deg_cent = degree_centrality[node]
    betw_cent = betweenness_centrality[node]
    main_comm = main_community_map[node]
    sub_comm = subcommunity_map[node]
    
    comm_color = palette[main_comm % len(palette)]
    node_color.append(comm_color)
    
    # Size nodes by degree centrality
    node_size.append(10 + deg_cent * 50)
    
    # Hover text
    neighbors = list(G.neighbors(node))
    neighbor_comms = set(main_community_map[n] for n in neighbors)
    
    hover_text = (
        f"<b>{node}</b><br>"
        f"Degree Centrality: {deg_cent:.4f}<br>"
        f"Betweenness Centrality: {betw_cent:.4f}<br>"
        f"Main Community: {main_comm}<br>"
        f"Sub-community: {sub_comm}<br>"
        f"Connections: {len(neighbors)}<br>"
        f"Bridges {len(neighbor_comms)} main communities"
    )
    node_text.append(hover_text)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=False,
        color=node_color,
        size=node_size,
        line=dict(width=1, color='white')
    ),
    hovertext=node_text,
)

# Build figure
fig = go.Figure(
    data=[edge_trace, node_trace],
    layout=go.Layout(
        title=dict(
            text=f'Under-30 Mortality Disease Co-occurrence Network<br><sub>{len(main_communities)} main communities, hierarchical sub-structure (modularity: {modularity:.3f})</sub>',
            font=dict(size=16)
        ),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=60),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )
)
fig.show()

print("\nVisualization Notes:")
print("- Node size represents degree centrality")
print("- Node color represents main community membership")
print("- Hover over nodes to see both main community and sub-community")
print("- Sub-communities provide finer clinical granularity for analysis")


