import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from collections import Counter

def plot_network(G, layout="spring", highlight_nodes=None, highlight_path=None, max_nodes=500):
    """
    Plot a network visualization.
    
    Args:
        G (networkx.Graph): The network to visualize
        layout (str): Layout algorithm to use
        highlight_nodes (list): Nodes to highlight
        highlight_path (list): Path to highlight
        max_nodes (int): Maximum number of nodes to display
    """
    # For large networks, sample a subset of nodes
    if G.number_of_nodes() > max_nodes:
        st.warning(f"Network is too large to visualize completely. Showing a sample of {max_nodes} nodes.")
        
        # If there's a highlight path, make sure those nodes are included
        if highlight_path:
            path_nodes = set(highlight_path)
            remaining_nodes = set(G.nodes()) - path_nodes
            
            # Select random nodes to complete the sample
            sample_size = max_nodes - len(path_nodes)
            if sample_size > 0 and len(remaining_nodes) > 0:
                sampled_remaining = np.random.choice(list(remaining_nodes), 
                                                    size=min(sample_size, len(remaining_nodes)), 
                                                    replace=False)
                sampled_nodes = list(path_nodes) + list(sampled_remaining)
            else:
                sampled_nodes = list(path_nodes)
        
        # If there are highlight nodes, make sure those are included
        elif highlight_nodes:
            highlight_set = set(highlight_nodes)
            remaining_nodes = set(G.nodes()) - highlight_set
            
            # Select random nodes to complete the sample
            sample_size = max_nodes - len(highlight_set)
            if sample_size > 0 and len(remaining_nodes) > 0:
                sampled_remaining = np.random.choice(list(remaining_nodes), 
                                                    size=min(sample_size, len(remaining_nodes)), 
                                                    replace=False)
                sampled_nodes = list(highlight_set) + list(sampled_remaining)
            else:
                sampled_nodes = list(highlight_set)
        
        # If no specific nodes to highlight, just sample randomly
        else:
            sampled_nodes = np.random.choice(list(G.nodes()), 
                                            size=min(max_nodes, G.number_of_nodes()), 
                                            replace=False)
        
        # Create a subgraph with the sampled nodes
        subgraph = G.subgraph(sampled_nodes).copy()
    else:
        subgraph = G
    
    # Compute layout
    if layout == "spring":
        pos = nx.spring_layout(subgraph, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(subgraph)
    elif layout == "circular":
        pos = nx.circular_layout(subgraph)
    elif layout == "spectral":
        pos = nx.spectral_layout(subgraph)
    else:
        pos = nx.spring_layout(subgraph, seed=42)
    
    # Create a figure using Plotly for interactive visualization
    edge_x = []
    edge_y = []
    
    # Add edges to the plot
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Add nodes to the plot
    node_x = []
    node_y = []
    for node in subgraph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    # Set default node color and size
    node_color = ['#1f77b4'] * len(node_x)  # Default blue
    node_size = [10] * len(node_x)          # Default size
    
    # Highlight specific nodes if provided
    if highlight_nodes:
        for i, node in enumerate(subgraph.nodes()):
            if node in highlight_nodes:
                node_color[i] = '#ff7f0e'  # Orange
                node_size[i] = 20          # Larger size
    
    # Highlight path if provided
    if highlight_path:
        path_set = set(highlight_path)
        for i, node in enumerate(subgraph.nodes()):
            if node in path_set:
                node_color[i] = '#ff7f0e'  # Orange
                node_size[i] = 15          # Larger size
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(width=1, color='#333')
        ),
        text=[f"Node ID: {node}<br>Degree: {subgraph.degree(node)}" for node in subgraph.nodes()]
    )
    
    # If there's a path to highlight, add lines for the path
    path_edge_traces = []
    if highlight_path and len(highlight_path) > 1:
        path_x = []
        path_y = []
        
        for i in range(len(highlight_path) - 1):
            u, v = highlight_path[i], highlight_path[i+1]
            if u in pos and v in pos:  # Make sure nodes are in the position dictionary
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                path_x.extend([x0, x1, None])
                path_y.extend([y0, y1, None])
        
        if path_x:  # Only add the trace if there are edges to highlight
            path_edge_trace = go.Scatter(
                x=path_x, y=path_y,
                line=dict(width=2.5, color='#d62728'),  # Red
                hoverinfo='none',
                mode='lines'
            )
            path_edge_traces.append(path_edge_trace)
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace, *path_edge_traces],
                   layout=go.Layout(
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                   ))
    
    fig.update_layout(title="Network Visualization")
    
    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def plot_degree_distribution(G):
    """
    Plot the degree distribution of the network.
    
    Args:
        G (networkx.Graph): The network to analyze
    """
    degrees = [d for _, d in G.degree()]
    degree_counts = Counter(degrees)
    
    # Convert to DataFrame for plotting
    df = pd.DataFrame({"Degree": list(degree_counts.keys()), 
                       "Count": list(degree_counts.values())})
    df = df.sort_values("Degree")
    
    # Create plot
    fig = px.bar(df, x="Degree", y="Count", log_y=True,
                 labels={"Degree": "Node Degree", "Count": "Number of Nodes (log scale)"},
                 title="Degree Distribution")
    
    fig.update_layout(
        xaxis_title="Node Degree",
        yaxis_title="Number of Nodes (log scale)",
        xaxis=dict(range=[0, min(max(degrees) + 1, 50)]),  # Limit x-axis to avoid long tail
    )
    
    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def plot_centrality_distribution(centrality_values, metric_name):
    """
    Plot the distribution of centrality values.
    
    Args:
        centrality_values (dict): Dictionary of node:centrality pairs
        metric_name (str): Name of the centrality metric
    """
    # Convert to DataFrame for plotting
    df = pd.DataFrame({"Node": list(centrality_values.keys()), 
                       "Centrality": list(centrality_values.values())})
    df = df.sort_values("Centrality", ascending=False)
    
    # Histogram
    fig = px.histogram(df, x="Centrality", 
                      labels={"Centrality": metric_name},
                      title=f"{metric_name} Distribution")
    
    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    
    # Top nodes table
    st.subheader(f"Top 10 Nodes by {metric_name}")
    st.table(df.head(10))

def plot_community_structure(G, partition):
    """
    Visualize the community structure of the network.
    
    Args:
        G (networkx.Graph): The network to visualize
        partition (dict): Node-to-community mapping from community detection
    """
    # For large networks, sample a subset
    if G.number_of_nodes() > 500:
        # Sample nodes, ensuring we get nodes from different communities
        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node)
        
        # Sample from each community
        sampled_nodes = []
        for comm_nodes in communities.values():
            # Take a proportional sample from each community
            sample_size = min(len(comm_nodes), max(5, int(500 * len(comm_nodes) / G.number_of_nodes())))
            if sample_size < len(comm_nodes):
                sampled_from_comm = np.random.choice(comm_nodes, size=sample_size, replace=False)
            else:
                sampled_from_comm = comm_nodes
            sampled_nodes.extend(sampled_from_comm)
        
        # Create subgraph with sampled nodes
        subgraph = G.subgraph(sampled_nodes).copy()
        
        # Get the partition for the subgraph
        subgraph_partition = {n: partition[n] for n in subgraph.nodes() if n in partition}
    else:
        subgraph = G
        subgraph_partition = partition
    
    # Compute layout
    pos = nx.spring_layout(subgraph, seed=42)
    
    # Set up node colors based on community
    community_to_color = {}
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    node_colors = []
    for node in subgraph.nodes():
        if node in subgraph_partition:
            comm_id = subgraph_partition[node]
            if comm_id not in community_to_color:
                community_to_color[comm_id] = colors[len(community_to_color) % len(colors)]
            node_colors.append(community_to_color[comm_id])
        else:
            node_colors.append('#cccccc')  # Default gray for nodes without community
    
    # Create edge traces
    edge_x = []
    edge_y = []
    
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    
    for node in subgraph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        if node in subgraph_partition:
            comm_id = subgraph_partition[node]
            node_text.append(f"Node: {node}<br>Community: {comm_id}<br>Degree: {subgraph.degree(node)}")
        else:
            node_text.append(f"Node: {node}<br>No Community<br>Degree: {subgraph.degree(node)}")
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color=node_colors,
            size=10,
            line=dict(width=1, color='#333')
        ),
        text=node_text
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title="Community Structure Visualization",
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                   ))
    
    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    
    # Community size distribution
    community_sizes = {}
    for node, comm_id in partition.items():
        if comm_id not in community_sizes:
            community_sizes[comm_id] = 0
        community_sizes[comm_id] += 1
    
    # Convert to DataFrame for plotting
    df = pd.DataFrame({"Community": list(community_sizes.keys()), 
                       "Size": list(community_sizes.values())})
    df = df.sort_values("Size", ascending=False)
    
    # Create bar chart
    fig = px.bar(df.head(20), x="Community", y="Size",
                 labels={"Community": "Community ID", "Size": "Number of Nodes"},
                 title="Top 20 Communities by Size")
    
    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)
