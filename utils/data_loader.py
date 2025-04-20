import networkx as nx
import pandas as pd
import numpy as np
import requests
import io
import os
import streamlit as st
import tempfile
import json

def load_california_network():
    """
    Load the California road network from the Stanford Large Network Dataset Collection.
    
    Returns:
        networkx.Graph: The California road network as an undirected graph
    """
        # Create a NetworkX graph
    G = nx.Graph()
    
    with open('roadNet-CA.txt', 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            source, target = map(int, line.strip().split())
            G.add_edge(source, target)
    
    return G

def create_sample_network():
    """
    Create a sample road network for demonstration when the actual data cannot be loaded.
    
    Returns:
        networkx.Graph: A sample road network
    """
    # Create a grid-like network that mimics a simple road layout
    G = nx.grid_2d_graph(20, 20)  # 20x20 grid
    
    # Convert to regular graph with integer nodes
    G = nx.convert_node_labels_to_integers(G)
    
    # Add some random edges to create shortcuts and more complex patterns
    nodes = list(G.nodes())
    num_extra_edges = len(nodes) // 10
    
    for _ in range(num_extra_edges):
        u = np.random.choice(nodes)
        v = np.random.choice(nodes)
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)
    
    return G

def save_current_network(G, format="CSV"):
    """
    Save the current network to a file in the specified format.
    
    Args:
        G (networkx.Graph): The network to save
        format (str): The format to save in (CSV, JSON, or GraphML)
    
    Returns:
        tuple: (file_content, filename) - content as bytes and suggested filename
    """
    if format == "CSV":
        # Export edges to CSV
        edges_df = nx.to_pandas_edgelist(G)
        csv_buffer = io.StringIO()
        edges_df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue().encode(), "california_road_network_edges.csv"
    
    elif format == "JSON":
        # Export as JSON
        data = nx.node_link_data(G)
        json_str = json.dumps(data, indent=2)
        return json_str.encode(), "california_road_network.json"
    
    elif format == "GraphML":
        # Export as GraphML
        graphml_buffer = io.BytesIO()
        nx.write_graphml(G, graphml_buffer)
        return graphml_buffer.getvalue(), "california_road_network.graphml"
    
    else:
        raise ValueError(f"Unsupported format: {format}")
