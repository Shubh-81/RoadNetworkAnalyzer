import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from collections import Counter

def calculate_network_metrics(G):
    """
    Calculate basic metrics for the given network.
    
    Args:
        G (networkx.Graph): The network to analyze
    
    Returns:
        dict: A dictionary of network metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics["num_nodes"] = G.number_of_nodes()
    metrics["num_edges"] = G.number_of_edges()
    metrics["density"] = nx.density(G)
    
    # Degree metrics
    degrees = [d for _, d in G.degree()]
    metrics["avg_degree"] = np.mean(degrees) if degrees else 0
    metrics["max_degree"] = max(degrees) if degrees else 0
    metrics["degree_distribution"] = Counter(degrees)
    
    # Clustering
    # Use approximate clustering for large networks
    if G.number_of_nodes() > 1000:
        # Sample a subset of nodes for clustering calculation
        sample_size = min(1000, G.number_of_nodes())
        sampled_nodes = np.random.choice(list(G.nodes()), size=sample_size, replace=False)
        clustering_values = nx.clustering(G, nodes=sampled_nodes)
        metrics["avg_clustering"] = np.mean(list(clustering_values.values())) if clustering_values else 0
    else:
        metrics["avg_clustering"] = nx.average_clustering(G)
    
    # Connected components
    metrics["connected_components"] = nx.number_connected_components(G)
    
    # For large networks, skip expensive computations
    if G.number_of_nodes() <= 1000:
        # Path length and diameter (only for smaller networks or samples)
        if nx.is_connected(G):
            try:
                metrics["avg_path_length"] = nx.average_shortest_path_length(G)
                metrics["diameter"] = nx.diameter(G)
            except Exception as e:
                st.warning(f"Error calculating path metrics: {str(e)}")
                metrics["avg_path_length"] = None
                metrics["diameter"] = None
        else:
            # For disconnected networks, calculate metrics on the largest component
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc).copy()
            try:
                metrics["avg_path_length"] = nx.average_shortest_path_length(subgraph)
                metrics["diameter"] = nx.diameter(subgraph)
                metrics["largest_component_size"] = len(largest_cc)
                metrics["largest_component_ratio"] = len(largest_cc) / G.number_of_nodes()
            except Exception as e:
                st.warning(f"Error calculating path metrics on largest component: {str(e)}")
                metrics["avg_path_length"] = None
                metrics["diameter"] = None
    else:
        metrics["avg_path_length"] = None
        metrics["diameter"] = None
        st.info("Path length and diameter calculations skipped for large network")
    
    return metrics

def identify_congestion_hotspots(G, method="betweenness", top_n=20):
    """
    Identify potential congestion hotspots in the network.
    
    Args:
        G (networkx.Graph): The network to analyze
        method (str): Method to use for hotspot identification 
                     ("betweenness", "degree", "closeness")
        top_n (int): Number of top hotspots to return
    
    Returns:
        pandas.DataFrame: Top hotspots with their scores
    """
    if method == "betweenness":
        # For large networks, use approximate betweenness with sampling
        if G.number_of_nodes() > 1000:
            st.info("Using approximate betweenness with sampling for large network")
            k = min(100, G.number_of_nodes())  # Number of sample nodes
            centrality = nx.betweenness_centrality(G, k=k, normalized=True)
        else:
            centrality = nx.betweenness_centrality(G, normalized=True)
        metric_name = "Betweenness Centrality"
        
    elif method == "degree":
        centrality = dict(G.degree())
        metric_name = "Degree"
        
    elif method == "closeness":
        # For large networks, use approximate closeness with sampling
        if G.number_of_nodes() > 1000:
            # Sample nodes for closeness calculation
            sample_nodes = np.random.choice(list(G.nodes()), size=min(1000, G.number_of_nodes()), replace=False)
            centrality = {}
            for node in sample_nodes:
                centrality[node] = nx.closeness_centrality(G, u=node)
        else:
            centrality = nx.closeness_centrality(G)
        metric_name = "Closeness Centrality"
    
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(list(centrality.items()), columns=["Node", metric_name])
    df = df.sort_values(by=metric_name, ascending=False).head(top_n)
    df = df.reset_index(drop=True)
    
    return df

def analyze_community_structure(G, method="louvain", max_nodes=1000):
    """
    Analyze community structure of the network.
    
    Args:
        G (networkx.Graph): The network to analyze
        method (str): Community detection method
        max_nodes (int): Maximum network size for analysis
    
    Returns:
        dict: Community detection results
    """
    # For large networks, work with a sample
    if G.number_of_nodes() > max_nodes:
        st.warning(f"Network is large. Sampling {max_nodes} nodes for community detection.")
        
        # Get largest connected component
        components = list(nx.connected_components(G))
        if components:
            largest_cc = max(components, key=len)
            if len(largest_cc) > max_nodes:
                # Sample from largest component
                subgraph_nodes = list(largest_cc)[:max_nodes]
            else:
                subgraph_nodes = largest_cc
        else:
            # No connected components (unlikely), just sample random nodes
            subgraph_nodes = list(G.nodes())[:max_nodes]
        
        subgraph = G.subgraph(subgraph_nodes).copy()
    else:
        subgraph = G.copy()
    
    # Detect communities
    try:
        if method == "louvain":
            import community as community_louvain
            partition = community_louvain.best_partition(subgraph)
        elif method == "label_propagation":
            partition = {node: i for i, community in enumerate(nx.algorithms.community.label_propagation_communities(subgraph)) 
                         for node in community}
        else:
            raise ValueError(f"Unsupported community detection method: {method}")
        
        # Analyze results
        communities = {}
        for node, community_id in partition.items():
            if community_id not in communities:
                communities[community_id] = []
            communities[community_id].append(node)
        
        # Get statistics
        community_sizes = [len(nodes) for nodes in communities.values()]
        
        return {
            "num_communities": len(communities),
            "avg_community_size": np.mean(community_sizes),
            "max_community_size": max(community_sizes),
            "min_community_size": min(community_sizes),
            "communities": communities,
            "partition": partition
        }
        
    except Exception as e:
        st.error(f"Error in community detection: {str(e)}")
        return {
            "error": str(e),
            "num_communities": 0,
            "communities": {},
            "partition": {}
        }
