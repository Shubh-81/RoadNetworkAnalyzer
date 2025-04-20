import networkx as nx
import numpy as np
import heapq
import streamlit as st
from collections import defaultdict

def find_shortest_path(G, source, target):
    """
    Find the shortest path between two nodes in the network.
    
    Args:
        G (networkx.Graph): The network
        source (int): Source node ID
        target (int): Target node ID
    
    Returns:
        list: List of nodes in the shortest path or None if no path exists
    """
    try:
        path = nx.shortest_path(G, source=source, target=target)
        return path
    except nx.NetworkXNoPath:
        return None
    except nx.NodeNotFound:
        return None

def find_all_shortest_paths(G, source, target, max_paths=10):
    """
    Find all shortest paths between two nodes in the network.
    
    Args:
        G (networkx.Graph): The network
        source (int): Source node ID
        target (int): Target node ID
        max_paths (int): Maximum number of paths to return
    
    Returns:
        list: List of paths (each path is a list of nodes)
    """
    try:
        paths = list(nx.all_shortest_paths(G, source=source, target=target))
        return paths[:max_paths]  # Limit the number of paths
    except nx.NetworkXNoPath:
        return []
    except nx.NodeNotFound:
        return []

def route_optimization(G, source, target, method="shortest", num_paths=3):
    """
    Find optimized routes between source and target.
    
    Args:
        G (networkx.Graph): The network
        source (int): Source node ID
        target (int): Target node ID
        method (str): Routing method ("shortest", "alternates", "traffic_aware")
        num_paths (int): Number of alternative paths to find
    
    Returns:
        dict: Dictionary containing the optimized routes
    """
    if method == "shortest":
        path = find_shortest_path(G, source, target)
        return {
            "primary_path": path,
            "alternates": [],
            "method": "shortest",
            "metrics": {
                "path_length": len(path) - 1 if path else 0
            }
        }
    
    elif method == "alternates":
        # Find multiple shortest paths if possible
        paths = find_all_shortest_paths(G, source, target, max_paths=num_paths)
        
        if not paths:
            return {
                "primary_path": None,
                "alternates": [],
                "method": "alternates",
                "metrics": {}
            }
        
        # If we need more paths and only found one shortest path, 
        # find some approximate alternates
        if len(paths) < num_paths and len(paths) > 0:
            primary = paths[0]
            
            # Create a copy of the graph and modify edge weights
            # to discourage using the same edges as the primary path
            H = G.copy()
            
            for i in range(len(primary) - 1):
                u, v = primary[i], primary[i+1]
                if H.has_edge(u, v):
                    # Make this edge less attractive for the next search
                    H[u][v]['weight'] = 10.0
            
            # Find additional paths with the modified graph
            for _ in range(num_paths - len(paths)):
                try:
                    alt_path = nx.shortest_path(H, source=source, target=target)
                    if alt_path not in paths:
                        paths.append(alt_path)
                    
                    # Further modify the weights to get more diverse paths
                    for i in range(len(alt_path) - 1):
                        u, v = alt_path[i], alt_path[i+1]
                        if H.has_edge(u, v):
                            H[u][v]['weight'] = H[u][v].get('weight', 1.0) + 5.0
                except:
                    break
        
        return {
            "primary_path": paths[0] if paths else None,
            "alternates": paths[1:] if len(paths) > 1 else [],
            "method": "alternates",
            "metrics": {
                "num_paths": len(paths),
                "path_lengths": [len(p) - 1 for p in paths]
            }
        }
    
    elif method == "traffic_aware":
        # For demonstration, we'll simulate traffic by assigning
        # random weights to edges and finding the shortest weighted path
        
        # Create a copy of the graph with random weights
        H = G.copy()
        
        # Assign random weights to simulate traffic conditions
        for u, v in H.edges():
            # Higher weight = more congestion
            # Nodes with high betweenness are more likely to be congested
            H[u][v]['weight'] = np.random.uniform(1.0, 3.0)
        
        try:
            # Find the path with minimum total weight
            path = nx.shortest_path(H, source=source, target=target, weight='weight')
            
            # Calculate path cost
            path_cost = sum(H[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
            
            # For comparison, find the topologically shortest path
            topo_path = nx.shortest_path(G, source=source, target=target)
            
            # Calculate its cost in the traffic-aware model
            topo_cost = sum(H[topo_path[i]][topo_path[i+1]].get('weight', 1.0) 
                           for i in range(len(topo_path)-1))
            
            return {
                "primary_path": path,
                "alternates": [topo_path] if topo_path != path else [],
                "method": "traffic_aware",
                "metrics": {
                    "path_length": len(path) - 1,
                    "path_cost": path_cost,
                    "topo_path_length": len(topo_path) - 1,
                    "topo_path_cost": topo_cost,
                    "cost_reduction": topo_cost - path_cost if path != topo_path else 0
                }
            }
        except:
            return {
                "primary_path": None,
                "alternates": [],
                "method": "traffic_aware",
                "metrics": {}
            }
    
    else:
        raise ValueError(f"Unsupported routing method: {method}")

def identify_traffic_bottlenecks(G, num_bottlenecks=20, method="betweenness"):
    """
    Identify potential traffic bottlenecks in the network.
    
    Args:
        G (networkx.Graph): The network
        num_bottlenecks (int): Number of top bottlenecks to identify
        method (str): Method for identifying bottlenecks
    
    Returns:
        list: Top bottleneck nodes or edges
    """
    if method == "betweenness":
        # For large networks, use approximate betweenness with sampling
        if G.number_of_nodes() > 1000:
            k = min(100, G.number_of_nodes())  # Number of sample nodes
            centrality = nx.betweenness_centrality(G, k=k, normalized=True)
        else:
            centrality = nx.betweenness_centrality(G, normalized=True)
        
        # Sort nodes by betweenness centrality
        bottlenecks = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return bottlenecks[:num_bottlenecks]
    
    elif method == "edge_betweenness":
        # For large networks, use approximate edge betweenness with sampling
        if G.number_of_edges() > 5000:
            k = min(100, G.number_of_nodes())  # Number of sample nodes
            centrality = nx.edge_betweenness_centrality(G, k=k, normalized=True)
        else:
            centrality = nx.edge_betweenness_centrality(G, normalized=True)
        
        # Sort edges by betweenness centrality
        bottlenecks = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return bottlenecks[:num_bottlenecks]
    
    elif method == "degree":
        # Simple approach: nodes with highest degree
        degrees = dict(G.degree())
        bottlenecks = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        return bottlenecks[:num_bottlenecks]
    
    else:
        raise ValueError(f"Unsupported bottleneck identification method: {method}")

def simulate_traffic_flow(G, traffic_sources, traffic_sinks, flow_amount=100):
    """
    Simulate traffic flow through the network.
    
    Args:
        G (networkx.Graph): The network
        traffic_sources (list): List of source nodes generating traffic
        traffic_sinks (list): List of sink nodes absorbing traffic
        flow_amount (int): Total amount of traffic to simulate
    
    Returns:
        dict: Traffic flow results
    """
    # Create a copy of the graph for simulation
    H = G.copy()
    
    # Initialize edge flow counters
    edge_flows = {edge: 0 for edge in H.edges()}
    node_flows = {node: 0 for node in H.nodes()}
    
    # For each source-sink pair, simulate traffic
    for source in traffic_sources:
        for sink in traffic_sinks:
            # Skip if same node or nodes not in graph
            if source == sink or source not in H or sink not in H:
                continue
            
            # Amount of flow for this source-sink pair
            flow = flow_amount / (len(traffic_sources) * len(traffic_sinks))
            
            try:
                # Find shortest path
                path = nx.shortest_path(H, source=source, target=sink)
                
                # Add flow to each edge and node in the path
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    edge = (u, v) if u < v else (v, u)  # Canonical edge representation
                    edge_flows[edge] += flow
                    node_flows[u] += flow
                
                # Add flow to the last node
                node_flows[path[-1]] += flow
                
            except nx.NetworkXNoPath:
                continue
    
    # Find congested edges and nodes
    edge_congestion = sorted(edge_flows.items(), key=lambda x: x[1], reverse=True)
    node_congestion = sorted(node_flows.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "edge_flows": edge_flows,
        "node_flows": node_flows,
        "congested_edges": edge_congestion[:20],  # Top 20 congested edges
        "congested_nodes": node_congestion[:20],  # Top 20 congested nodes
        "total_flow": flow_amount
    }
