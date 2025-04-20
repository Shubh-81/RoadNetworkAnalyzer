import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import random
from collections import Counter

from utils.routing import route_optimization, identify_traffic_bottlenecks
from utils.visualization import plot_network

# Configure page
st.set_page_config(
    page_title="Routing Optimization | California Road Network Analysis",
    page_icon="🚗",
    layout="wide"
)

# Title
st.title("Routing Optimization")
st.write("Optimize routes and analyze traffic patterns in the California road network")

# Check if network is loaded
if 'network' not in st.session_state or st.session_state.network is None:
    st.warning("Please load the California road network from the home page first")
    st.stop()

# Get the network to analyze
network_to_analyze = st.session_state.selected_region if st.session_state.selected_region is not None else st.session_state.network

# Add tabs for different analyses
tab1, tab2, tab3 = st.tabs(["Route Finding", "Alternative Routes", "Routing Strategies"])

with tab1:
    st.subheader("Find Optimal Routes")
    st.write("""
    Find the shortest path between two intersections in the road network.
    """)
    
    # Select source and target nodes
    col1, col2 = st.columns(2)
    with col1:
        source_node = st.number_input("Source Intersection ID", min_value=0, 
                                    max_value=max(network_to_analyze.nodes()) if network_to_analyze.nodes() else 0, 
                                    step=1)
    with col2:
        target_node = st.number_input("Target Intersection ID", min_value=0, 
                                     max_value=max(network_to_analyze.nodes()) if network_to_analyze.nodes() else 0, 
                                     step=1)
    
    # Find route
    if st.button("Find Route"):
        if source_node in network_to_analyze and target_node in network_to_analyze:
            with st.spinner("Finding optimal route..."):
                # Use the route optimization function with shortest path method
                route_results = route_optimization(
                    network_to_analyze,
                    source_node,
                    target_node,
                    method="shortest"
                )
                
                path = route_results["primary_path"]
                
                if path:
                    st.success(f"Found path with {len(path)-1} steps")
                    
                    # Display path and metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Path Length", len(path)-1)
                    with col2:
                        avg_degree = np.mean([network_to_analyze.degree(node) for node in path])
                        st.metric("Average Node Degree on Path", f"{avg_degree:.2f}")
                    
                    # Show the path
                    st.write("Path:", path)
                    
                    # Visualize the path
                    plot_network(network_to_analyze, highlight_path=path)
                    
                    # Path analysis
                    if len(path) > 2:
                        st.subheader("Path Analysis")
                        
                        # Calculate degree distribution along the path
                        path_degrees = [network_to_analyze.degree(node) for node in path]
                        
                        # Create dataframe for path nodes
                        path_data = []
                        for i, node in enumerate(path):
                            path_data.append({
                                "Position": i,
                                "Node": node,
                                "Degree": network_to_analyze.degree(node)
                            })
                        
                        path_df = pd.DataFrame(path_data)
                        
                        # Plot degree distribution along the path
                        fig = px.line(path_df, x="Position", y="Degree", 
                                     title="Degree Distribution Along Path")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Identify potential bottlenecks on the path
                        st.subheader("Potential Bottlenecks on Route")
                        
                        # Calculate betweenness for path nodes
                        path_set = set(path)
                        subgraph = network_to_analyze.subgraph(path_set).copy()
                        
                        try:
                            # Use betweenness on the path subgraph
                            betweenness = nx.betweenness_centrality(subgraph)
                            
                            # Combine with degree
                            bottleneck_score = {}
                            for node in path:
                                # Get position in path (excluding source and target)
                                if node != source_node and node != target_node:
                                    node_degree = network_to_analyze.degree(node)
                                    node_betweenness = betweenness.get(node, 0)
                                    
                                    # Bottleneck score: high betweenness but lower degree is a potential bottleneck
                                    bottleneck_score[node] = node_betweenness / max(1, node_degree/4)
                            
                            # Get top potential bottlenecks
                            bottlenecks = sorted(bottleneck_score.items(), key=lambda x: x[1], reverse=True)
                            
                            if bottlenecks:
                                bottleneck_data = []
                                for node, score in bottlenecks[:5]:  # Top 5
                                    bottleneck_data.append({
                                        "Node": node,
                                        "Bottleneck Score": f"{score:.4f}",
                                        "Degree": network_to_analyze.degree(node),
                                        "Betweenness": betweenness.get(node, 0)
                                    })
                                
                                bottleneck_df = pd.DataFrame(bottleneck_data)
                                st.dataframe(bottleneck_df)
                                
                                # Highlight bottlenecks in the path visualization
                                if st.checkbox("Highlight bottlenecks in path"):
                                    bottleneck_nodes = bottleneck_df["Node"].tolist()
                                    plot_network(network_to_analyze, highlight_path=path, highlight_nodes=bottleneck_nodes)
                            else:
                                st.info("No significant bottlenecks identified on this route")
                        except Exception as e:
                            st.warning(f"Could not compute detailed bottleneck analysis: {str(e)}")
                else:
                    st.error("No path found between the selected intersections")
        else:
            st.error("Source or target intersection not found in the network")

with tab2:
    st.subheader("Find Alternative Routes")
    st.write("""
    Find multiple alternative routes between two intersections to distribute traffic or provide options.
    """)
    
    # Select source and target nodes
    col1, col2 = st.columns(2)
    with col1:
        alt_source = st.number_input("Source Intersection ID", min_value=0, 
                                   max_value=max(network_to_analyze.nodes()) if network_to_analyze.nodes() else 0, 
                                   step=1, key="alt_source")
    with col2:
        alt_target = st.number_input("Target Intersection ID", min_value=0, 
                                    max_value=max(network_to_analyze.nodes()) if network_to_analyze.nodes() else 0, 
                                    step=1, key="alt_target")
    
    # Number of alternative routes
    num_routes = st.slider("Number of alternative routes to find", 2, 10, 3)
    
    # Find alternative routes
    if st.button("Find Alternative Routes"):
        if alt_source in network_to_analyze and alt_target in network_to_analyze:
            with st.spinner("Finding alternative routes..."):
                # Use the route optimization function with alternates method
                route_results = route_optimization(
                    network_to_analyze,
                    alt_source,
                    alt_target,
                    method="alternates",
                    num_paths=num_routes
                )
                
                primary_path = route_results["primary_path"]
                alternates = route_results["alternates"]
                
                if primary_path:
                    st.success(f"Found {1 + len(alternates)} routes")
                    
                    # Display primary path
                    st.subheader("Primary Route")
                    st.write(f"Path length: {len(primary_path)-1} steps")
                    st.write("Path:", primary_path)
                    
                    # Visualize primary path
                    plot_network(network_to_analyze, highlight_path=primary_path)
                    
                    # Display alternative paths
                    if alternates:
                        st.subheader("Alternative Routes")
                        
                        for i, path in enumerate(alternates):
                            st.write(f"Alternative {i+1} (Length: {len(path)-1}):", path)
                        
                        # Compare routes
                        st.subheader("Route Comparison")
                        
                        # Create comparison table
                        route_data = []
                        
                        # Add primary route
                        primary_degree = np.mean([network_to_analyze.degree(node) for node in primary_path])
                        route_data.append({
                            "Route": "Primary",
                            "Length": len(primary_path) - 1,
                            "Avg Node Degree": f"{primary_degree:.2f}"
                        })
                        
                        # Add alternative routes
                        for i, path in enumerate(alternates):
                            path_degree = np.mean([network_to_analyze.degree(node) for node in path])
                            route_data.append({
                                "Route": f"Alternative {i+1}",
                                "Length": len(path) - 1,
                                "Avg Node Degree": f"{path_degree:.2f}"
                            })
                        
                        # Display comparison table
                        st.table(pd.DataFrame(route_data))
                        
                        # Route overlap analysis
                        st.subheader("Route Overlap Analysis")
                        
                        # Calculate overlap between routes
                        all_paths = [primary_path] + alternates
                        overlap_matrix = np.zeros((len(all_paths), len(all_paths)))
                        
                        for i in range(len(all_paths)):
                            for j in range(i, len(all_paths)):
                                if i == j:
                                    overlap_matrix[i][j] = 1.0  # Self-overlap is 100%
                                else:
                                    # Calculate Jaccard similarity
                                    path_i_set = set(all_paths[i])
                                    path_j_set = set(all_paths[j])
                                    
                                    intersection = len(path_i_set.intersection(path_j_set))
                                    union = len(path_i_set.union(path_j_set))
                                    
                                    similarity = intersection / union if union > 0 else 0
                                    
                                    overlap_matrix[i][j] = similarity
                                    overlap_matrix[j][i] = similarity
                        
                        # Create labels for heatmap
                        route_labels = ["Primary"] + [f"Alt {i+1}" for i in range(len(alternates))]
                        
                        # Create heatmap
                        fig = go.Figure(data=go.Heatmap(
                            z=overlap_matrix,
                            x=route_labels,
                            y=route_labels,
                            colorscale='Viridis',
                            zmin=0, zmax=1
                        ))
                        
                        fig.update_layout(
                            title="Route Overlap (Jaccard Similarity)",
                            xaxis_title="Route",
                            yaxis_title="Route"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Option to visualize all routes
                        selected_routes = st.multiselect(
                            "Select routes to visualize",
                            ["Primary"] + [f"Alternative {i+1}" for i in range(len(alternates))],
                            ["Primary"]
                        )
                        
                        if selected_routes:
                            # Collect paths to visualize
                            paths_to_viz = []
                            
                            if "Primary" in selected_routes:
                                paths_to_viz.append(primary_path)
                            
                            for i in range(len(alternates)):
                                if f"Alternative {i+1}" in selected_routes:
                                    paths_to_viz.append(alternates[i])
                            
                            # Visualize selected routes
                            if len(paths_to_viz) == 1:
                                plot_network(network_to_analyze, highlight_path=paths_to_viz[0])
                            else:
                                # For multiple routes, we need a custom visualization
                                # We'll highlight the nodes that appear in any selected path
                                all_path_nodes = set()
                                for path in paths_to_viz:
                                    all_path_nodes.update(path)
                                
                                plot_network(network_to_analyze, highlight_nodes=list(all_path_nodes))
                                st.info("For multiple routes, all nodes in any of the selected routes are highlighted")
                    else:
                        st.info("No alternative routes found")
                else:
                    st.error("No path found between the selected intersections")
        else:
            st.error("Source or target intersection not found in the network")

with tab3:
    st.subheader("Advanced Routing Strategies")
    st.write("""
    Compare different routing strategies including traffic-aware routing.
    Traffic-aware routing takes into account simulated congestion levels.
    """)
    
    # Select source and target nodes
    col1, col2 = st.columns(2)
    with col1:
        strat_source = st.number_input("Source Intersection ID", min_value=0, 
                                     max_value=max(network_to_analyze.nodes()) if network_to_analyze.nodes() else 0, 
                                     step=1, key="strat_source")
    with col2:
        strat_target = st.number_input("Target Intersection ID", min_value=0, 
                                      max_value=max(network_to_analyze.nodes()) if network_to_analyze.nodes() else 0, 
                                      step=1, key="strat_target")
    
    # Select routing strategy
    routing_strategy = st.selectbox(
        "Routing Strategy",
        ["Shortest Path", "Traffic-Aware Routing"]
    )
    
    # Find route using selected strategy
    if st.button("Find Route", key="strat_button"):
        if strat_source in network_to_analyze and strat_target in network_to_analyze:
            with st.spinner(f"Finding route using {routing_strategy}..."):
                method = "shortest" if routing_strategy == "Shortest Path" else "traffic_aware"
                
                # Use route optimization with the selected method
                route_results = route_optimization(
                    network_to_analyze,
                    strat_source,
                    strat_target,
                    method=method
                )
                
                primary_path = route_results["primary_path"]
                
                if primary_path:
                    st.success(f"Found route with {len(primary_path)-1} steps")
                    
                    # Display path and metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Path Length", len(primary_path)-1)
                    with col2:
                        if "path_cost" in route_results["metrics"]:
                            st.metric("Path Cost (considering traffic)", f"{route_results['metrics']['path_cost']:.2f}")
                        else:
                            st.metric("Average Node Degree", 
                                     f"{np.mean([network_to_analyze.degree(node) for node in primary_path]):.2f}")
                    
                    # Show the path
                    st.write("Path:", primary_path)
                    
                    # Visualize the path
                    plot_network(network_to_analyze, highlight_path=primary_path)
                    
                    # For traffic-aware routing, compare with standard shortest path
                    if routing_strategy == "Traffic-Aware Routing" and route_results["alternates"]:
                        st.subheader("Comparison with Shortest Path")
                        
                        topo_path = route_results["alternates"][0]  # The topological shortest path
                        
                        # Create comparison table
                        comparison_data = [
                            {
                                "Route": "Traffic-Aware",
                                "Length": len(primary_path) - 1,
                                "Traffic Cost": f"{route_results['metrics']['path_cost']:.2f}"
                            },
                            {
                                "Route": "Shortest Path",
                                "Length": len(topo_path) - 1,
                                "Traffic Cost": f"{route_results['metrics']['topo_path_cost']:.2f}"
                            }
                        ]
                        
                        # Display comparison
                        st.table(pd.DataFrame(comparison_data))
                        
                        # Calculate improvement
                        if "cost_reduction" in route_results["metrics"]:
                            improvement = route_results["metrics"]["cost_reduction"]
                            if improvement > 0:
                                st.success(f"Traffic-aware routing reduces estimated travel cost by {improvement:.2f} " +
                                         f"({improvement/route_results['metrics']['topo_path_cost']*100:.1f}%)")
                            else:
                                st.info("Traffic-aware routing did not find a better route than the shortest path")
                        
                        # Show both paths for comparison
                        if st.checkbox("Visualize both routes"):
                            # Traffic-aware route
                            st.subheader("Traffic-Aware Route")
                            plot_network(network_to_analyze, highlight_path=primary_path)
                            
                            # Shortest path
                            st.subheader("Standard Shortest Path")
                            plot_network(network_to_analyze, highlight_path=topo_path)
                else:
                    st.error("No path found between the selected intersections")
        else:
            st.error("Source or target intersection not found in the network")
    
    # Traffic impact analysis
    st.subheader("Traffic Impact Analysis")
    st.write("""
    Analyze how traffic routing decisions affect the overall network congestion.
    This simulation distributes traffic through the network using different routing strategies.
    """)
    
    # Simulation parameters
    col1, col2 = st.columns(2)
    with col1:
        num_pairs = st.slider("Number of source-target pairs", 5, 50, 20)
    with col2:
        percent_traffic_aware = st.slider("Percentage of traffic using traffic-aware routing", 0, 100, 50)
    
    # Run simulation
    if st.button("Run Traffic Impact Simulation"):
        with st.spinner("Simulating traffic impact..."):
            # Create random source-target pairs
            all_nodes = list(network_to_analyze.nodes())
            if len(all_nodes) >= num_pairs * 2:
                # Select random nodes
                random_nodes = random.sample(all_nodes, num_pairs * 2)
                
                # Create pairs
                source_target_pairs = []
                for i in range(0, num_pairs * 2, 2):
                    source_target_pairs.append((random_nodes[i], random_nodes[i+1]))
                
                # Determine how many pairs use traffic-aware routing
                num_traffic_aware = int(num_pairs * percent_traffic_aware / 100)
                
                # Track edge usage for each strategy
                edge_usage_shortest = Counter()
                edge_usage_traffic = Counter()
                
                # Track routes
                shortest_paths = []
                traffic_aware_paths = []
                
                # Route each pair
                for i, (source, target) in enumerate(source_target_pairs):
                    try:
                        if i < num_traffic_aware:
                            # Use traffic-aware routing
                            route_result = route_optimization(
                                network_to_analyze,
                                source,
                                target,
                                method="traffic_aware"
                            )
                            
                            if route_result["primary_path"]:
                                traffic_aware_paths.append(route_result["primary_path"])
                                
                                # Track edge usage
                                path = route_result["primary_path"]
                                for j in range(len(path) - 1):
                                    edge = (path[j], path[j+1])
                                    if path[j] > path[j+1]:
                                        edge = (path[j+1], path[j])
                                    edge_usage_traffic[edge] += 1
                        else:
                            # Use shortest path routing
                            route_result = route_optimization(
                                network_to_analyze,
                                source,
                                target,
                                method="shortest"
                            )
                            
                            if route_result["primary_path"]:
                                shortest_paths.append(route_result["primary_path"])
                                
                                # Track edge usage
                                path = route_result["primary_path"]
                                for j in range(len(path) - 1):
                                    edge = (path[j], path[j+1])
                                    if path[j] > path[j+1]:
                                        edge = (path[j+1], path[j])
                                    edge_usage_shortest[edge] += 1
                    except:
                        continue
                
                # Combine edge usage
                edge_usage_combined = Counter()
                for edge, count in edge_usage_shortest.items():
                    edge_usage_combined[edge] += count
                for edge, count in edge_usage_traffic.items():
                    edge_usage_combined[edge] += count
                
                # Calculate congestion metrics
                if edge_usage_shortest and edge_usage_traffic:
                    # Top congested edges with shortest path routing
                    top_shortest = sorted(edge_usage_shortest.items(), key=lambda x: x[1], reverse=True)
                    
                    # Top congested edges with traffic-aware routing
                    top_traffic = sorted(edge_usage_traffic.items(), key=lambda x: x[1], reverse=True)
                    
                    # Top congested edges overall
                    top_combined = sorted(edge_usage_combined.items(), key=lambda x: x[1], reverse=True)
                    
                    # Calculate congestion metrics
                    shortest_max = max(edge_usage_shortest.values()) if edge_usage_shortest else 0
                    traffic_max = max(edge_usage_traffic.values()) if edge_usage_traffic else 0
                    combined_max = max(edge_usage_combined.values()) if edge_usage_combined else 0
                    
                    shortest_avg = np.mean(list(edge_usage_shortest.values())) if edge_usage_shortest else 0
                    traffic_avg = np.mean(list(edge_usage_traffic.values())) if edge_usage_traffic else 0
                    combined_avg = np.mean(list(edge_usage_combined.values())) if edge_usage_combined else 0
                    
                    # Display congestion metrics
                    st.subheader("Congestion Metrics")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Max Congestion (Shortest)", shortest_max)
                        st.metric("Avg Congestion (Shortest)", f"{shortest_avg:.2f}")
                    with col2:
                        st.metric("Max Congestion (Traffic-Aware)", traffic_max)
                        st.metric("Avg Congestion (Traffic-Aware)", f"{traffic_avg:.2f}")
                    with col3:
                        st.metric("Max Overall Congestion", combined_max)
                        st.metric("Avg Overall Congestion", f"{combined_avg:.2f}")
                    
                    # Calculate improvement
                    if shortest_max > 0 and traffic_max > 0:
                        max_improvement = (shortest_max - traffic_max) / shortest_max * 100
                        avg_improvement = (shortest_avg - traffic_avg) / shortest_avg * 100
                        
                        if max_improvement > 0:
                            st.success(f"Traffic-aware routing reduced maximum congestion by {max_improvement:.1f}%")
                        elif max_improvement < 0:
                            st.warning(f"Traffic-aware routing increased maximum congestion by {abs(max_improvement):.1f}%")
                        else:
                            st.info("Traffic-aware routing had no effect on maximum congestion")
                        
                        if avg_improvement > 0:
                            st.success(f"Traffic-aware routing reduced average congestion by {avg_improvement:.1f}%")
                        elif avg_improvement < 0:
                            st.warning(f"Traffic-aware routing increased average congestion by {abs(avg_improvement):.1f}%")
                        else:
                            st.info("Traffic-aware routing had no effect on average congestion")
                    
                    # Display top congested edges
                    st.subheader("Top Congested Edges")
                    
                    tab1, tab2, tab3 = st.tabs(["Overall", "Shortest Path", "Traffic-Aware"])
                    
                    with tab1:
                        edge_data = []
                        for edge, count in top_combined[:20]:
                            edge_data.append({
                                "Edge": f"{edge[0]}-{edge[1]}",
                                "Congestion": count
                            })
                        
                        st.table(pd.DataFrame(edge_data))
                    
                    with tab2:
                        edge_data = []
                        for edge, count in top_shortest[:20]:
                            edge_data.append({
                                "Edge": f"{edge[0]}-{edge[1]}",
                                "Congestion": count
                            })
                        
                        st.table(pd.DataFrame(edge_data))
                    
                    with tab3:
                        edge_data = []
                        for edge, count in top_traffic[:20]:
                            edge_data.append({
                                "Edge": f"{edge[0]}-{edge[1]}",
                                "Congestion": count
                            })
                        
                        st.table(pd.DataFrame(edge_data))
                else:
                    st.warning("Insufficient data collected from the simulation")
            else:
                st.error("Not enough nodes in the network for the requested number of pairs")
