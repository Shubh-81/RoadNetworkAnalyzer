import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import time

from utils.network_analysis import calculate_network_metrics, identify_congestion_hotspots
from utils.visualization import plot_network, plot_degree_distribution, plot_centrality_distribution

# Configure page
st.set_page_config(
    page_title="Network Metrics | California Road Network Analysis",
    page_icon="🚗",
    layout="wide"
)

# Title
st.title("Network Metrics Analysis")
st.write("Analyze the structural properties of the California road network")

# Check if network is loaded
if 'network' not in st.session_state or st.session_state.network is None:
    st.warning("Please load the California road network from the home page first")
    st.stop()

# Get the network to analyze
network_to_analyze = st.session_state.selected_region if st.session_state.selected_region is not None else st.session_state.network

# Display basic network info
st.subheader("Basic Network Information")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Number of Nodes", network_to_analyze.number_of_nodes())
with col2:
    st.metric("Number of Edges", network_to_analyze.number_of_edges())
with col3:
    density = nx.density(network_to_analyze)
    st.metric("Network Density", f"{density:.8f}")

# Add tabs for different metrics
tab1, tab2, tab3, tab4 = st.tabs(["Degree Analysis", "Centrality Measures", "Path Analysis", "Connectivity"])

with tab1:
    st.subheader("Degree Distribution")
    
    # Plot degree distribution
    plot_degree_distribution(network_to_analyze)
    
    # Calculate degree statistics
    degrees = [d for _, d in network_to_analyze.degree()]
    avg_degree = np.mean(degrees)
    max_degree = max(degrees)
    min_degree = min(degrees)
    median_degree = np.median(degrees)
    
    # Display degree statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Degree", f"{avg_degree:.2f}")
    with col2:
        st.metric("Median Degree", f"{median_degree:.2f}")
    with col3:
        st.metric("Max Degree", max_degree)
    with col4:
        st.metric("Min Degree", min_degree)
    
    # Find nodes with highest degree
    high_degree_nodes = sorted(network_to_analyze.degree(), key=lambda x: x[1], reverse=True)[:20]
    
    # Display as a table
    st.subheader("Nodes with Highest Degree")
    high_degree_df = pd.DataFrame(high_degree_nodes, columns=["Node", "Degree"])
    st.dataframe(high_degree_df)
    
    # Option to visualize high-degree nodes
    if st.checkbox("Visualize high-degree nodes"):
        high_degree_node_ids = high_degree_df["Node"].tolist()
        plot_network(network_to_analyze, highlight_nodes=high_degree_node_ids)

with tab2:
    st.subheader("Centrality Measures")
    st.write("Centrality measures identify important nodes in the network that could be congestion points.")
    
    # Select centrality measure
    centrality_measure = st.selectbox(
        "Select centrality measure",
        ["Betweenness Centrality", "Degree Centrality", "Closeness Centrality", "Eigenvector Centrality"]
    )
    
    # Check if the network is too large for some centrality calculations
    is_large_network = network_to_analyze.number_of_nodes() > 1000
    
    if is_large_network and centrality_measure in ["Betweenness Centrality", "Closeness Centrality"]:
        st.warning(f"The network is large. Computing {centrality_measure} on a sample of nodes.")
        sample_size = st.slider("Sample size", min_value=10, max_value=1000, value=100)
    
    # Compute centrality
    with st.spinner(f"Computing {centrality_measure}..."):
        if centrality_measure == "Betweenness Centrality":
            if is_large_network:
                # Use approximate betweenness with sampling
                centrality = nx.betweenness_centrality(network_to_analyze, k=sample_size, normalized=True)
            else:
                centrality = nx.betweenness_centrality(network_to_analyze, normalized=True)
        
        elif centrality_measure == "Degree Centrality":
            centrality = nx.degree_centrality(network_to_analyze)
        
        elif centrality_measure == "Closeness Centrality":
            if is_large_network:
                # For large networks, compute closeness for a sample of nodes
                sampled_nodes = np.random.choice(list(network_to_analyze.nodes()), 
                                                size=min(sample_size, network_to_analyze.number_of_nodes()), 
                                                replace=False)
                centrality = {}
                for node in sampled_nodes:
                    centrality[node] = nx.closeness_centrality(network_to_analyze, u=node)
            else:
                centrality = nx.closeness_centrality(network_to_analyze)
        
        elif centrality_measure == "Eigenvector Centrality":
            try:
                # Try with a reasonable max_iter value
                centrality = nx.eigenvector_centrality(network_to_analyze, max_iter=100)
            except:
                st.error("Eigenvector centrality calculation did not converge. Using degree centrality instead.")
                centrality = nx.degree_centrality(network_to_analyze)
    
    # Plot centrality distribution
    plot_centrality_distribution(centrality, centrality_measure)
    
    # Top centrality nodes
    st.subheader(f"Top Nodes by {centrality_measure}")
    top_central_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:20]
    central_df = pd.DataFrame(top_central_nodes, columns=["Node", "Centrality Value"])
    st.dataframe(central_df)
    
    # Option to visualize central nodes
    if st.checkbox("Visualize central nodes"):
        central_node_ids = central_df["Node"].tolist()
        plot_network(network_to_analyze, highlight_nodes=central_node_ids)

with tab3:
    st.subheader("Path Analysis")
    
    # Check if path analysis is feasible for the network size
    if network_to_analyze.number_of_nodes() > 5000:
        st.warning("The network is too large for exhaustive path analysis. Using sampling and approximations.")
        
        # Find diameter and average path length for the largest connected component
        with st.spinner("Analyzing largest connected component..."):
            components = list(nx.connected_components(network_to_analyze))
            if components:
                largest_cc = max(components, key=len)
                st.info(f"Largest connected component has {len(largest_cc)} nodes")
                
                # Sample a smaller subgraph from the largest component for analysis
                sample_size = min(1000, len(largest_cc))
                sampled_nodes = list(largest_cc)[:sample_size]
                subgraph = network_to_analyze.subgraph(sampled_nodes).copy()
                
                try:
                    diameter = nx.diameter(subgraph)
                    avg_path = nx.average_shortest_path_length(subgraph)
                    
                    st.metric("Estimated Diameter", diameter)
                    st.metric("Estimated Average Path Length", f"{avg_path:.4f}")
                    st.info("These are estimates based on a sample of the network")
                except:
                    st.error("Error calculating path metrics on the sample")
            else:
                st.error("No connected components found in the network")
    else:
        # For smaller networks, check if it's connected
        if nx.is_connected(network_to_analyze):
            with st.spinner("Calculating exact path metrics..."):
                try:
                    diameter = nx.diameter(network_to_analyze)
                    avg_path = nx.average_shortest_path_length(network_to_analyze)
                    
                    st.metric("Network Diameter", diameter)
                    st.metric("Average Path Length", f"{avg_path:.4f}")
                except:
                    st.error("Error calculating exact path metrics")
        else:
            # For disconnected networks, analyze largest component
            components = list(nx.connected_components(network_to_analyze))
            if components:
                largest_cc = max(components, key=len)
                st.info(f"Network is not connected. Largest component has {len(largest_cc)} nodes " +
                        f"({len(largest_cc)/network_to_analyze.number_of_nodes()*100:.1f}% of the network)")
                
                subgraph = network_to_analyze.subgraph(largest_cc).copy()
                
                with st.spinner("Calculating path metrics for largest component..."):
                    try:
                        diameter = nx.diameter(subgraph)
                        avg_path = nx.average_shortest_path_length(subgraph)
                        
                        st.metric("Diameter (largest component)", diameter)
                        st.metric("Average Path Length (largest component)", f"{avg_path:.4f}")
                    except:
                        st.error("Error calculating path metrics on largest component")
            else:
                st.error("No connected components found in the network")
    
    # Path finding between nodes
    st.subheader("Find Path Between Intersections")
    
    col1, col2 = st.columns(2)
    with col1:
        source = st.number_input("Source Node ID", min_value=0, 
                                max_value=max(network_to_analyze.nodes()) if network_to_analyze.nodes() else 0, 
                                step=1)
    with col2:
        target = st.number_input("Target Node ID", min_value=0, 
                                max_value=max(network_to_analyze.nodes()) if network_to_analyze.nodes() else 0, 
                                step=1)
    
    if st.button("Find Path"):
        if source in network_to_analyze.nodes() and target in network_to_analyze.nodes():
            try:
                path = nx.shortest_path(network_to_analyze, source=source, target=target)
                st.success(f"Found path with {len(path)-1} steps")
                st.write("Path:", path)
                
                # Visualize the path
                plot_network(network_to_analyze, highlight_path=path)
                
                # Calculate path statistics
                path_length = len(path) - 1
                avg_degree_on_path = np.mean([network_to_analyze.degree(node) for node in path])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Path Length", path_length)
                with col2:
                    st.metric("Average Degree on Path", f"{avg_degree_on_path:.2f}")
                
            except nx.NetworkXNoPath:
                st.error("No path exists between the selected nodes")
        else:
            st.error("Source or target node not in the network")

with tab4:
    st.subheader("Connectivity Analysis")
    
    # Connected components analysis
    with st.spinner("Analyzing connected components..."):
        components = list(nx.connected_components(network_to_analyze))
        num_components = len(components)
        
        if num_components > 0:
            largest_cc = max(components, key=len)
            largest_cc_size = len(largest_cc)
            largest_cc_ratio = largest_cc_size / network_to_analyze.number_of_nodes()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of Connected Components", num_components)
            with col2:
                st.metric("Largest Component Size", largest_cc_size)
            with col3:
                st.metric("Largest Component Ratio", f"{largest_cc_ratio:.2%}")
            
            # Component size distribution
            if num_components > 1:
                component_sizes = [len(c) for c in components]
                
                # Create dataframe for plotting
                comp_df = pd.DataFrame({
                    "Component Size": component_sizes
                })
                
                # Histogram of component sizes
                fig = px.histogram(comp_df, x="Component Size", log_y=True,
                                   title="Connected Component Size Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            # Articulation points (cut vertices)
            if network_to_analyze.number_of_nodes() <= 5000:
                st.subheader("Articulation Points (Cut Vertices)")
                st.write("These are critical intersections where removal would increase the number of connected components.")
                
                with st.spinner("Finding articulation points..."):
                    try:
                        cut_vertices = list(nx.articulation_points(network_to_analyze))
                        st.write(f"Found {len(cut_vertices)} articulation points")
                        
                        if len(cut_vertices) > 0:
                            # Display top cut vertices by degree
                            cut_vertex_degrees = [(node, network_to_analyze.degree(node)) for node in cut_vertices]
                            cut_vertex_degrees.sort(key=lambda x: x[1], reverse=True)
                            
                            cut_df = pd.DataFrame(cut_vertex_degrees[:20], columns=["Node ID", "Degree"])
                            st.dataframe(cut_df)
                            
                            # Option to visualize articulation points
                            if st.checkbox("Visualize top articulation points"):
                                top_cut_vertices = cut_df["Node ID"].tolist()
                                plot_network(network_to_analyze, highlight_nodes=top_cut_vertices)
                    except:
                        st.error("Error calculating articulation points")
            else:
                st.info("Articulation point analysis skipped for large networks")
                
            # Edge connectivity
            if network_to_analyze.number_of_nodes() <= 1000:
                st.subheader("Network Connectivity")
                
                # Sample a subgraph if needed
                sample_graph = network_to_analyze
                if network_to_analyze.number_of_nodes() > 100:
                    st.info("Using largest connected component sample for connectivity analysis")
                    sample_nodes = list(largest_cc)[:100]
                    sample_graph = network_to_analyze.subgraph(sample_nodes).copy()
                
                try:
                    edge_connectivity = nx.edge_connectivity(sample_graph)
                    st.metric("Edge Connectivity", edge_connectivity)
                    st.write("This is the minimum number of edges that need to be removed to disconnect the network.")
                except:
                    st.error("Error calculating edge connectivity")
        else:
            st.error("No connected components found in the network")
