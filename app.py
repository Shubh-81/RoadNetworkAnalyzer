import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import os
import io
from utils.data_loader import load_california_network, save_current_network
from utils.network_analysis import calculate_network_metrics
from utils.visualization import plot_network, plot_degree_distribution
from utils.routing import find_shortest_path

# Configure page
st.set_page_config(
    page_title="California Road Network Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'network' not in st.session_state:
    st.session_state.network = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'loading_complete' not in st.session_state:
    st.session_state.loading_complete = False
if 'selected_region' not in st.session_state:
    st.session_state.selected_region = None

# Title and description
st.title("California Road Network Analysis")
st.write("""
This application analyzes the California road network to study traffic congestion and optimize routing.
The network consists of approximately 1.9 million nodes (intersections) and 2.7 million edges (road segments).
""")

# Sidebar for data loading and filtering
with st.sidebar:
    st.header("Data Management")
    
    # Load network button
    if st.button("Load California Road Network"):
        with st.spinner("Loading network data... This may take a while for the full dataset"):
            try:
                st.session_state.network = load_california_network()
                st.session_state.loading_complete = True
                st.success(f"Network loaded successfully with {st.session_state.network.number_of_nodes()} nodes and {st.session_state.network.number_of_edges()} edges")
            except Exception as e:
                st.error(f"Error loading network: {str(e)}")
    
    # Sample data option for testing
    if st.checkbox("Use sample data for testing", value=False):
        with st.spinner("Generating sample network..."):
            # Create a small sample network
            G = nx.grid_2d_graph(20, 20)  # 20x20 grid
            # Convert to regular graph with integer nodes
            G = nx.convert_node_labels_to_integers(G)
            st.session_state.network = G
            st.session_state.loading_complete = True
            st.success(f"Sample network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Region selection
    st.header("Region Selection")
    st.write("Select a subregion of the network to analyze")
    
    if st.session_state.loading_complete:
        # In a real implementation, this would include actual region data
        # For now, we'll use node degree as a proxy for choosing regions
        region_method = st.selectbox(
            "Selection method",
            ["High degree nodes", "Random sample", "Connected component"]
        )
        
        sample_size = st.slider("Sample size", 10, 1000, 100)
        
        if st.button("Select Region"):
            G = st.session_state.network
            if G == None:
               st.error("Please load network first") 
                
            if region_method == "High degree nodes":
                # Get nodes with highest degrees
                degrees = dict(G.degree())
                top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:sample_size]
                subgraph = G.subgraph(top_nodes)
                degrees = [d for _, d in subgraph.degree()]
            
            elif region_method == "Random sample":
                # Get random nodes
                all_nodes = list(G.nodes())
                if len(all_nodes) > sample_size:
                    selected_nodes = np.random.choice(all_nodes, size=sample_size, replace=False)
                    subgraph = G.subgraph(selected_nodes)
                else:
                    subgraph = G.copy()
            
            else:  # Connected component
                # Get largest connected component
                components = list(nx.connected_components(G))
                if components:
                    largest_cc = max(components, key=len)
                    if len(largest_cc) > sample_size:
                        selected_nodes = list(largest_cc)[:sample_size]
                        subgraph = G.subgraph(selected_nodes)
                    else:
                        subgraph = G.subgraph(largest_cc)
                else:
                    st.warning("No connected components found")
                    subgraph = G.copy()
            
            st.session_state.selected_region = subgraph
            st.success(f"Selected region with {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges")

    # Export options
    st.header("Export Results")
    export_format = st.selectbox("Export format", ["CSV", "JSON", "GraphML"])
    
    if st.button("Export Current Network"):
        if st.session_state.network is not None:
            network_to_export = st.session_state.selected_region if st.session_state.selected_region is not None else st.session_state.network
            try:
                file_content, filename = save_current_network(network_to_export, export_format)
                st.download_button(
                    label="Download",
                    data=file_content,
                    file_name=filename,
                    mime="application/octet-stream"
                )
            except Exception as e:
                st.error(f"Error exporting network: {str(e)}")
        else:
            st.warning("No network data to export")

# Main content area
if st.session_state.loading_complete:
    # Display network visualization
    st.header("Network Visualization")
    
    # Choose which network to visualize
    network_to_viz = st.session_state.selected_region if st.session_state.selected_region is not None else st.session_state.network
    
    # For large networks, recommend using sampling
    if network_to_viz.number_of_nodes() > 1000:
        st.warning("Network is very large. Consider selecting a smaller region for detailed visualization.")
        if st.checkbox("Visualize sample anyway", value=False):
            plot_network(network_to_viz, max_nodes=1000)
    else:
        # Visualization options
        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            plot_network(network_to_viz)
        with viz_col2:
            plot_degree_distribution(network_to_viz)
    
    # Network Metrics Section
    st.header("Basic Network Metrics")
    
    with st.spinner("Calculating network metrics..."):
        metrics = calculate_network_metrics(network_to_viz)
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Nodes", metrics["num_nodes"])
            st.metric("Number of Edges", metrics["num_edges"])
            st.metric("Network Density", f"{metrics['density']:.6f}")
            
        with col2:
            st.metric("Average Degree", f"{metrics['avg_degree']:.2f}")
            st.metric("Average Clustering", f"{metrics['avg_clustering']:.4f}")
            st.metric("Average Path Length", 
                     f"{metrics['avg_path_length']:.2f}" if metrics['avg_path_length'] else "N/A")
            
        with col3:
            st.metric("Diameter", 
                     f"{metrics['diameter']}" if metrics['diameter'] else "N/A")
            st.metric("Connected Components", metrics["connected_components"])
            st.metric("Maximum Degree", metrics["max_degree"])
        
        # Store metrics in session state
        st.session_state.metrics = metrics
    
    # Routing functionality
    st.header("Test Path Finding")
    
    col1, col2 = st.columns(2)
    with col1:
        source_node = st.number_input("Source node ID", min_value=0, 
                                      max_value=max(network_to_viz.nodes()) if network_to_viz.nodes() else 0, 
                                      step=1)
    with col2:
        target_node = st.number_input("Target node ID", min_value=0, 
                                       max_value=max(network_to_viz.nodes()) if network_to_viz.nodes() else 0, 
                                       step=1)
    
    if st.button("Find Shortest Path"):
        if source_node in network_to_viz.nodes() and target_node in network_to_viz.nodes():
            path = find_shortest_path(network_to_viz, source_node, target_node)
            if path:
                st.success(f"Found path with {len(path)-1} steps")
                st.write("Path:", path)
                # Visualize the path
                plot_network(network_to_viz, highlight_path=path)
            else:
                st.error("No path found between the selected nodes")
        else:
            st.error("Source or target node not in the selected network")

else:
    # Initial state - instructions for user
    st.info("👈 Please load the California Road Network data from the sidebar to begin analysis")
    
    # Sample visualization of what will be available
    st.subheader("Sample Network Visualization (Preview)")
    
    # Create a small example graph
    example_graph = nx.barabasi_albert_graph(50, 2)
    plot_network(example_graph)
    
    # Sample metrics that will be calculated
    st.subheader("Available Network Metrics")
    st.write("""
    Once the network is loaded, you'll be able to:
    - Calculate degree distribution, centrality, and other network metrics
    - Identify potential congestion hotspots
    - Optimize routing strategies
    - Generate detailed visualizations
    - Export analysis results
    """)
