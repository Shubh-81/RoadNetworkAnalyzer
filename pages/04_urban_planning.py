import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from collections import Counter

from utils.network_analysis import calculate_network_metrics, identify_congestion_hotspots
from utils.visualization import plot_network, plot_degree_distribution
from utils.routing import identify_traffic_bottlenecks

# Configure page
st.set_page_config(
    page_title="Urban Planning | California Road Network Analysis",
    page_icon="🚗",
    layout="wide"
)

# Title
st.title("Urban Planning Insights")
st.write("""
This page provides insights and recommendations for urban planners based on the analysis of the 
California road network. The insights can help in making informed decisions about infrastructure
development and traffic management.
""")

# Check if network is loaded
if 'network' not in st.session_state or st.session_state.network is None:
    st.warning("Please load the California road network from the home page first")
    st.stop()

# Get the network to analyze
network_to_analyze = st.session_state.selected_region if st.session_state.selected_region is not None else st.session_state.network

# Add tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs([
    "Infrastructure Assessment", 
    "Connectivity Analysis", 
    "Growth Simulation",
    "Recommendations"
])

with tab1:
    st.subheader("Infrastructure Assessment")
    st.write("""
    Assess the current state of the road network infrastructure and identify 
    critical points that need attention.
    """)
    
    # Calculate basic network metrics
    with st.spinner("Calculating network metrics..."):
        metrics = calculate_network_metrics(network_to_analyze)
        
        # Display basic metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Intersections", metrics["num_nodes"])
            st.metric("Number of Road Segments", metrics["num_edges"])
        with col2:
            st.metric("Average Intersection Connectivity", f"{metrics['avg_degree']:.2f}")
            st.metric("Maximum Intersection Connectivity", metrics["max_degree"])
        with col3:
            st.metric("Network Density", f"{metrics['density']:.6f}")
            if metrics['avg_path_length']:
                st.metric("Average Path Length", f"{metrics['avg_path_length']:.2f}")
    
    # Identify critical infrastructure points
    st.subheader("Critical Infrastructure Points")
    st.write("""
    Critical infrastructure points are intersections or road segments that, if disrupted, 
    would significantly impact the overall network connectivity and traffic flow.
    """)
    
    # Method selection
    analysis_method = st.selectbox(
        "Analysis Method",
        ["Betweenness Centrality", "Articulation Points", "Edge Betweenness"]
    )
    
    # Run analysis
    if st.button("Identify Critical Points"):
        with st.spinner(f"Analyzing using {analysis_method}..."):
            if analysis_method == "Betweenness Centrality":
                # Use betweenness centrality to identify critical nodes
                # For large networks, use approximation
                if network_to_analyze.number_of_nodes() > 1000:
                    k = min(100, network_to_analyze.number_of_nodes())
                    centrality = nx.betweenness_centrality(network_to_analyze, k=k, normalized=True)
                else:
                    centrality = nx.betweenness_centrality(network_to_analyze, normalized=True)
                
                # Sort nodes by centrality
                critical_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:20]
                
                # Create dataframe for display
                node_data = []
                for node, centrality in critical_nodes:
                    node_data.append({
                        "Intersection ID": node,
                        "Betweenness Centrality": f"{centrality:.6f}",
                        "Connectivity (Degree)": network_to_analyze.degree(node)
                    })
                
                critical_df = pd.DataFrame(node_data)
                
                # Display critical nodes
                st.dataframe(critical_df)
                
                # Visualize critical nodes
                st.subheader("Critical Intersections Visualization")
                critical_node_ids = [node for node, _ in critical_nodes]
                plot_network(network_to_analyze, highlight_nodes=critical_node_ids)
                
                # Recommendations
                st.subheader("Infrastructure Recommendations")
                st.write("""
                Based on betweenness centrality analysis, the intersections listed above are critical 
                for traffic flow across the network. Consider the following recommendations:
                """)
                
                st.write("""
                1. **Capacity Enhancement**: Increase capacity at high-betweenness intersections by adding lanes or improving signal timing.
                2. **Alternative Routes**: Develop alternative routes to reduce dependence on these critical intersections.
                3. **Redundancy**: Add new road segments to create redundant paths around critical intersections.
                4. **Traffic Management**: Implement advanced traffic management systems at these locations.
                """)
            
            elif analysis_method == "Articulation Points":
                # For large networks, work with a sample or the largest connected component
                if network_to_analyze.number_of_nodes() > 5000:
                    st.info("Network is large. Analyzing the largest connected component.")
                    components = list(nx.connected_components(network_to_analyze))
                    if components:
                        largest_cc = max(components, key=len)
                        subgraph = network_to_analyze.subgraph(largest_cc).copy()
                        st.write(f"Analyzing a connected component with {subgraph.number_of_nodes()} intersections.")
                    else:
                        subgraph = network_to_analyze
                else:
                    subgraph = network_to_analyze
                
                try:
                    # Find articulation points (cut vertices)
                    cut_vertices = list(nx.articulation_points(subgraph))
                    
                    if cut_vertices:
                        st.success(f"Found {len(cut_vertices)} articulation points")
                        
                        # Analyze top articulation points by degree
                        art_points_data = []
                        for node in cut_vertices[:20]:  # Limit to top 20
                            art_points_data.append({
                                "Intersection ID": node,
                                "Connectivity (Degree)": subgraph.degree(node)
                            })
                        
                        art_points_df = pd.DataFrame(art_points_data)
                        art_points_df = art_points_df.sort_values("Connectivity (Degree)", ascending=False)
                        
                        st.dataframe(art_points_df)
                        
                        # Visualize articulation points
                        st.subheader("Articulation Points Visualization")
                        plot_network(subgraph, highlight_nodes=art_points_df["Intersection ID"].tolist()[:10])
                        
                        # Recommendations
                        st.subheader("Infrastructure Recommendations")
                        st.write("""
                        Articulation points are intersections that, if removed, would disconnect parts of the 
                        network. These represent critical vulnerabilities. Consider the following recommendations:
                        """)
                        
                        st.write("""
                        1. **Network Redundancy**: Add new road connections to create alternative paths around articulation points.
                        2. **Critical Infrastructure Protection**: Ensure these intersections have robust maintenance schedules.
                        3. **Disaster Planning**: Include these points in emergency response planning.
                        4. **Access Improvement**: Create additional access points to isolated areas connected through articulation points.
                        """)
                    else:
                        st.info("No articulation points found. The network has good redundancy.")
                except Exception as e:
                    st.error(f"Error in articulation point analysis: {str(e)}")
            
            elif analysis_method == "Edge Betweenness":
                # Identify critical edges using edge betweenness
                if network_to_analyze.number_of_edges() > 5000:
                    k = min(100, network_to_analyze.number_of_nodes())
                    edge_centrality = nx.edge_betweenness_centrality(network_to_analyze, k=k, normalized=True)
                else:
                    edge_centrality = nx.edge_betweenness_centrality(network_to_analyze, normalized=True)
                
                # Sort edges by centrality
                critical_edges = sorted(edge_centrality.items(), key=lambda x: x[1], reverse=True)[:20]
                
                # Create dataframe for display
                edge_data = []
                for edge, centrality in critical_edges:
                    edge_data.append({
                        "Road Segment": f"{edge[0]}-{edge[1]}",
                        "Start Intersection": edge[0],
                        "End Intersection": edge[1],
                        "Edge Betweenness": f"{centrality:.6f}"
                    })
                
                critical_edges_df = pd.DataFrame(edge_data)
                
                # Display critical edges
                st.dataframe(critical_edges_df)
                
                # Collect nodes involved in critical edges for visualization
                critical_nodes = set()
                for edge, _ in critical_edges:
                    critical_nodes.add(edge[0])
                    critical_nodes.add(edge[1])
                
                # Visualize critical edges
                st.subheader("Critical Road Segments Visualization")
                plot_network(network_to_analyze, highlight_nodes=list(critical_nodes))
                
                # Recommendations
                st.subheader("Infrastructure Recommendations")
                st.write("""
                Based on edge betweenness analysis, the road segments listed above are critical 
                for traffic flow across the network. Consider the following recommendations:
                """)
                
                st.write("""
                1. **Capacity Enhancement**: Increase capacity on high-betweenness road segments by adding lanes.
                2. **Parallel Routes**: Develop parallel routes to distribute traffic.
                3. **Maintenance Priority**: Prioritize maintenance for these critical segments.
                4. **Traffic Management**: Implement dynamic traffic management on these segments.
                """)

with tab2:
    st.subheader("Connectivity Analysis")
    st.write("""
    Analyze the connectivity of the road network to identify areas with poor connectivity and 
    suggest improvements.
    """)
    
    # Connectivity metrics
    with st.spinner("Calculating connectivity metrics..."):
        # Calculate connected components
        components = list(nx.connected_components(network_to_analyze))
        num_components = len(components)
        
        if num_components > 0:
            # Sort components by size
            components = sorted(components, key=len, reverse=True)
            largest_cc = components[0]
            largest_cc_size = len(largest_cc)
            largest_cc_ratio = largest_cc_size / network_to_analyze.number_of_nodes()
            
            # Display connectivity metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Connected Components", num_components)
            with col2:
                st.metric("Largest Component Size", largest_cc_size)
            with col3:
                st.metric("Largest Component Ratio", f"{largest_cc_ratio:.2%}")
            
            # Analyze component sizes
            if num_components > 1:
                component_sizes = [len(c) for c in components]
                
                # Create histogram of component sizes
                fig = px.histogram(
                    pd.DataFrame({"Component Size": component_sizes}),
                    x="Component Size",
                    log_y=True,
                    title="Connected Component Size Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Analyze isolated components
                if num_components > 1:
                    st.subheader("Isolated Network Areas")
                    st.write("""
                    These are areas of the road network that are not connected to the main network.
                    Improving connectivity to these areas can enhance overall network efficiency.
                    """)
                    
                    # Analyze top isolated components (excluding the largest)
                    isolated_data = []
                    for i, component in enumerate(components[1:6]):  # Take 2nd to 6th largest
                        isolated_data.append({
                            "Component ID": i+2,  # Start from 2
                            "Number of Intersections": len(component),
                            "Percentage of Network": f"{len(component)/network_to_analyze.number_of_nodes()*100:.2f}%"
                        })
                    
                    isolated_df = pd.DataFrame(isolated_data)
                    st.dataframe(isolated_df)
                    
                    # Option to visualize an isolated component
                    if len(components) > 1:
                        component_to_show = st.selectbox(
                            "Select component to visualize",
                            ["Largest Component"] + [f"Component {i+2}" for i in range(min(5, len(components)-1))]
                        )
                        
                        if component_to_show == "Largest Component":
                            subgraph = network_to_analyze.subgraph(components[0]).copy()
                        else:
                            # Extract component number from string
                            comp_num = int(component_to_show.split(" ")[1]) - 1
                            if comp_num < len(components):
                                subgraph = network_to_analyze.subgraph(components[comp_num]).copy()
                            else:
                                st.error("Component not found")
                                subgraph = None
                        
                        if subgraph:
                            st.write(f"Visualizing {component_to_show} with {subgraph.number_of_nodes()} intersections")
                            plot_network(subgraph)
                
                # Connectivity improvement recommendations
                st.subheader("Connectivity Improvement Recommendations")
                
                if num_components > 1:
                    st.write("""
                    The network contains disconnected components. Here are recommendations to improve connectivity:
                    """)
                    
                    st.write("""
                    1. **New Connections**: Identify the closest points between the main network and isolated components.
                    2. **Bridge Segments**: Add new road segments to connect isolated components to the main network.
                    3. **Priority Connections**: Prioritize connecting larger isolated components first.
                    4. **Alternative Access Routes**: Create multiple access points to each isolated component for redundancy.
                    """)
                else:
                    st.success("The network is fully connected. Focus on improving redundancy and reliability.")
        else:
            st.error("No connected components found in the network")
    
    # Network resilience analysis
    st.subheader("Network Resilience Analysis")
    st.write("""
    Analyze how resilient the network is to disruptions such as road closures or traffic accidents.
    """)
    
    # Run resilience analysis
    if st.button("Analyze Network Resilience"):
        with st.spinner("Analyzing network resilience..."):
            # For large networks, work with the largest component or a sample
            if network_to_analyze.number_of_nodes() > 5000:
                st.info("Network is large. Analyzing a sample of the largest connected component.")
                
                if num_components > 0:
                    sample_size = min(5000, len(largest_cc))
                    sampled_nodes = list(largest_cc)[:sample_size]
                    subgraph = network_to_analyze.subgraph(sampled_nodes).copy()
                else:
                    sample_size = min(5000, network_to_analyze.number_of_nodes())
                    sampled_nodes = list(network_to_analyze.nodes())[:sample_size]
                    subgraph = network_to_analyze.subgraph(sampled_nodes).copy()
            else:
                subgraph = network_to_analyze
            
            try:
                # Calculate initial connectivity metrics
                initial_components = nx.number_connected_components(subgraph)
                
                if nx.is_connected(subgraph):
                    initial_avg_path = nx.average_shortest_path_length(subgraph)
                else:
                    # Use largest connected component for path length
                    largest_cc = max(nx.connected_components(subgraph), key=len)
                    sg = subgraph.subgraph(largest_cc).copy()
                    initial_avg_path = nx.average_shortest_path_length(sg)
                
                # Identify high betweenness nodes
                n = min(10, subgraph.number_of_nodes() // 10)  # Remove up to 10% of nodes
                if subgraph.number_of_nodes() > 1000:
                    k = min(100, subgraph.number_of_nodes())
                    centrality = nx.betweenness_centrality(subgraph, k=k, normalized=True)
                else:
                    centrality = nx.betweenness_centrality(subgraph, normalized=True)
                
                high_centrality_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:n]
                
                # Simulate removal of high centrality nodes
                resilience_data = []
                
                # Baseline
                resilience_data.append({
                    "Scenario": "Baseline",
                    "Nodes Removed": 0,
                    "Connected Components": initial_components,
                    "Avg Path Length": initial_avg_path,
                    "Largest Component Ratio": 1.0
                })
                
                # Create a copy of the subgraph to modify
                test_graph = subgraph.copy()
                
                # Remove nodes one by one and measure impact
                for i, (node, centrality) in enumerate(high_centrality_nodes):
                    # Remove the node
                    if node in test_graph:
                        test_graph.remove_node(node)
                        
                        # Calculate new metrics
                        new_components = nx.number_connected_components(test_graph)
                        
                        # Find largest connected component ratio
                        cc_sizes = [len(c) for c in nx.connected_components(test_graph)]
                        largest_cc_size = max(cc_sizes) if cc_sizes else 0
                        largest_cc_ratio = largest_cc_size / subgraph.number_of_nodes()
                        
                        # Calculate average path length on largest component if connected
                        if cc_sizes:
                            largest_cc = max(nx.connected_components(test_graph), key=len)
                            sg = test_graph.subgraph(largest_cc).copy()
                            if sg.number_of_nodes() > 1:
                                try:
                                    new_avg_path = nx.average_shortest_path_length(sg)
                                except:
                                    new_avg_path = None
                            else:
                                new_avg_path = None
                        else:
                            new_avg_path = None
                        
                        # Add to data
                        resilience_data.append({
                            "Scenario": f"Remove Top {i+1} Nodes",
                            "Nodes Removed": i+1,
                            "Connected Components": new_components,
                            "Avg Path Length": new_avg_path,
                            "Largest Component Ratio": largest_cc_ratio
                        })
                
                # Convert to DataFrame
                resilience_df = pd.DataFrame(resilience_data)
                
                # Display results
                st.subheader("Resilience Simulation Results")
                st.dataframe(resilience_df)
                
                # Plot resilience metrics
                fig = go.Figure()
                
                # Number of connected components
                fig.add_trace(go.Scatter(
                    x=resilience_df["Nodes Removed"],
                    y=resilience_df["Connected Components"],
                    mode='lines+markers',
                    name='Connected Components'
                ))
                
                # Largest component ratio
                fig.add_trace(go.Scatter(
                    x=resilience_df["Nodes Removed"],
                    y=resilience_df["Largest Component Ratio"],
                    mode='lines+markers',
                    name='Largest Component Ratio',
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    title='Network Resilience to Node Removal',
                    xaxis_title='Number of High-Centrality Nodes Removed',
                    yaxis_title='Number of Connected Components',
                    yaxis2=dict(
                        title='Largest Component Ratio',
                        overlaying='y',
                        side='right'
                    ),
                    legend=dict(x=0.01, y=0.99),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Plot average path length if available
                path_length_data = resilience_df.dropna(subset=["Avg Path Length"])
                if len(path_length_data) > 1:
                    fig = px.line(
                        path_length_data,
                        x="Nodes Removed",
                        y="Avg Path Length",
                        title="Impact on Average Path Length",
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations based on resilience analysis
                st.subheader("Resilience Improvement Recommendations")
                
                if len(resilience_data) > 1:
                    # Calculate vulnerabilities
                    cc_increase = resilience_data[-1]["Connected Components"] - resilience_data[0]["Connected Components"]
                    cc_percent = cc_increase / max(1, resilience_data[0]["Connected Components"]) * 100
                    
                    largest_cc_decrease = resilience_data[0]["Largest Component Ratio"] - resilience_data[-1]["Largest Component Ratio"]
                    largest_cc_percent = largest_cc_decrease / max(0.01, resilience_data[0]["Largest Component Ratio"]) * 100
                    
                    if cc_increase > 0 or largest_cc_decrease > 0.1:
                        st.warning(f"""
                        The network shows significant vulnerability to targeted disruptions:
                        - Connected components increased by {cc_increase} ({cc_percent:.1f}%)
                        - Largest component ratio decreased by {largest_cc_decrease:.4f} ({largest_cc_percent:.1f}%)
                        """)
                        
                        st.write("""
                        Recommendations to improve network resilience:
                        
                        1. **Redundant Connections**: Add alternative routes around critical intersections.
                        2. **Distributed Centrality**: Redesign to distribute betweenness centrality more evenly.
                        3. **Critical Node Protection**: Improve infrastructure at high-centrality nodes.
                        4. **Alternate Route Signage**: Ensure drivers can easily find alternative routes.
                        5. **Grid Structure**: Work towards a more grid-like structure in future development.
                        """)
                    else:
                        st.success("""
                        The network shows good resilience to disruptions. Continue to maintain this 
                        property in future developments.
                        """)
            except Exception as e:
                st.error(f"Error in resilience analysis: {str(e)}")

with tab3:
    st.subheader("Urban Growth Simulation")
    st.write("""
    Simulate how urban growth and new developments would affect the road network and traffic patterns.
    """)
    
    # Simulation parameters
    st.subheader("Growth Scenario Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        growth_pattern = st.selectbox(
            "Growth Pattern",
            ["Expand from High-Degree Nodes", "New Development Zones", "Connect Isolated Components"]
        )
        
        num_new_intersections = st.slider("Number of New Intersections", 5, 100, 20)
    
    with col2:
        connection_strategy = st.selectbox(
            "Connection Strategy",
            ["Preferential Attachment", "Grid Pattern", "Random Connections"]
        )
        
        num_new_connections = st.slider("Number of New Road Segments", 5, 200, 30)
    
    # Run simulation
    if st.button("Simulate Urban Growth"):
        with st.spinner("Simulating urban growth..."):
            # Create a copy of the network to modify
            growth_network = network_to_analyze.copy()
            original_size = growth_network.number_of_nodes()
            
            # Track new nodes and edges for visualization
            new_nodes = []
            new_edges = []
            
            try:
                # Determine starting points for growth based on selected pattern
                if growth_pattern == "Expand from High-Degree Nodes":
                    # Use high-degree nodes as starting points
                    degrees = dict(growth_network.degree())
                    starting_points = sorted(degrees, key=degrees.get, reverse=True)[:20]
                
                elif growth_pattern == "New Development Zones":
                    # Create a few development zones (randomly selected nodes)
                    all_nodes = list(growth_network.nodes())
                    starting_points = np.random.choice(all_nodes, size=min(5, len(all_nodes)), replace=False)
                
                elif growth_pattern == "Connect Isolated Components":
                    # Use nodes from different connected components
                    components = list(nx.connected_components(growth_network))
                    
                    starting_points = []
                    for i, component in enumerate(components[:min(5, len(components))]):
                        # Select one node from each component
                        starting_points.append(list(component)[0])
                
                # Add new nodes and edges based on the connection strategy
                if connection_strategy == "Preferential Attachment":
                    # Add nodes with preferential attachment (more connections to high-degree nodes)
                    for i in range(num_new_intersections):
                        # Create new node
                        new_node = max(growth_network.nodes()) + 1
                        growth_network.add_node(new_node)
                        new_nodes.append(new_node)
                        
                        # Choose existing nodes to connect to based on degree
                        degrees = dict(growth_network.degree())
                        # Convert degrees to probabilities
                        total_degree = sum(degrees.values())
                        probabilities = [d/total_degree for d in degrees.values()]
                        
                        # Choose a number of connections for this node
                        num_connections = min(
                            np.random.randint(1, 5),  # 1-4 connections
                            len(growth_network.nodes()) - 1  # Can't exceed number of other nodes
                        )
                        
                        # Connect to existing nodes
                        existing_nodes = list(degrees.keys())
                        connected_to = np.random.choice(
                            existing_nodes, 
                            size=min(num_connections, len(existing_nodes)),
                            replace=False,
                            p=probabilities
                        )
                        
                        for existing_node in connected_to:
                            growth_network.add_edge(new_node, existing_node)
                            new_edges.append((new_node, existing_node))
                
                elif connection_strategy == "Grid Pattern":
                    # Create a more grid-like structure around starting points
                    for starting_point in starting_points:
                        # Create a small grid around this point
                        grid_size = int(np.sqrt(num_new_intersections / len(starting_points)))
                        grid_size = max(2, grid_size)  # At least 2x2
                        
                        # Create a grid of new nodes
                        base_new_node = max(growth_network.nodes()) + 1
                        
                        # Add grid nodes
                        grid_nodes = []
                        for i in range(grid_size):
                            row_nodes = []
                            for j in range(grid_size):
                                new_node = base_new_node + i*grid_size + j
                                growth_network.add_node(new_node)
                                row_nodes.append(new_node)
                                new_nodes.append(new_node)
                            grid_nodes.append(row_nodes)
                        
                        # Connect grid nodes to each other
                        for i in range(grid_size):
                            for j in range(grid_size):
                                current = grid_nodes[i][j]
                                
                                # Connect to right neighbor
                                if j < grid_size - 1:
                                    right = grid_nodes[i][j+1]
                                    growth_network.add_edge(current, right)
                                    new_edges.append((current, right))
                                
                                # Connect to bottom neighbor
                                if i < grid_size - 1:
                                    bottom = grid_nodes[i+1][j]
                                    growth_network.add_edge(current, bottom)
                                    new_edges.append((current, bottom))
                        
                        # Connect the grid to the starting point
                        # Choose one of the corner nodes to connect
                        corner_node = grid_nodes[0][0]
                        growth_network.add_edge(starting_point, corner_node)
                        new_edges.append((starting_point, corner_node))
                
                elif connection_strategy == "Random Connections":
                    # Add random nodes and connect them randomly
                    for i in range(num_new_intersections):
                        # Create new node
                        new_node = max(growth_network.nodes()) + 1
                        growth_network.add_node(new_node)
                        new_nodes.append(new_node)
                        
                        # Connect to a random existing node from the starting points
                        connect_to = np.random.choice(starting_points)
                        growth_network.add_edge(new_node, connect_to)
                        new_edges.append((new_node, connect_to))
                        
                        # Also add some random connections between new nodes
                        if len(new_nodes) > 1 and np.random.random() < 0.5:
                            # Connect to a previously added new node
                            other_new = np.random.choice(new_nodes[:-1])  # Exclude the current node
                            growth_network.add_edge(new_node, other_new)
                            new_edges.append((new_node, other_new))
                
                # Add additional random connections if requested
                remaining_connections = num_new_connections - len(new_edges)
                
                if remaining_connections > 0:
                    for _ in range(remaining_connections):
                        # Choose two random nodes, with higher probability for new nodes
                        all_nodes = list(growth_network.nodes())
                        
                        # Assign higher probabilities to new nodes
                        probabilities = [5.0 if node in new_nodes else 1.0 for node in all_nodes]
                        probabilities = [p/sum(probabilities) for p in probabilities]
                        
                        # Select two nodes
                        nodes = np.random.choice(all_nodes, size=2, replace=False, p=probabilities)
                        
                        # Add edge if it doesn't already exist
                        if not growth_network.has_edge(nodes[0], nodes[1]):
                            growth_network.add_edge(nodes[0], nodes[1])
                            new_edges.append((nodes[0], nodes[1]))
                
                # Calculate network metrics before and after
                original_metrics = calculate_network_metrics(network_to_analyze)
                new_metrics = calculate_network_metrics(growth_network)
                
                # Display results
                st.subheader("Growth Simulation Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Intersections", 
                        new_metrics["num_nodes"],
                        f"+{new_metrics['num_nodes'] - original_metrics['num_nodes']}"
                    )
                with col2:
                    st.metric(
                        "Road Segments", 
                        new_metrics["num_edges"],
                        f"+{new_metrics['num_edges'] - original_metrics['num_edges']}"
                    )
                with col3:
                    st.metric(
                        "Avg Connectivity", 
                        f"{new_metrics['avg_degree']:.2f}",
                        f"{new_metrics['avg_degree'] - original_metrics['avg_degree']:.2f}"
                    )
                
                # Visualize the growth
                st.subheader("Growth Visualization")
                
                # Create a visualization highlighting new nodes and edges
                plot_network(growth_network, highlight_nodes=new_nodes)
                
                st.write(f"Added {len(new_nodes)} new intersections and {len(new_edges)} new road segments")
                
                # Compare connected components before and after
                original_components = nx.number_connected_components(network_to_analyze)
                new_components = nx.number_connected_components(growth_network)
                
                if original_components != new_components:
                    st.success(f"Growth reduced the number of isolated areas from {original_components} to {new_components}")
                
                # Impact on traffic flow
                st.subheader("Impact on Traffic Flow")
                
                # Identify potential congestion points in the new network
                hotspots = identify_congestion_hotspots(growth_network, method="betweenness", top_n=10)
                
                # Check if new intersections are among the congestion hotspots
                new_hotspots = hotspots[hotspots["Node"].isin(new_nodes)]
                
                if not new_hotspots.empty:
                    st.warning(f"{len(new_hotspots)} new intersections may become congestion hotspots:")
                    st.dataframe(new_hotspots)
                else:
                    st.success("No new intersections are predicted to become congestion hotspots")
                
                # Calculate impact on path lengths
                if nx.is_connected(network_to_analyze) and nx.is_connected(growth_network):
                    # Sample some node pairs to calculate average path length
                    all_nodes = list(network_to_analyze.nodes())
                    sample_size = min(20, len(all_nodes))
                    sample_nodes = np.random.choice(all_nodes, size=sample_size, replace=False)
                    
                    original_paths = []
                    new_paths = []
                    
                    for i in range(sample_size):
                        for j in range(i+1, sample_size):
                            try:
                                original_path = nx.shortest_path_length(network_to_analyze, source=sample_nodes[i], target=sample_nodes[j])
                                new_path = nx.shortest_path_length(growth_network, source=sample_nodes[i], target=sample_nodes[j])
                                
                                original_paths.append(original_path)
                                new_paths.append(new_path)
                            except:
                                continue
                    
                    if original_paths and new_paths:
                        avg_original = np.mean(original_paths)
                        avg_new = np.mean(new_paths)
                        
                        st.metric(
                            "Avg Path Length (Sample)", 
                            f"{avg_new:.2f}",
                            f"{avg_original - avg_new:.2f}"
                        )
                        
                        if avg_new < avg_original:
                            st.success(f"The new development reduces average travel distances by {(avg_original - avg_new)/avg_original*100:.1f}%")
                        else:
                            st.info("The new development does not significantly reduce travel distances")
            
            except Exception as e:
                st.error(f"Error in growth simulation: {str(e)}")

with tab4:
    st.subheader("Urban Planning Recommendations")
    st.write("""
    Based on the analysis of the California road network, here are key recommendations for urban planners
    to improve traffic flow, reduce congestion, and enhance infrastructure resilience.
    """)
    
    # Calculate basic network metrics for recommendations
    metrics = calculate_network_metrics(network_to_analyze)
    
    # Identify bottlenecks for recommendations
    bottlenecks = identify_traffic_bottlenecks(network_to_analyze, num_bottlenecks=10)
    
    # Create recommendations
    st.subheader("Traffic Management Recommendations")
    
    st.markdown("""
    1. **Congestion Hotspot Mitigation**
       - Implement intelligent traffic signaling at high-betweenness intersections
       - Consider dedicated turn lanes at intersections with high degree centrality
       - Deploy real-time traffic monitoring at key bottlenecks
    
    2. **Traffic Flow Optimization**
       - Develop one-way street systems in areas with grid-like patterns to improve flow
       - Implement time-of-day based lane direction changes on major corridors
       - Consider traffic calming measures in residential areas adjacent to major thoroughfares
    
    3. **Public Transportation Enhancement**
       - Prioritize public transit routes along high-betweenness corridors
       - Develop dedicated bus lanes on roads with high edge betweenness
       - Create transit hubs at intersections with high degree centrality
    """)
    
    st.subheader("Infrastructure Development Recommendations")
    
    st.markdown("""
    1. **Network Connectivity Improvements**
       - Identify and connect isolated network components to improve overall accessibility
       - Add strategic road connections to reduce average path length across the network
       - Create redundant connections around critical articulation points
    
    2. **Road Capacity Enhancements**
       - Prioritize widening of road segments with high edge betweenness
       - Upgrade intersections with high betweenness centrality
       - Develop alternative routes parallel to congested corridors
    
    3. **Resilience Planning**
       - Design redundant paths around critical network nodes
       - Develop emergency response routes that avoid high-betweenness bottlenecks
       - Prioritize maintenance for critical infrastructure components
    """)
    
    st.subheader("Urban Growth Planning Recommendations")
    
    st.markdown("""
    1. **New Development Guidelines**
       - Require multiple access points for new developments to avoid creating bottlenecks
       - Design new road networks with grid patterns to distribute traffic more evenly
       - Connect new developments to multiple points in the existing network
    
    2. **Land Use Integration**
       - Promote mixed-use development to reduce trip distances and overall traffic demand
       - Align high-density development with areas of high network connectivity
       - Create local service centers to reduce dependence on distant urban cores
    
    3. **Alternative Transportation Infrastructure**
       - Develop bicycle and pedestrian networks parallel to high-congestion corridors
       - Integrate alternative transportation modes at key network nodes
       - Design shared spaces in areas with high intersection density
    """)
    
    # Export recommendations
    if st.button("Export Recommendations"):
        # Create a comprehensive PDF-like text of recommendations
        recommendations_text = """
        # Urban Planning Recommendations for California Road Network
        
        ## Executive Summary
        
        This document provides comprehensive recommendations for urban planners based on
        network analysis of the California road infrastructure. The recommendations focus on
        improving traffic flow, reducing congestion, and enhancing infrastructure resilience.
        
        ## Network Analysis Overview
        
        - Network Size: {nodes} intersections, {edges} road segments
        - Average Connectivity: {avg_degree:.2f} connections per intersection
        - Network Density: {density:.6f}
        
        ## Traffic Management Recommendations
        
        1. Congestion Hotspot Mitigation
           - Implement intelligent traffic signaling at high-betweenness intersections
           - Consider dedicated turn lanes at intersections with high degree centrality
           - Deploy real-time traffic monitoring at key bottlenecks
        
        2. Traffic Flow Optimization
           - Develop one-way street systems in areas with grid-like patterns to improve flow
           - Implement time-of-day based lane direction changes on major corridors
           - Consider traffic calming measures in residential areas adjacent to major thoroughfares
        
        3. Public Transportation Enhancement
           - Prioritize public transit routes along high-betweenness corridors
           - Develop dedicated bus lanes on roads with high edge betweenness
           - Create transit hubs at intersections with high degree centrality
        
        ## Infrastructure Development Recommendations
        
        1. Network Connectivity Improvements
           - Identify and connect isolated network components to improve overall accessibility
           - Add strategic road connections to reduce average path length across the network
           - Create redundant connections around critical articulation points
        
        2. Road Capacity Enhancements
           - Prioritize widening of road segments with high edge betweenness
           - Upgrade intersections with high betweenness centrality
           - Develop alternative routes parallel to congested corridors
        
        3. Resilience Planning
           - Design redundant paths around critical network nodes
           - Develop emergency response routes that avoid high-betweenness bottlenecks
           - Prioritize maintenance for critical infrastructure components
        
        ## Urban Growth Planning Recommendations
        
        1. New Development Guidelines
           - Require multiple access points for new developments to avoid creating bottlenecks
           - Design new road networks with grid patterns to distribute traffic more evenly
           - Connect new developments to multiple points in the existing network
        
        2. Land Use Integration
           - Promote mixed-use development to reduce trip distances and overall traffic demand
           - Align high-density development with areas of high network connectivity
           - Create local service centers to reduce dependence on distant urban cores
        
        3. Alternative Transportation Infrastructure
           - Develop bicycle and pedestrian networks parallel to high-congestion corridors
           - Integrate alternative transportation modes at key network nodes
           - Design shared spaces in areas with high intersection density
        
        ## Implementation Priority
        
        1. High Priority (0-2 years)
           - Address identified congestion hotspots
           - Improve connectivity of isolated network components
           - Enhance resilience around critical articulation points
        
        2. Medium Priority (2-5 years)
           - Develop alternative routes around high-betweenness corridors
           - Implement public transportation enhancements
           - Establish connectivity requirements for new developments
        
        3. Long-term Strategies (5+ years)
           - Gradual network restructuring to improve overall connectivity
           - Development of comprehensive alternative transportation networks
           - Strategic land use planning integrated with transportation networks
        """.format(
            nodes=metrics["num_nodes"],
            edges=metrics["num_edges"],
            avg_degree=metrics["avg_degree"],
            density=metrics["density"]
        )
        
        # Create a download button
        st.download_button(
            label="Download Recommendations",
            data=recommendations_text,
            file_name="urban_planning_recommendations.txt",
            mime="text/plain"
        )
    
    # Interactive planning tool
    st.subheader("Interactive Planning Priority Tool")
    st.write("""
    Use this tool to prioritize infrastructure improvements based on different objectives.
    Adjust the sliders to reflect the relative importance of each goal.
    """)
    
    # Priority sliders
    congestion_priority = st.slider("Congestion Reduction Priority", 0, 10, 5)
    connectivity_priority = st.slider("Network Connectivity Priority", 0, 10, 5)
    resilience_priority = st.slider("Network Resilience Priority", 0, 10, 5)
    
    if st.button("Generate Prioritized Recommendations"):
        with st.spinner("Analyzing priorities..."):
            # Calculate a priority score for different intervention types
            
            # Score categories based on priorities
            congestion_score = congestion_priority / 10.0
            connectivity_score = connectivity_priority / 10.0
            resilience_score = resilience_priority / 10.0
            
            # Normalize to ensure total is 1.0
            total = congestion_score + connectivity_score + resilience_score
            congestion_score /= total
            connectivity_score /= total
            resilience_score /= total
            
            # Generate prioritized recommendations
            st.subheader("Prioritized Infrastructure Interventions")
            
            # Create dataframe with recommendations and scores
            interventions = [
                {
                    "Intervention": "Upgrade high-betweenness intersections",
                    "Type": "Congestion Reduction",
                    "Priority Score": round(0.9 * congestion_score + 0.1 * resilience_score, 2),
                    "Estimated Impact": "High",
                    "Implementation Complexity": "Medium"
                },
                {
                    "Intervention": "Add parallel routes to congested corridors",
                    "Type": "Congestion Reduction",
                    "Priority Score": round(0.8 * congestion_score + 0.2 * connectivity_score, 2),
                    "Estimated Impact": "High",
                    "Implementation Complexity": "High"
                },
                {
                    "Intervention": "Implement smart traffic signals",
                    "Type": "Congestion Reduction",
                    "Priority Score": round(0.7 * congestion_score, 2),
                    "Estimated Impact": "Medium",
                    "Implementation Complexity": "Low"
                },
                {
                    "Intervention": "Connect isolated network components",
                    "Type": "Network Connectivity",
                    "Priority Score": round(0.9 * connectivity_score + 0.1 * resilience_score, 2),
                    "Estimated Impact": "High",
                    "Implementation Complexity": "High"
                },
                {
                    "Intervention": "Add strategic links to reduce path lengths",
                    "Type": "Network Connectivity",
                    "Priority Score": round(0.8 * connectivity_score + 0.2 * congestion_score, 2),
                    "Estimated Impact": "Medium",
                    "Implementation Complexity": "Medium"
                },
                {
                    "Intervention": "Create grid connections in developing areas",
                    "Type": "Network Connectivity",
                    "Priority Score": round(0.7 * connectivity_score, 2),
                    "Estimated Impact": "Medium",
                    "Implementation Complexity": "Medium"
                },
                {
                    "Intervention": "Add redundant paths around articulation points",
                    "Type": "Network Resilience",
                    "Priority Score": round(0.9 * resilience_score + 0.1 * connectivity_score, 2),
                    "Estimated Impact": "High",
                    "Implementation Complexity": "Medium"
                },
                {
                    "Intervention": "Strengthen critical infrastructure at high-betweenness nodes",
                    "Type": "Network Resilience",
                    "Priority Score": round(0.8 * resilience_score + 0.2 * congestion_score, 2),
                    "Estimated Impact": "Medium",
                    "Implementation Complexity": "Low"
                },
                {
                    "Intervention": "Create emergency alternative routes",
                    "Type": "Network Resilience",
                    "Priority Score": round(0.7 * resilience_score, 2),
                    "Estimated Impact": "Medium",
                    "Implementation Complexity": "Medium"
                }
            ]
            
            # Create DataFrame and sort by priority score
            interventions_df = pd.DataFrame(interventions)
            interventions_df = interventions_df.sort_values("Priority Score", ascending=False)
            
            # Display as a table
            st.dataframe(interventions_df)
            
            # Visualize priorities
            fig = px.bar(
                interventions_df.head(5),  # Top 5 interventions
                x="Intervention",
                y="Priority Score",
                color="Type",
                title="Top 5 Prioritized Interventions"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary based on priorities
            st.subheader("Implementation Strategy")
            
            # Determine overall strategy based on highest priority
            if congestion_priority >= connectivity_priority and congestion_priority >= resilience_priority:
                st.write("""
                ### Congestion-Focused Strategy
                
                Based on your priorities, the recommended strategy focuses on congestion reduction:
                
                1. **Short-term (0-2 years):**
                   - Implement intelligent traffic management systems at high-betweenness intersections
                   - Optimize signal timing in congested corridors
                   - Introduce peak-hour lane management
                
                2. **Medium-term (2-5 years):**
                   - Develop parallel routes to most congested corridors
                   - Implement public transportation improvements in high-traffic areas
                   - Convert appropriate streets to one-way systems to improve flow
                
                3. **Long-term (5+ years):**
                   - Major capacity enhancements on critical corridors
                   - Develop comprehensive alternative transportation network
                   - Implement land use policies that reduce traffic demand
                """)
            
            elif connectivity_priority >= congestion_priority and connectivity_priority >= resilience_priority:
                st.write("""
                ### Connectivity-Focused Strategy
                
                Based on your priorities, the recommended strategy focuses on network connectivity:
                
                1. **Short-term (0-2 years):**
                   - Identify and implement quick connections between nearby network segments
                   - Improve accessibility to isolated areas
                   - Enhance signage and wayfinding to improve network utilization
                
                2. **Medium-term (2-5 years):**
                   - Connect major isolated components to the main network
                   - Develop strategic links that significantly reduce average path lengths
                   - Create multi-modal connection points
                
                3. **Long-term (5+ years):**
                   - Comprehensive network restructuring to improve connectivity
                   - Develop grid-like structures in developing areas
                   - Implement policies requiring connectivity in new developments
                """)
            
            else:  # Resilience is highest priority
                st.write("""
                ### Resilience-Focused Strategy
                
                Based on your priorities, the recommended strategy focuses on network resilience:
                
                1. **Short-term (0-2 years):**
                   - Identify and protect critical infrastructure points
                   - Develop emergency response routes
                   - Improve maintenance at articulation points
                
                2. **Medium-term (2-5 years):**
                   - Add redundant connections around critical network points
                   - Implement distributed network management systems
                   - Strengthen infrastructure at high-centrality nodes
                
                3. **Long-term (5+ years):**
                   - Develop fully redundant network structure
                   - Create distributed service centers to reduce network dependence
                   - Implement comprehensive disaster response network
                """)
