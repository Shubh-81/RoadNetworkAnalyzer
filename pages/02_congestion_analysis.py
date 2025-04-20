import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from collections import Counter

from utils.network_analysis import identify_congestion_hotspots, analyze_community_structure
from utils.visualization import plot_network, plot_centrality_distribution, plot_community_structure
from utils.routing import identify_traffic_bottlenecks, simulate_traffic_flow

# Configure page
st.set_page_config(
    page_title="Congestion Analysis | California Road Network Analysis",
    page_icon="🚗",
    layout="wide"
)

# Title
st.title("Congestion Analysis")
st.write("Identify and analyze potential traffic congestion hotspots in the California road network")

# Check if network is loaded
if 'network' not in st.session_state or st.session_state.network is None:
    st.warning("Please load the California road network from the home page first")
    st.stop()

# Get the network to analyze
network_to_analyze = st.session_state.selected_region if st.session_state.selected_region is not None else st.session_state.network

# Add tabs for different analyses
tab1, tab2, tab3 = st.tabs(["Congestion Hotspots", "Community Analysis", "Traffic Simulation"])

with tab1:
    st.subheader("Identify Congestion Hotspots")
    st.write("""
    Congestion hotspots are intersections where traffic is likely to build up due to their 
    structural position in the network. These can be identified using different centrality measures.
    """)
    
    # Select method for hotspot identification
    method = st.selectbox(
        "Select method for identifying hotspots",
        ["Betweenness Centrality", "Degree Centrality", "Closeness Centrality", "Edge Betweenness"]
    )
    
    # Number of hotspots to identify
    num_hotspots = st.slider("Number of hotspots to identify", 5, 50, 20)
    
    # Compute hotspots
    with st.spinner(f"Identifying congestion hotspots using {method}..."):
        if method == "Betweenness Centrality":
            hotspots_df = identify_congestion_hotspots(network_to_analyze, method="betweenness", top_n=num_hotspots)
            metric_col = "Betweenness Centrality"
        
        elif method == "Degree Centrality":
            hotspots_df = identify_congestion_hotspots(network_to_analyze, method="degree", top_n=num_hotspots)
            metric_col = "Degree"
        
        elif method == "Closeness Centrality":
            hotspots_df = identify_congestion_hotspots(network_to_analyze, method="closeness", top_n=num_hotspots)
            metric_col = "Closeness Centrality"
        
        elif method == "Edge Betweenness":
            # Identify bottleneck edges
            bottlenecks = identify_traffic_bottlenecks(
                network_to_analyze, 
                num_bottlenecks=num_hotspots,
                method="edge_betweenness"
            )
            
            # Create a dataframe for visualization
            edge_data = []
            for edge, centrality in bottlenecks:
                edge_data.append({
                    "Edge": f"{edge[0]}-{edge[1]}",
                    "Node1": edge[0],
                    "Node2": edge[1],
                    "Edge Betweenness": centrality
                })
            hotspots_df = pd.DataFrame(edge_data)
            metric_col = "Edge Betweenness"
    
    # Display hotspots
    if method != "Edge Betweenness":
        st.subheader(f"Top {num_hotspots} Congestion Hotspots")
        
        # Show table of hotspots
        st.dataframe(hotspots_df)
        
        # Visualize hotspots
        st.subheader("Hotspot Visualization")
        hotspot_nodes = hotspots_df["Node"].tolist()
        plot_network(network_to_analyze, highlight_nodes=hotspot_nodes)
        
        # Bar chart of hotspots
        fig = px.bar(hotspots_df, x="Node", y=metric_col, 
                     title=f"Top {num_hotspots} Hotspots by {metric_col}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.subheader(f"Top {num_hotspots} Bottleneck Edges")
        
        # Show table of bottleneck edges
        st.dataframe(hotspots_df)
        
        # Bar chart of bottleneck edges
        fig = px.bar(hotspots_df, x="Edge", y=metric_col, 
                     title=f"Top {num_hotspots} Bottleneck Edges by {metric_col}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Collect the nodes involved in bottleneck edges for visualization
        bottleneck_nodes = set()
        for _, row in hotspots_df.iterrows():
            bottleneck_nodes.add(row["Node1"])
            bottleneck_nodes.add(row["Node2"])
        
        # Visualize the bottleneck edges
        st.subheader("Bottleneck Edge Visualization")
        plot_network(network_to_analyze, highlight_nodes=list(bottleneck_nodes))
    
    # Hotspot analysis
    st.subheader("Hotspot Analysis")
    
    # For node-based hotspots
    if method != "Edge Betweenness":
        # Analyze degree distribution of hotspots
        hotspot_degrees = []
        for node in hotspots_df["Node"]:
            if node in network_to_analyze:
                hotspot_degrees.append(network_to_analyze.degree(node))
        
        if hotspot_degrees:
            avg_hotspot_degree = np.mean(hotspot_degrees)
            avg_network_degree = np.mean([d for _, d in network_to_analyze.degree()])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Degree of Hotspots", f"{avg_hotspot_degree:.2f}")
            with col2:
                st.metric("Average Degree of Network", f"{avg_network_degree:.2f}")
            
            st.write(f"Hotspots have {avg_hotspot_degree/avg_network_degree:.1f}x the average degree of the network")
    
    # For edge-based hotspots
    else:
        # Analyze nodes connected by bottleneck edges
        bottleneck_node_degrees = []
        for _, row in hotspots_df.iterrows():
            node1, node2 = row["Node1"], row["Node2"]
            if node1 in network_to_analyze and node2 in network_to_analyze:
                bottleneck_node_degrees.append(network_to_analyze.degree(node1))
                bottleneck_node_degrees.append(network_to_analyze.degree(node2))
        
        if bottleneck_node_degrees:
            avg_bottleneck_node_degree = np.mean(bottleneck_node_degrees)
            avg_network_degree = np.mean([d for _, d in network_to_analyze.degree()])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Avg Degree of Nodes at Bottlenecks", f"{avg_bottleneck_node_degree:.2f}")
            with col2:
                st.metric("Average Degree of Network", f"{avg_network_degree:.2f}")
    
    # Export options
    if st.checkbox("Export hotspot data"):
        csv = hotspots_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"congestion_hotspots_{method.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )

with tab2:
    st.subheader("Community Structure Analysis")
    st.write("""
    Communities in the road network can represent areas with high internal connectivity but fewer 
    connections between them. Boundary areas between communities can be potential congestion points.
    """)
    
    # Select community detection method
    comm_method = st.selectbox(
        "Select community detection method",
        ["Louvain", "Label Propagation"]
    )
    
    # Run community detection
    if st.button("Detect Communities"):
        with st.spinner(f"Detecting communities using {comm_method} method..."):
            try:
                method_str = "louvain" if comm_method == "Louvain" else "label_propagation"
                community_results = analyze_community_structure(network_to_analyze, method=method_str)
                
                if "error" in community_results:
                    st.error(f"Error in community detection: {community_results['error']}")
                else:
                    # Display community statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Number of Communities", community_results["num_communities"])
                    with col2:
                        st.metric("Average Community Size", f"{community_results['avg_community_size']:.1f}")
                    with col3:
                        st.metric("Maximum Community Size", community_results["max_community_size"])
                    
                    # Visualize communities
                    st.subheader("Community Visualization")
                    plot_community_structure(network_to_analyze, community_results["partition"])
                    
                    # Identify boundary nodes (nodes with neighbors in different communities)
                    st.subheader("Boundary Nodes Analysis")
                    st.write("""
                    Boundary nodes connect different communities and can be potential congestion points
                    where traffic flows between different areas of the network.
                    """)
                    
                    boundary_nodes = []
                    
                    with st.spinner("Identifying boundary nodes..."):
                        partition = community_results["partition"]
                        for node in network_to_analyze.nodes():
                            if node in partition:
                                node_comm = partition[node]
                                has_diff_comm = False
                                
                                # Check if any neighbor is in a different community
                                for neighbor in network_to_analyze.neighbors(node):
                                    if neighbor in partition and partition[neighbor] != node_comm:
                                        has_diff_comm = True
                                        break
                                
                                if has_diff_comm:
                                    boundary_nodes.append({
                                        "Node": node,
                                        "Community": node_comm,
                                        "Degree": network_to_analyze.degree(node)
                                    })
                    
                    # Display boundary nodes
                    if boundary_nodes:
                        boundary_df = pd.DataFrame(boundary_nodes)
                        boundary_df = boundary_df.sort_values("Degree", ascending=False)
                        
                        st.write(f"Found {len(boundary_nodes)} boundary nodes connecting different communities")
                        st.dataframe(boundary_df.head(20))
                        
                        # Visualize top boundary nodes
                        if st.checkbox("Visualize top boundary nodes"):
                            top_boundary = boundary_df.head(20)["Node"].tolist()
                            plot_network(network_to_analyze, highlight_nodes=top_boundary)
                        
                        # Community connections heatmap
                        st.subheader("Community Connections")
                        
                        # Count connections between communities
                        comm_connections = Counter()
                        
                        with st.spinner("Analyzing inter-community connections..."):
                            for u, v in network_to_analyze.edges():
                                if u in partition and v in partition:
                                    comm_u = partition[u]
                                    comm_v = partition[v]
                                    if comm_u != comm_v:
                                        # Sort to ensure consistent counting
                                        key = tuple(sorted([comm_u, comm_v]))
                                        comm_connections[key] += 1
                        
                        # Convert to dataframe for visualization
                        if comm_connections:
                            connections_data = []
                            for (comm1, comm2), count in comm_connections.most_common(20):
                                connections_data.append({
                                    "Community Pair": f"{comm1}-{comm2}",
                                    "Community 1": comm1,
                                    "Community 2": comm2,
                                    "Connection Count": count
                                })
                            
                            connections_df = pd.DataFrame(connections_data)
                            
                            # Bar chart of community connections
                            fig = px.bar(connections_df, x="Community Pair", y="Connection Count",
                                         title="Top 20 Community Connections")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No connections between different communities found")
                    else:
                        st.info("No boundary nodes found")
            except Exception as e:
                st.error(f"Error in community analysis: {str(e)}")

with tab3:
    st.subheader("Traffic Simulation")
    st.write("""
    Simulate traffic flow through the network to identify potential congestion points under 
    different traffic scenarios.
    """)
    
    # Select traffic simulation parameters
    st.subheader("Simulation Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        # For simplicity, use high degree nodes as traffic sources
        num_sources = st.slider("Number of traffic sources", 1, 20, 5)
        # Use random nodes as traffic sinks
        num_sinks = st.slider("Number of traffic sinks", 1, 20, 5)
    
    with col2:
        # Traffic flow amount
        flow_amount = st.slider("Traffic flow amount", 100, 1000, 500)
        
        # Select traffic pattern
        traffic_pattern = st.selectbox(
            "Traffic pattern",
            ["High degree sources to random sinks", "Random sources to high degree sinks", "Random to random"]
        )
    
    # Run simulation
    if st.button("Run Traffic Simulation"):
        with st.spinner("Simulating traffic flow..."):
            # Identify sources and sinks based on selected pattern
            if traffic_pattern == "High degree sources to random sinks":
                # Use high degree nodes as sources
                degrees = dict(network_to_analyze.degree())
                traffic_sources = sorted(degrees, key=degrees.get, reverse=True)[:num_sources]
                
                # Use random nodes as sinks
                all_nodes = list(network_to_analyze.nodes())
                traffic_sinks = np.random.choice(all_nodes, size=num_sinks, replace=False)
                
            elif traffic_pattern == "Random sources to high degree sinks":
                # Use random nodes as sources
                all_nodes = list(network_to_analyze.nodes())
                traffic_sources = np.random.choice(all_nodes, size=num_sources, replace=False)
                
                # Use high degree nodes as sinks
                degrees = dict(network_to_analyze.degree())
                traffic_sinks = sorted(degrees, key=degrees.get, reverse=True)[:num_sinks]
                
            else:  # Random to random
                # Use random nodes as both sources and sinks
                all_nodes = list(network_to_analyze.nodes())
                traffic_sources = np.random.choice(all_nodes, size=num_sources, replace=False)
                traffic_sinks = np.random.choice(all_nodes, size=num_sinks, replace=False)
            
            # Run the simulation
            try:
                sim_results = simulate_traffic_flow(
                    network_to_analyze,
                    traffic_sources,
                    traffic_sinks,
                    flow_amount
                )
                
                # Display simulation results
                st.subheader("Simulation Results")
                
                # Traffic sources and sinks
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Traffic Sources:")
                    source_data = []
                    for source in traffic_sources:
                        source_data.append({
                            "Node": source,
                            "Degree": network_to_analyze.degree(source)
                        })
                    st.dataframe(pd.DataFrame(source_data))
                
                with col2:
                    st.write("Traffic Sinks:")
                    sink_data = []
                    for sink in traffic_sinks:
                        sink_data.append({
                            "Node": sink,
                            "Degree": network_to_analyze.degree(sink)
                        })
                    st.dataframe(pd.DataFrame(sink_data))
                
                # Congested nodes
                st.subheader("Most Congested Nodes")
                congested_node_data = []
                for node, flow in sim_results["congested_nodes"]:
                    congested_node_data.append({
                        "Node": node,
                        "Traffic Flow": flow,
                        "Degree": network_to_analyze.degree(node)
                    })
                
                congested_nodes_df = pd.DataFrame(congested_node_data)
                st.dataframe(congested_nodes_df)
                
                # Visualize congested nodes
                if st.checkbox("Visualize congested nodes"):
                    congested_node_ids = congested_nodes_df["Node"].tolist()[:10]  # Top 10
                    plot_network(network_to_analyze, highlight_nodes=congested_node_ids)
                
                # Bar chart of congested nodes
                fig = px.bar(congested_nodes_df.head(20), x="Node", y="Traffic Flow",
                             title="Top 20 Most Congested Nodes")
                st.plotly_chart(fig, use_container_width=True)
                
                # Congested edges
                st.subheader("Most Congested Road Segments")
                congested_edge_data = []
                for edge, flow in sim_results["congested_edges"]:
                    congested_edge_data.append({
                        "Edge": f"{edge[0]}-{edge[1]}",
                        "Node1": edge[0],
                        "Node2": edge[1],
                        "Traffic Flow": flow
                    })
                
                congested_edges_df = pd.DataFrame(congested_edge_data)
                st.dataframe(congested_edges_df)
                
                # Bar chart of congested edges
                fig = px.bar(congested_edges_df.head(20), x="Edge", y="Traffic Flow",
                             title="Top 20 Most Congested Road Segments")
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations based on simulation
                st.subheader("Traffic Management Recommendations")
                
                # Compare congested nodes with high betweenness
                with st.spinner("Generating recommendations..."):
                    # Calculate betweenness for a sample
                    sample_size = min(100, network_to_analyze.number_of_nodes())
                    betweenness = nx.betweenness_centrality(network_to_analyze, k=sample_size)
                    
                    # Get betweenness for congested nodes
                    congested_ids = congested_nodes_df["Node"].tolist()[:10]  # Top 10
                    congested_betweenness = {node: betweenness.get(node, 0) for node in congested_ids}
                    
                    # Generate recommendations
                    recommendations = []
                    
                    # Find nodes that are highly congested but have lower betweenness
                    for node, flow in sim_results["congested_nodes"][:10]:
                        node_degree = network_to_analyze.degree(node)
                        node_betweenness = betweenness.get(node, 0)
                        
                        if node_betweenness < 0.01 and node_degree > 2:
                            recommendations.append({
                                "Node": node,
                                "Recommendation": "Consider adding capacity to this intersection. " +
                                                "It handles high traffic but isn't a critical network hub.",
                                "Traffic Flow": flow,
                                "Betweenness": node_betweenness,
                                "Degree": node_degree
                            })
                        elif node_betweenness > 0.05:
                            recommendations.append({
                                "Node": node,
                                "Recommendation": "Critical network hub. Consider traffic management and alternate " +
                                                "routes to reduce reliance on this intersection.",
                                "Traffic Flow": flow,
                                "Betweenness": node_betweenness,
                                "Degree": node_degree
                            })
                    
                    # Display recommendations
                    if recommendations:
                        recommendations_df = pd.DataFrame(recommendations)
                        st.dataframe(recommendations_df)
                    else:
                        st.info("No specific recommendations generated for this simulation")
                
            except Exception as e:
                st.error(f"Error in traffic simulation: {str(e)}")
