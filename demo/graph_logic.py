"""
Pied Piper AI - Graph Logic
The "brain" of the fraud detection system.

This module constructs a network graph from insurance claims data and uses
graph algorithms to detect fraud patterns (simulating GNN output for demo purposes).

Detection Strategy:
- Community Detection: Identifies tightly-knit groups (fraud rings)
- Degree Centrality: Finds highly connected nodes (fraud hubs)
- Risk Scoring: Combines metrics to assign suspicion levels
"""

import networkx as nx
import pandas as pd
from collections import defaultdict
import community as community_louvain  # python-louvain package


def build_graph(nodes_df, edges_df):
    """
    Construct a NetworkX graph from nodes and edges DataFrames.
    
    Args:
        nodes_df: DataFrame with columns ['id', 'label', 'type', 'is_fraud']
        edges_df: DataFrame with columns ['from', 'to', 'amount', 'type', 'is_fraud']
        
    Returns:
        networkx.Graph: Constructed graph with node and edge attributes
    """
    G = nx.Graph()
    
    # Add nodes with attributes
    for _, node in nodes_df.iterrows():
        G.add_node(
            node['id'],
            label=node['label'],
            node_type=node['type'],
            is_fraud=node['is_fraud'],
            title=f"{node['type']}: {node['label']}"  # For PyVis hover tooltip
        )
    
    # Add edges with attributes
    for _, edge in edges_df.iterrows():
        # If edge already exists, increment weight; otherwise create new edge
        if G.has_edge(edge['from'], edge['to']):
            G[edge['from']][edge['to']]['weight'] += 1
            G[edge['from']][edge['to']]['total_amount'] += edge['amount']
        else:
            G.add_edge(
                edge['from'],
                edge['to'],
                weight=1,
                total_amount=edge['amount'],
                edge_type=edge['type']
            )
    
    return G


def detect_communities(G):
    """
    Detect communities (fraud rings) using Louvain algorithm.
    
    The Louvain method identifies densely connected groups, which in our case
    often correspond to fraud rings where multiple entities are colluding.
    
    Args:
        G: NetworkX graph
        
    Returns:
        dict: Mapping of node_id -> community_id
    """
    # Apply Louvain community detection
    partition = community_louvain.best_partition(G, weight='weight')
    return partition


def calculate_risk_scores(G, communities):
    """
    Calculate risk scores for each node based on graph metrics.
    
    This simulates the output of a GNN model by combining multiple graph features:
    - Degree Centrality: How connected is the node?
    - Community Size: Is the node in a tight-knit group?
    - Edge Weight: Does the node have repeated interactions?
    - Clustering Coefficient: Is the node part of closed triangles?
    
    Args:
        G: NetworkX graph
        communities: Dict mapping node_id -> community_id
        
    Returns:
        dict: Mapping of node_id -> risk_score (0-100)
    """
    risk_scores = {}
    
    # Calculate graph metrics
    degree_centrality = nx.degree_centrality(G)
    clustering_coeff = nx.clustering(G, weight='weight')
    
    # Calculate community sizes
    community_sizes = defaultdict(int)
    for node, comm_id in communities.items():
        community_sizes[comm_id] += 1
    
    # Normalize factors
    max_degree = max(degree_centrality.values()) if degree_centrality else 1
    max_clustering = max(clustering_coeff.values()) if clustering_coeff else 1
    
    for node in G.nodes():
        # Factor 1: Degree centrality (normalized)
        degree_score = degree_centrality.get(node, 0) / max_degree if max_degree > 0 else 0
        
        # Factor 2: Clustering coefficient (high clustering = closed groups)
        cluster_score = clustering_coeff.get(node, 0) / max_clustering if max_clustering > 0 else 0
        
        # Factor 3: Community size (small tight communities are suspicious)
        comm_id = communities.get(node, -1)
        comm_size = community_sizes.get(comm_id, 0)
        # Suspicious if community is small (5-30 nodes) and tight
        if 5 <= comm_size <= 30:
            community_score = 1.0
        elif comm_size < 5:
            community_score = 0.5
        else:
            community_score = 0.2
        
        # Factor 4: Edge weight (repeated interactions)
        neighbors = list(G.neighbors(node))
        avg_edge_weight = 0
        if neighbors:
            total_weight = sum(G[node][neighbor].get('weight', 1) for neighbor in neighbors)
            avg_edge_weight = total_weight / len(neighbors)
        
        weight_score = min(avg_edge_weight / 5.0, 1.0)  # Normalize, cap at 1.0
        
        # Combine scores with weights (tuned for fraud detection)
        risk_score = (
            degree_score * 30 +        # High connections = 30% weight
            cluster_score * 25 +        # Tight clustering = 25% weight
            community_score * 30 +      # Suspicious community size = 30% weight
            weight_score * 15           # Repeated interactions = 15% weight
        )
        
        risk_scores[node] = min(risk_score, 100)  # Cap at 100
    
    return risk_scores


def identify_fraud_rings(G, communities, risk_scores, threshold=50):
    """
    Identify fraud rings by analyzing communities with high-risk nodes.
    
    Args:
        G: NetworkX graph
        communities: Dict mapping node_id -> community_id
        risk_scores: Dict mapping node_id -> risk_score
        threshold: Minimum risk score to consider a node suspicious
        
    Returns:
        dict: Information about detected fraud rings
    """
    # Find high-risk nodes
    suspicious_nodes = {node: score for node, score in risk_scores.items() if score >= threshold}
    
    # Group suspicious nodes by community
    fraud_communities = defaultdict(list)
    for node in suspicious_nodes:
        comm_id = communities.get(node, -1)
        fraud_communities[comm_id].append(node)
    
    # Analyze each potential fraud ring
    fraud_rings = []
    for comm_id, nodes in fraud_communities.items():
        if len(nodes) >= 3:  # At least 3 suspicious nodes to be considered a ring
            # Calculate ring statistics
            ring_nodes = nodes
            avg_risk = sum(risk_scores[n] for n in ring_nodes) / len(ring_nodes)
            
            # Identify hub nodes (highest degree)
            node_degrees = {n: G.degree(n) for n in ring_nodes}
            hub = max(node_degrees, key=node_degrees.get)
            
            # Detect pattern type
            pattern_type = detect_pattern_type(G, ring_nodes, hub)
            
            fraud_rings.append({
                'ring_id': len(fraud_rings) + 1,
                'community_id': comm_id,
                'nodes': ring_nodes,
                'size': len(ring_nodes),
                'avg_risk_score': avg_risk,
                'hub_node': hub,
                'pattern_type': pattern_type
            })
    
    return {
        'suspicious_nodes': suspicious_nodes,
        'fraud_rings': fraud_rings,
        'total_fraud_rings': len(fraud_rings)
    }


def detect_pattern_type(G, nodes, hub):
    """
    Detect the type of fraud pattern based on graph topology.
    
    Args:
        G: NetworkX graph
        nodes: List of nodes in the potential fraud ring
        hub: The hub node (highest degree)
        
    Returns:
        str: Pattern type description
    """
    # Check for star topology (hub-and-spoke pattern)
    hub_degree = G.degree(hub)
    other_degrees = [G.degree(n) for n in nodes if n != hub]
    avg_other_degree = sum(other_degrees) / len(other_degrees) if other_degrees else 0
    
    if hub_degree > avg_other_degree * 2:
        return "Star Topology (Hub-based Collusion)"
    
    # Check for circular pattern
    subgraph = G.subgraph(nodes)
    cycles = list(nx.simple_cycles(subgraph.to_directed()))
    if len(cycles) > 0:
        return "Circular Pattern (Staged Loop)"
    
    # Check for dense clique
    density = nx.density(subgraph)
    if density > 0.7:
        return "Dense Clique (Mutual Collusion)"
    
    return "Complex Network (Multi-party Fraud)"


def analyze_graph(nodes_df, edges_df):
    """
    Main analysis function: Build graph and detect fraud.
    
    This is the primary entry point for the fraud detection system.
    
    Args:
        nodes_df: DataFrame with entity nodes
        edges_df: DataFrame with claim edges
        
    Returns:
        dict: Complete analysis results including graph, communities, and fraud detection
    """
    print("[*] Building network graph...")
    G = build_graph(nodes_df, edges_df)
    
    print("[*] Detecting communities...")
    communities = detect_communities(G)
    
    print("[*] Calculating risk scores...")
    risk_scores = calculate_risk_scores(G, communities)
    
    print("[*] Identifying fraud rings...")
    fraud_detection = identify_fraud_rings(G, communities, risk_scores)
    
    print(f"[SUCCESS] Analysis complete! Found {fraud_detection['total_fraud_rings']} fraud rings.\n")
    
    return {
        'graph': G,
        'communities': communities,
        'risk_scores': risk_scores,
        'fraud_detection': fraud_detection
    }


# For testing purposes
if __name__ == "__main__":
    from data_generator import generate_all_data
    
    print("Testing graph logic with generated data...\n")
    nodes_df, edges_df = generate_all_data()
    
    results = analyze_graph(nodes_df, edges_df)
    
    print(f"Graph Statistics:")
    print(f"  Nodes: {results['graph'].number_of_nodes()}")
    print(f"  Edges: {results['graph'].number_of_edges()}")
    print(f"  Communities: {len(set(results['communities'].values()))}")
    print(f"  Suspicious Nodes: {len(results['fraud_detection']['suspicious_nodes'])}")
    print(f"\nDetected Fraud Rings:")
    for ring in results['fraud_detection']['fraud_rings']:
        print(f"  Ring #{ring['ring_id']}: {ring['size']} nodes, Pattern: {ring['pattern_type']}")
