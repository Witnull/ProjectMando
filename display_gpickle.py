import networkx as nx
from graphviz import Digraph
MAX = 100 #limit nodes

def display_graph(input_file, output_file):
    # Load the graph from the .gpickle file
    G = nx.read_gpickle(input_file)
    
    # Initialize a directed graph using Graphviz
    graph = Digraph(format='png')
    
    # Limit to MAX nodes
    nodes = list(G.nodes())[:MAX]
    subgraph = G.subgraph(nodes)
    
    # Add nodes and edges to the Graphviz Digraph
    for src, dst in subgraph.edges():
        graph.edge(str(src), str(dst))
    
    # Render the output file
    graph.render(output_file)
    print(f"Graph saved to {output_file}.png (limited to {MAX} nodes)")

def display_graph_with_context(input_file, output_file):
    # Load the graph from the .gpickle file
    G = nx.read_gpickle(input_file)
    
    # Initialize a directed graph using Graphviz
    graph = Digraph(format='png')
    
    # Limit to MAX nodes
    nodes = list(G.nodes())[:MAX]
    subgraph = G.subgraph(nodes)
    
    # Add nodes with context (if available) to the Graphviz Digraph
    for node, attributes in subgraph.nodes(data=True):
        label = str(node)
        if attributes:
            label += "\n" + "\n".join([f"{key}: {value}" for key, value in attributes.items()])#([attributes["function_fullname"],attributes["contract_name"]])
        graph.node(str(node), label=label)
    
    # Add edges with context (if available) to the Graphviz Digraph
    for src, dst, edge_attributes in subgraph.edges(data=True):
        label = ""
        if edge_attributes:
            label = "\n".join([f"{key}: {value}" for key, value in edge_attributes.items()])
        graph.edge(str(src), str(dst), label=label)
    
    # Render the output file
    graph.render(output_file)
    print(f"Graph with context saved to {output_file}.png (limited to {MAX} nodes)")

# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Display a graph from a .gpickle file.")
    parser.add_argument("--input", required=True, help="Input .gpickle file path")
    parser.add_argument("--output", required=True, help="Output file name (without extension)")
    
    args = parser.parse_args()
    display_graph(args.input, args.output+"_numbered")
    display_graph_with_context(args.input, args.output+"_contexted")