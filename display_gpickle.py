import networkx as nx
from graphviz import Digraph

def display_graph(input_file, output_file):
    # Load the graph from the .gpickle file
    G = nx.read_gpickle(input_file)
    
    # Initialize a directed graph using Graphviz
    graph = Digraph()
    
    # Add nodes and edges to the Graphviz Digraph
    for src, dst in G.edges():
        graph.edge(str(src), str(dst))  # Ensure both src and dst are specified
    
    # Render the output file
    graph.render(output_file, format='png')
    print(f"Graph saved to {output_file}_number.png")

def display_graph_with_context(input_file, output_file):
    # Load the graph from the .gpickle file
    G = nx.read_gpickle(input_file)
    
    # Initialize a directed graph using Graphviz
    graph = Digraph()
    
    # Add nodes with context (if available) to the Graphviz Digraph
    for node, attributes in G.nodes(data=True):
        label = str(node)
        # Check if there are attributes, add them as labels if they exist
        if attributes:
            # Example: add attributes as a list on the label
            label += "\n" + "\n".join([f"{key}: {value}" for key, value in attributes.items()])
        graph.node(str(node), label=label)
    
    # Add edges with context (if available) to the Graphviz Digraph
    for src, dst, edge_attributes in G.edges(data=True):
        label = ""
        # Check if there are edge attributes, add them as edge labels if they exist
        if edge_attributes:
            # Example: add edge attributes as a list on the edge label
            label = "\n".join([f"{key}: {value}" for key, value in edge_attributes.items()])
        graph.edge(str(src), str(dst), label=label)
    
    # Render the output file
    graph.render(output_file, format='png')
    print(f"Graph with context saved to {output_file}_contexted.png")

# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Display a graph from a .gpickle file.")
    parser.add_argument("--input", required=True, help="Input .gpickle file path")
    parser.add_argument("--output", required=True, help="Output file name (without extension)")
    
    args = parser.parse_args()
    display_graph(args.input, args.output+"_numbered")
    display_graph_with_context(args.input, args.output+"_contexted")
