from graphviz import Digraph

def visualize_tree_graphviz(node, vocab, graph=None, parent_id=None, node_id_counter=None, label_map=None):
    """
    Recursively traverse the tree and add nodes and edges to the Graphviz Digraph.

    Args:
        node (Node): The current node to visualize.
        vocab (list): List of vocabulary words.
        graph (Digraph, optional): The Graphviz Digraph object. Defaults to None.
        parent_id (int, optional): The ID of the parent node. Defaults to None.
        node_id_counter (list, optional): A single-element list acting as a mutable counter for node IDs. Defaults to None.
        label_map (dict, optional): Mapping from node IDs to labels. Defaults to None.

    Returns:
        tuple: (graph, current_node_id, label_map)
    """
    if graph is None:
        graph = Digraph(comment='nCRP Tree')
        graph.attr('node', shape='box', style='filled', color='lightblue')
        label_map = {}
    
    if node_id_counter is None:
        node_id_counter = [0]  # Initialize counter
    
    current_id = node_id_counter[0]
    
    # Create a label for the current node based on top words
    top_words = sorted(node.word_counts.keys(), key=lambda w: node.word_counts[w], reverse=True)[:3]
    label = f"Level {node.level}\nDocs: {node.documents}\nWords: {', '.join(top_words)}"
    label_map[current_id] = label
    graph.node(str(current_id), label=label)
    
    # Add edge from parent to current node
    if parent_id is not None:
        graph.edge(str(parent_id), str(current_id))
    
    # Traverse children
    for child_topic_id, child_node in node.children.items():
        node_id_counter[0] += 1  # Increment counter for the child
        child_id = node_id_counter[0]
        graph, node_id_counter, label_map = visualize_tree_graphviz(
            child_node, vocab, graph, parent_id=current_id, node_id_counter=node_id_counter, label_map=label_map
        )
    
    return graph, node_id_counter, label_map

def print_tree_graphviz(root, vocab, filename='ncrp_tree', view=False):
    """
    Generate and render the tree visualization using Graphviz.

    Args:
        root (Node): The root node of the tree.
        vocab (list): List of vocabulary words.
        filename (str, optional): Filename for the output. Defaults to 'ncrp_tree'.
        view (bool, optional): Whether to automatically open the visualization. Defaults to False.
    """
    graph, _, _ = visualize_tree_graphviz(root, vocab)
    graph.render(filename, view=view, format='png')
    print(f"Tree visualization saved as {filename}.png")
    
def visualize_document_path(tree, doc_id, vocab, filename='doc_path', view=False, top_n=5):
    """
    Visualize the path taken by a particular document through the hierarchical tree.

    Args:
        tree (nCRPTree): The hierarchical LDA tree.
        doc_id (int): ID of the document whose path we want to visualize.
        vocab (list): The vocabulary list.
        filename (str, optional): The output filename (without extension) for the graph. Defaults to 'doc_path'.
        view (bool, optional): Whether to open the generated file after creation. Defaults to False.
        top_n (int, optional): Number of top words to display per node. Defaults to 5.

    Returns:
        None: Saves a .png visualization of the document's path.
    """
    if doc_id not in tree.paths:
        print(f"Document {doc_id} not found in the tree.")
        return

    path_nodes = tree.paths[doc_id]

    # Create a new Graphviz graph
    graph = Digraph(comment=f'Document {doc_id} Path')
    graph.attr('node', shape='box', style='filled', color='lightblue')

    # Function to get top words
    def get_top_words(node, vocab, top_n):
        word_counts = list(node.word_counts.items())
        word_counts.sort(key=lambda x: x[1], reverse=True)
        top_words = [w for w, count in word_counts[:top_n]]
        return top_words

    # Add nodes for each node in the path
    node_ids = []
    for i, node in enumerate(path_nodes):
        tw = get_top_words(node, vocab, top_n)
        label = f"Level {node.level}\nDocs: {node.documents}\nTop words: {', '.join(tw)}"
        node_id = f"{doc_id}_{i}"
        graph.node(node_id, label=label)
        node_ids.append(node_id)

    # Add edges to represent the path
    for i in range(len(node_ids)-1):
        graph.edge(node_ids[i], node_ids[i+1])

    # Render the graph
    graph.render(filename, view=view, format='png')
    print(f"Document {doc_id} path visualization saved as {filename}.png")