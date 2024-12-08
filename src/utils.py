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
    
def get_top_words(node, vocab, top_n=5):
    word_counts = list(node.word_counts.items())
    word_counts.sort(key=lambda x: x[1], reverse=True)
    top_words = [w for w, count in word_counts[:top_n]]
    return top_words

def print_tree(node, vocab, level=0):
    indent = "  " * level
    print(f"{indent}Level {node.level}: docs={node.documents}, total_words={node.total_words}")
    top_words = get_top_words(node, vocab, top_n=5)
    print(f"{indent}  Top words: {top_words}")
    for child_id, child_node in node.children.items():
        print_tree(child_node, vocab, level+1)

def print_document_assignments(tree, doc_id):
    doc_words = tree.document_words[doc_id]
    doc_levels = tree.levels[doc_id]
    print(f"Document {doc_id}:")
    for w, lvl in zip(doc_words, doc_levels):
        print(f"  {w} -> level {lvl}")