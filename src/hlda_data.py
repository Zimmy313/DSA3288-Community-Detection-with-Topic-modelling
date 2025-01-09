import numpy as np

def generate_synthetic_corpus(
    hlda_model,
    num_docs=100,
    doc_length=250,
    seed=42
):
    """
    Generates a synthetic corpus by sampling documents from the hierarchical
    topic structure learned by 'hlda_model,' following Blei's approach.

    Specifically:
    1. Randomly select an existing document (i.e., doc index) from the model.
    2. Retrieve its path (root -> leaf).
    3. Partition 'doc_length' among the levels using a Dirichlet(alpha_levels).
    4. Sample tokens from each level's distribution, concatenating to form one doc.

    Parameters
    ----------
    hlda_model : HierarchicalLDA
        A trained HLDA model instance, which has:
         - doc_to_leaf[d] : leaf node for doc d
         - get_path_from_leaf(...) : retrieves path from root->leaf
    num_docs : int
        How many synthetic documents to generate in total.
    doc_length : int
        Number of tokens (words) per synthetic document.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    synthetic_corpus : list of list
        synthetic_corpus[i] is a synthetic document (list of tokens).
    doc_paths : list of list of int
        doc_paths[i] is the list of node indices along the path used to 
        generate synthetic_corpus[i].
    """
    
    rng = np.random.default_rng(seed)
    alpha_levels = hlda_model.alpha
    n_original_docs = hlda_model.num_docs  # how many docs the model originally had

    synthetic_corpus = []
    doc_paths = []

    for _ in range(num_docs):
        # 1) Randomly pick an existing doc index from the trained model
        original_doc_idx = rng.integers(0, n_original_docs)

        # 2) Retrieve the leaf node, then the path from root -> leaf
        leaf_node = hlda_model.doc_to_leaf[original_doc_idx]
        path_nodes = hlda_model.get_path_from_leaf(leaf_node)

        # 3) Partition doc_length among levels (Blei-style) via Dirichlet
        L = len(path_nodes)
        if isinstance(alpha_levels, (int, float)):
            alpha_vec = [alpha_levels]*L
        else:
            alpha_vec = alpha_levels

        level_props = rng.dirichlet(alpha_vec)
        level_counts = [int(round(p * doc_length)) for p in level_props]

        # fix rounding
        diff = doc_length - sum(level_counts)
        while diff != 0:
            idx = rng.integers(0, L)
            if diff > 0:
                level_counts[idx] += 1
                diff -= 1
            else:  # diff < 0
                if level_counts[idx] > 0:
                    level_counts[idx] -= 1
                    diff += 1

        # 4) Sample tokens from each node’s distribution
        doc_tokens = []
        for lvl, node in enumerate(path_nodes):
            words = node.vocabulary
            freqs = node.word_freqs
            total_freq = float(node.word_total)

            if total_freq <= 0:
                # fallback to uniform distribution
                probs = np.ones(len(words)) / len(words)
            else:
                probs = freqs / total_freq

            chosen = rng.choice(words, size=level_counts[lvl], replace=True, p=probs)
            doc_tokens.extend(chosen)

        synthetic_corpus.append(doc_tokens)
        doc_paths.append([node.node_index for node in path_nodes])

    return synthetic_corpus, doc_paths