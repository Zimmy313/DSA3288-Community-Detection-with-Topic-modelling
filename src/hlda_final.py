import numpy as np
import time
import matplotlib.pyplot as plt
import io
from PIL import Image
from math import log
from numpy.random import RandomState
from graphviz import Digraph


class NCRPNode:
    """
    This class represents a node (topic).
    
    Each node has:
      - n_w       : array of counts of each word w in this node/topic
      - n_sum     : total count of words assigned to this node
      - parent    : link to the parent in the tree
      - children  : a list of child nodes
      - num_docs  : how many documents currently use this node in their path
      - level_id  : which level of the hierarchy this node is at (0 = root)
    """

    total_created_nodes = 0
    highest_node_index = 0

    def __init__(self, L, vocabulary, parent=None, level_id=0, rng=None):
        """
        Parameters
        ----------
        L : int
            The total number of levels in the hierarchy.
        vocabulary : list or array of strings
        parent : NCRPNode or None
        level_id : int
            Which level (0-based) this node is at (root=0).
        rng : np.random.RandomState
        """
        self.node_index = NCRPNode.highest_node_index
        NCRPNode.highest_node_index += 1

        self.num_docs = 0            # number of documents served by this node
        self.parent = parent
        self.children = []
        self.level_id = level_id
        self.hierarchy_depth = L

        # Word-count arrays
        self.vocabulary = np.array(vocabulary)
        self.n_w = np.zeros(len(vocabulary), dtype=int)  # n_{topic,w}
        self.n_sum = 0  # sum_w n_{topic,w}

        if rng is None:
            self.rng = RandomState()
        else:
            self.rng = rng

    def __repr__(self):
        p_idx = None if self.parent is None else self.parent.node_index
        return (f'NCRPNode(idx={self.node_index}, level={self.level_id}, '
                f'docs={self.num_docs}, n_sum={self.n_sum}, parent={p_idx})')

    def is_leaf_level(self):
        return (self.level_id == self.hierarchy_depth - 1)

    def add_child(self):
        """Create a new child node (topic) at the next level down."""
        child = NCRPNode(self.hierarchy_depth, self.vocabulary,
                         parent=self, level_id=self.level_id + 1,
                         rng=self.rng)
        self.children.append(child)
        NCRPNode.total_created_nodes += 1
        return child

    def remove_child(self, child):
        """Remove a child node reference from this node."""
        self.children.remove(child)
        NCRPNode.total_created_nodes -= 1

    def grow_branch_to_leaf(self):
        """
        If this node is not at the leaf level, create children all the way down
        until reaching a leaf, and return that leaf node.
        """
        cursor = self
        while not cursor.is_leaf_level():
            cursor = cursor.add_child()
        return cursor

    def decrement_doc(self):
        """
        Decrement one 'customer' (document) from this node's usage.
        If num_docs goes to zero, prune this node from its parent's children.
        Also propagate up to ancestor nodes.
        """
        cursor = self
        cursor.num_docs -= 1
        if cursor.num_docs == 0 and cursor.parent is not None:
            cursor.parent.remove_child(cursor)
        # Keep moving up the chain for each level
        for _ in range(1, self.hierarchy_depth):
            cursor = cursor.parent
            if cursor is None:
                break
            cursor.num_docs -= 1
            if cursor.num_docs == 0 and cursor.parent is not None:
                cursor.parent.remove_child(cursor)

    def increment_doc(self):
        """
        Add one 'customer' (document) to this node, and propagate upwards.
        """
        cursor = self
        cursor.num_docs += 1
        for _ in range(1, self.hierarchy_depth):
            cursor = cursor.parent
            if cursor is None:
                break
            cursor.num_docs += 1

    def ncrp_draw_child(self, gamma):
        """
        Draw a child according to the nested CRP:
          Probability of picking an existing child node:
              (child.num_docs) / (self.num_docs + gamma)
          Probability of creating a new child:
              gamma / (self.num_docs + gamma)
        """
        
        probs = []

        # Probability of new child
        probs.append(gamma / (self.num_docs + gamma))

        # Probabilities for existing children
        for ch in self.children:
            probs.append(ch.num_docs / (self.num_docs + gamma))

        chosen_idx = self.rng.multinomial(1, probs).argmax()
        if chosen_idx == 0:
            return self.add_child()
        else:
            return self.children[chosen_idx - 1]

    def top_words(self, n_terms=5, show_counts=True):
        """Return the top n words in this topic."""
        order = np.argsort(self.n_w)[::-1]
        best_ids = order[:n_terms]
        best_counts = self.n_w[best_ids]
        best_words = self.vocabulary[best_ids]
        if show_counts:
            return ', '.join(f'{w}({c})' for w, c in zip(best_words, best_counts))
        else:
            return ', '.join(best_words)

##############################################################################
# Hierarchical LDA main class
##############################################################################
class hLDA:
    """
    This class implements hierarchical LDA using the nested CRP,
    following the notation and sampling approach in Blei (2004).
    """

    def __init__(self, corpus, vocabulary, L,
                 alpha=10.0, gamma=1.0, eta=0.1,
                 seed=0,verbose = False):
        """
        Parameters
        ----------
        corpus : list of lists
            Each item is a document, which is a list of integer word IDs.
        vocabulary : list of str
            The mapping from word_id -> actual word string.
        L : int
            Number of levels in the topic hierarchy.
        alpha : float
                - alpha = document-level smoothing (Dirichlet prior)
        gamma : float
            - gamma = nCRP parameter controlling how likely new branches are created
        eta : float
            - eta   = topic-word Dirichlet parameter
        seed : int
        verbose: bool
        """
        # Reset counters
        NCRPNode.total_created_nodes = 1 # This account for the root
        NCRPNode.highest_node_index = 0

        self.documents = corpus
        self.vocab = vocabulary
        self.num_docs = len(corpus)
        self.L = L
        self.alpha = alpha
        self.gamma = gamma
        self.eta = eta

        self.rng = RandomState(seed)

        self.V = len(vocabulary)
        self.eta_sum = self.eta * self.V  # sum_{w} eta if symmetric
        self.verbose = verbose

        # Keep track of each doc's leaf node c_d
        self.doc_leaf = {}
        
        # For each doc, store the level assignment z_{di}
        self.z_tokens = np.zeros(self.num_docs, dtype=object)

        # Create root node
        self.root = NCRPNode(L=self.L, vocabulary=self.vocab, level_id=0, rng=self.rng)
        

        # Initialize each doc's path (c_d) at random
        path_buffer = np.zeros(self.L, dtype=object)
        path_buffer[0] = self.root

        for d_idx, doc in enumerate(self.documents):
            # "Seat" doc d at each level from root down
            self.root.num_docs += 1
            for depth in range(1, self.L):
                child_node = path_buffer[depth - 1].ncrp_draw_child(self.gamma)
                child_node.num_docs += 1
                path_buffer[depth] = child_node

            leaf_node = path_buffer[self.L - 1]
            self.doc_leaf[d_idx] = leaf_node

            # Randomly initialize each token's level z_{di}
            self.z_tokens[d_idx] = np.zeros(len(doc), dtype=int)
            for i, w_id in enumerate(doc):
                z_level = self.rng.randint(self.L)
                self.z_tokens[d_idx][i] = z_level
                path_buffer[z_level].n_w[w_id] += 1
                path_buffer[z_level].n_sum += 1

    def gibbs_sampling(self, iterations,
                   display_interval=50, top_n=5, show = True):
        """
        Run collapsed Gibbs sampling for hierarchical LDA, with burn-in.

        Parameters
        ----------
        iterations : int
            Total number of Gibbs sampling iterations.
        display_interval : int
            Print out topics every N iterations.
        top_n : int
            Number of top words to display for each node.
        show : bool
            Whether or not to print the structure
        """
        print("Starting hierarchical LDA sampling...")
        start_time = time.time()

        for it in range(iterations):
            # 1) Path sampling
            for d in range(self.num_docs):
                self._sample_path_for_doc(d)

            # 2) Level assignment sampling
            for d in range(self.num_docs):
                self._sample_levels_for_doc(d)

            
            # Display topics
            if (it + 1) % display_interval == 0 and show == True:
                print(f"Iteration {it+1}")
                self.exhibit_topics(top_n=top_n)
                
        end_time = time.time()
        total_time_seconds = end_time - start_time
        total_time_minutes = round(total_time_seconds / 60, 2)
        print(f"Total topic nodes created = {NCRPNode.total_created_nodes}")
        return total_time_minutes

    ##########################################################################
    # PATH SAMPLING: Sample c_d for each document
    ##########################################################################
    def _sample_path_for_doc(self, d_idx):
        """
        Sample a new path c_d for document d via the nCRP prior * doc-likelihood.
        """
        # 1) Extract the old path from leaf -> root
        old_path = []
        cursor = self.doc_leaf[d_idx]
        while cursor is not None:
            old_path.append(cursor)
            cursor = cursor.parent
        old_path.reverse()  # now root -> leaf

        # 2) Remove the doc from that old path
        self.doc_leaf[d_idx].decrement_doc()

        # 3) Build local word hist of doc by level
        doc_levels = self.z_tokens[d_idx]
        doc_words = self.documents[d_idx]
        level_hist = [{} for _ in range(self.L)]
        # Decrement from each node's counts
        for token_i, w_id in enumerate(doc_words):
            z_level = doc_levels[token_i]
            level_hist[z_level][w_id] = level_hist[z_level].get(w_id, 0) + 1
            old_path[z_level].n_w[w_id] -= 1
            old_path[z_level].n_sum -= 1

        # 4) Compute nCRP prior for all possible paths
        path_logprob = {}
        self._compute_ncrp_prior(node=self.root, log_prob=0.0, prior_map=path_logprob)

        # 5) Compute doc-likelihood for each node
        #    (assuming that node is "leaf" for doc d)
        self._calc_doc_likelihood(path_logprob, level_hist)

        # 6) Sample new leaf from path_logprob
        node_candidates = np.array(list(path_logprob.keys()))
        vals = np.array([path_logprob[n] for n in node_candidates])
        # log-sum-exp
        vals = np.exp(vals - np.max(vals))
        vals /= np.sum(vals)
        chosen_idx = self.rng.multinomial(1, vals).argmax()
        new_leaf = node_candidates[chosen_idx]

        # If chosen node not leaf => grow branch
        if not new_leaf.is_leaf_level():
            new_leaf = new_leaf.grow_branch_to_leaf()

        # 7) Add doc back into new_leaf path
        new_leaf.increment_doc()
        self.doc_leaf[d_idx] = new_leaf

        # 8) Restore doc's token counts in new path
        cursor = new_leaf
        for lvl in range(self.L-1, -1, -1):
            for w, ct in level_hist[lvl].items():
                cursor.n_w[w] += ct
                cursor.n_sum += ct
            cursor = cursor.parent

    def _compute_ncrp_prior(self, node, log_prob, prior_map):
        """
        Recursively compute the nCRP prior from root down.
        For each node, probability = num_docs/(num_docs + gamma), plus
        the probability of a new child = gamma/(num_docs + gamma).
        """
        # For each existing child
        for child in node.children:
            lp_child = log_prob + log(child.num_docs / (node.num_docs + self.gamma))
            self._compute_ncrp_prior(child, lp_child, prior_map)

        # Also the 'new child' branch
        prior_map[node] = log_prob + log(self.gamma / (node.num_docs + self.gamma))

    def _calc_doc_likelihood(self, path_logprob, level_hist):
        """
        For each node (potential leaf), add doc-likelihood from the tokens
        that doc d would place at each level.
        """
        # Precompute hypothetical new-topic log-likelihood at each level
        new_topic_log = np.zeros(self.L)
        for lvl in range(1, self.L):
            wdict = level_hist[lvl]
            tot = 0
            # brand-new node => no existing counts
            for w_id, count in wdict.items():
                for i in range(count):
                    new_topic_log[lvl] += log((self.eta + i) /
                                              (self.eta_sum + tot))
                    tot += 1

        self._accumulate_likelihood(node=self.root,
                                    base_val=0.0,
                                    level_hist=level_hist,
                                    new_topic_log=new_topic_log,
                                    depth=0,
                                    map_out=path_logprob)

    def _accumulate_likelihood(self, node, base_val, level_hist, new_topic_log, depth, map_out):
        """Recursively compute the word likelihood for each node in the path."""
        local_contrib = 0.0
        tokens_so_far = 0
        wdict = level_hist[depth]
        for w_id, count in wdict.items():
            for i in range(count):
                local_contrib += log((self.eta + node.n_w[w_id] + i) /
                                     (self.eta_sum + node.n_sum + tokens_so_far))
                tokens_so_far += 1

        # Recurse to children
        for ch in node.children:
            self._accumulate_likelihood(ch,
                                        base_val + local_contrib,
                                        level_hist,
                                        new_topic_log,
                                        depth+1,
                                        map_out)
        # If not leaf, consider hypothetical new-topic for deeper levels
        tmp_val = local_contrib
        lvl = depth + 1
        while lvl < self.L:
            tmp_val += new_topic_log[lvl]
            lvl += 1

        map_out[node] = map_out.get(node, base_val) + tmp_val

    ##########################################################################
    # LEVEL (z_{di}) SAMPLING: Resample each token in doc
    ##########################################################################
    def _sample_levels_for_doc(self, d_idx):
        """
        For each token in doc d_idx, resample which level z_{di} it belongs to,
        among the L nodes on that doc's path.
        """
        doc_words = self.documents[d_idx]
        z_doc = self.z_tokens[d_idx]
        level_counts = np.zeros(self.L, dtype=int)
        for lv in z_doc:
            level_counts[lv] += 1

        # Identify path from leaf->root
        path_arr = np.zeros(self.L, dtype=object)
        c_leaf = self.doc_leaf[d_idx]
        tmp = c_leaf
        for lv in range(self.L-1, -1, -1):
            path_arr[lv] = tmp
            tmp = tmp.parent

        for i, w_id in enumerate(doc_words):
            old_lv = z_doc[i]
            level_counts[old_lv] -= 1
            path_arr[old_lv].n_w[w_id] -= 1
            path_arr[old_lv].n_sum -= 1

            # Compute unnormalized probabilities
            p_z = np.zeros(self.L)
            for lv in range(self.L):
                p_z[lv] = ((self.alpha + level_counts[lv]) *
                           (self.eta + path_arr[lv].n_w[w_id]) /
                           (self.eta_sum + path_arr[lv].n_sum))

            # Normalize + sample
            p_z /= p_z.sum()
            new_lv = self.rng.multinomial(1, p_z).argmax()

            z_doc[i] = new_lv
            level_counts[new_lv] += 1
            path_arr[new_lv].n_w[w_id] += 1
            path_arr[new_lv].n_sum += 1

    ##########################################################################
    # Helper / Display
    ##########################################################################
    def exhibit_topics(self, top_n=5, show_counts=True, structure=False):
        """
        If structure=False:
            Print out the topics from the root downward with top words.
        If structure=True:
            Print out just the structure:
            Node X (level=Y, children=Z)
        """
        self._display_subtree(self.root, indent=0, top_n=top_n,
                            show_counts=show_counts, structure=structure)

    def _display_subtree(self, node, indent, top_n, show_counts, structure):
        if structure:
            # Show only structural info
            prefix = "  " * indent
            print(f"{prefix}Topic Node {node.node_index} "
                f"(level={node.level_id}, children={len(node.children)})")
        else:
            # Show topic info + top words
            prefix = "    " * indent
            desc = node.top_words(n_terms=top_n, show_counts=show_counts)
            print(f"{prefix}Topic Node {node.node_index} (level={node.level_id}, docs={node.num_docs}): {desc}")

        # Recurse for children
        for child in node.children:
            self._display_subtree(child, indent+1, top_n, show_counts, structure)
                
    def traverse_tree(self, node=None):
        """
        A depth-first traversal (DFS) generator that yields
        each node in the hierarchy (root -> all descendants).
        
        Parameters
        ----------
        node : NCRPNode or None
            If None, we start from self.root. Otherwise, we start from the specified node.
        
        Yields
        ------
        NCRPNode
            Each node in the DFS.
        """
        if node is None:
            node = self.root

        yield node
        for child in node.children:
            yield from self.traverse_tree(child)

    ## Retriving path information
    def get_path(self, leaf_node):
        """
        Return the list of nodes from root->leaf for the given leaf_node.
        """
        path_nodes = []
        cursor = leaf_node
        while cursor is not None:
            path_nodes.append(cursor)
            cursor = cursor.parent
        path_nodes.reverse()
        return path_nodes

    def get_path_info(self, leaf_node):
        """
        Return (node_index, level_id) for each node from root->leaf.
        """
        nodes = self.get_path(leaf_node)
        return [(n.node_index, n.level_id) for n in nodes]
    
    ## Hyperparameter testing result analysis
    def gamma_eval(self, levels=None):
        """
        Evaluate the effect of gamma by counting how many 'tables' (nodes)
        exist at each level in the nCRP tree.

        Parameters
        ----------
        levels : int or None
            If given, only compute for levels [0..levels-1].
            If None, use self.L (all levels).

        Returns
        -------
        list of int
            counts[i] = number of nodes at level i
        """
        if levels is None or levels > self.L:
            levels = self.L

        level_counts = [0] * levels

        for node in self.traverse_tree():
            if node.level_id < levels:
                level_counts[node.level_id] += 1

        return level_counts

    
    
    def alpha_eval(self, document_id):
        """
        Evaluate the effect of alpha by looking at how tokens in a given document
        are allocated across the L levels on its path.

        Parameters
        ----------
        document_id : int
            ID of the document to analyze.

        Returns
        -------
        list of float
            fractions[i] = fraction of doc's tokens assigned to level i.
                        Summation over i is 1.0.
        """
        # Check
        if document_id < 0 or document_id >= self.num_docs:
            raise ValueError(f"document_id must be in [0, {self.num_docs-1}]")

        doc_levels = self.z_tokens[document_id]  # array of level assignments
        doc_length = len(doc_levels)
        if doc_length == 0:
            return [0.0] * self.L  # edge case: empty doc

        level_counts = np.bincount(doc_levels, minlength=self.L)  # shape=(L,)
        fractions = level_counts / float(doc_length)

        return fractions.tolist()


    def eta_eval(self, n=5):
        """
        Evaluate the effect of eta by calculating, level by level,
        the average fraction of each topic's total word count
        that is contributed by its top-n words.

        For each level `lvl` in [0..L-1]:
        1. Find all nodes at that level.
        2. For each node, compute coverage = sum of counts in top-n words / node.n_sum
        3. Take the average coverage over all nodes at this level.

        Parameters
        ----------
        n : int
            The number of top words to consider.

        Returns
        -------
        list of float
            coverage_per_level[lvl] = average coverage of top-n words
                                    among all topics at level `lvl`.
                                    If there are no topics at that level,
                                    the coverage is reported as 0.
        """
        level_coverages = [[] for _ in range(self.L)]
        
        for node in self.traverse_tree():
            if node.n_sum > 0:
                # Sort node.n_w to find top-n counts
                order = np.argsort(node.n_w)[::-1]
                top_n_ids = order[:n]
                coverage = node.n_w[top_n_ids].sum() / node.n_sum
            else:
                coverage = 0.0
            level_coverages[node.level_id].append(coverage)

        coverage_per_level = []
        for lvl in range(self.L):
            vals = level_coverages[lvl]
            coverage_per_level.append(np.mean(vals) if vals else 0.0)

        return coverage_per_level

    def generate_synthetic_corpus(self, num_docs, words_per_doc, alpha_dir, rng=None):
        """
        Generates a synthetic corpus using the learned hLDA tree structure using the hlda generative process.

        Parameters
        ----------
        num_docs : int
            Number of documents to generate.
        words_per_doc : int
            Number of words per document.
        alpha_dir : float
            Concentration parameter for the Dirichlet distribution over topic proportions.
        rng : np.random.RandomState, optional
            Random number generator for reproducibility.
        Returns
        -------
        synthetic_corpus : list of lists
            Synthetic corpus where each document is a list of word IDs.
        """
        # 1) Collect unique leaves from doc_leaf
        leaf_nodes = set(self.doc_leaf.values())
        leaf_nodes = list(leaf_nodes)
        if not leaf_nodes:
            raise ValueError("No leaf nodes (leaf_nodes) found in the tree.")

        if rng is None:
            rng = self.rng

        synthetic_corpus = []
        V = len(self.vocab) 

        for _ in range(num_docs):
            # 2) Sample a random leaf node (uniform)
            leaf = rng.choice(leaf_nodes)
            # Get the path from root->leaf (length = self.L)
            path_nodes = self.get_path(leaf)

            # 3) Sample a Dirichlet distribution over these L nodes
            theta = rng.dirichlet([alpha_dir] * self.L)

            # 4) Generate a new document
            doc = []
            for _w in range(words_per_doc):
                z = rng.choice(self.L, p=theta)
                node = path_nodes[z]

                numerator = node.n_w + self.eta
                denominator = node.n_sum + self.eta_sum
                if denominator < 1e-12:
                    probs = np.ones(V) / V
                else:
                    probs = numerator / denominator

                # sample a word
                word = rng.choice(V, p=probs)
                doc.append(word)

            synthetic_corpus.append(doc)
        return synthetic_corpus
    
    def visualise_tree(self, show_level_info=False):
        """
        Visualize the hLDA tree using Graphviz and display it using matplotlib.
        
        Parameters
        ----------
        show_level_info : bool
            If True, include the node's level info in the label.
        """
        dot = Digraph(comment="hLDA Tree", format="png")

        def traverse(node):
            # Create a label with minimal info (node index, optionally level)
            label = f"{node.node_index}" if not show_level_info else f"{node.node_index}\nL{node.level_id}"
            dot.node(str(node.node_index), label)
            for child in node.children:
                dot.edge(str(node.node_index), str(child.node_index))
                traverse(child)

        traverse(self.root)
        
        # Render the diagram to PNG and display using matplotlib
        png_bytes = dot.pipe(format='png')
        image_stream = io.BytesIO(png_bytes)
        image = Image.open(image_stream)
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.show()