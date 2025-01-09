import numpy as np

from math import log
from numpy.random import RandomState


class HLDA_Node(object):

    # Class-wide tracking of how many nodes have been created
    total_created_nodes = 0
    highest_node_index = 0

    def __init__(self, hierarchy_levels, vocabulary, parent_node=None, current_level=0, rng=None):

        self.node_index = HLDA_Node.highest_node_index
        HLDA_Node.highest_node_index += 1

        self.num_customers = 0
        self.parent_node = parent_node
        self.child_nodes = []
        self.level_id = current_level
        self.word_total = 0
        self.hierarchy_depth = hierarchy_levels

        self.vocabulary = np.array(vocabulary)
        self.word_freqs = np.zeros(len(vocabulary))

        if rng is None:
            self.rng = RandomState()
        else:
            self.rng = rng

    def __repr__(self):
        parent_idx = None
        if self.parent_node is not None:
            parent_idx = self.parent_node.node_index
        return (f'Node={self.node_index} level={self.level_id} '
                f'customers={self.num_customers} total_words={self.word_total} '
                f'parent={parent_idx}')

    def add_child_node(self):
        new_child = HLDA_Node(self.hierarchy_depth, self.vocabulary, 
                              parent_node=self, 
                              current_level=self.level_id + 1)
        self.child_nodes.append(new_child)
        HLDA_Node.total_created_nodes += 1
        return new_child

    def is_bottom_level(self):
        return self.level_id == self.hierarchy_depth - 1

    def grow_branch_to_leaf(self):
        """
        Extend child nodes all the way down to create a leaf if this node 
        isn't already one.
        """
        node_cursor = self
        for lvl in range(self.level_id, self.hierarchy_depth - 1):
            node_cursor = node_cursor.add_child_node()
        return node_cursor

    def decrement_path(self):
        """
        Remove a document from the path. If a node becomes empty, prune it.
        """
        cursor = self
        cursor.num_customers -= 1
        if cursor.num_customers == 0:
            cursor.parent_node.remove_child(cursor)
        for _ in range(1, self.hierarchy_depth):  # starting from the level below root
            cursor = cursor.parent_node
            cursor.num_customers -= 1
            if cursor.num_customers == 0:
                cursor.parent_node.remove_child(cursor)

    def remove_child(self, child):
        """
        Remove a child node reference from this node.
        """
        self.child_nodes.remove(child)
        HLDA_Node.total_created_nodes -= 1

    def increment_path(self):
        """
        Add a document into this path, increasing the count of each node
        along the chain.
        """
        cursor = self
        cursor.num_customers += 1
        for _ in range(1, self.hierarchy_depth):
            cursor = cursor.parent_node
            cursor.num_customers += 1

    def CRP(self, gamma):
        """
        Implements the CRP logic: either select an existing child
        or create a new child (depending on gamma).
        
        :param gamma: Hyperparameter that controls the likelihood of creating a new child node.
                      - A higher gamma increases the probability of adding new nodes (more exploration).
                      - A lower gamma encourages nodes to reuse existing children (more exploitation).
                      - **Bounded**: `gamma >= 0`. Typically set to a value between 0 and 10.
        """
        all_candidates = len(self.child_nodes) + 1
        probabilities = []
        probabilities.append(gamma / (gamma + self.num_customers)) 
        for children in self.child_nodes:
            probabilities.append(float(children.num_customers) / (gamma + self.num_customers))

        chosen = self.rng.multinomial(1, probabilities).argmax()
        
        if chosen == 0:
            return self.add_child_node()
        else:
            return self.child_nodes[chosen - 1]

    def top_n_words(self, n_terms, show_weights):
        """
        Return the top n words (and optionally their counts).
        """
        sorted_positions = np.argsort(self.word_freqs)[::-1]
        sorted_vocab = self.vocabulary[sorted_positions][:n_terms]
        sorted_counts = self.word_freqs[sorted_positions][:n_terms]

        words_out = []
        for w, c in zip(sorted_vocab, sorted_counts):
            if show_weights:
                words_out.append(f'{w} ({int(c)})')
            else:
                words_out.append(w)
        return ', '.join(words_out)


class HierarchicalLDA(object):

    def __init__(self, corpus, vocabulary, levels,
                 alpha=10.0, gamma=1.0, eta=0.1,
                 seed=0, verbose=True):

        # Reset class-level counters
        HLDA_Node.total_created_nodes = 0
        HLDA_Node.highest_node_index = 0

        self.documents = corpus
        self.vocab = vocabulary
        
        self.alpha = alpha  # Smoothing for doc-topic distribution.
                             # A higher alpha increases topic diversity, while a lower alpha makes documents more specific to fewer topics.
                             # **Bounded**: `alpha > 0`. Typical values range from 1 to 100, depending on the desired granularity of topics.
        self.gamma = gamma  # CRP parameter: controls the likelihood of creating new nodes.
                             # - A larger gamma encourages the creation of new nodes (increased model complexity).
                             # - A smaller gamma favors reusing existing nodes (more stable structure).
                             # **Bounded**: `gamma >= 0`. Typical values range from 0.1 to 10.
        self.eta = eta      # Smoothing for topic-word distribution.
                             # A higher eta makes the model more likely to assign a word to multiple topics.
                             # A smaller eta makes the topic-word distributions more sparse.
                             # **Bounded**: `eta > 0`. Typical values range from 0.01 to 0.5.


        self.rand_seed = seed
        self.rng = RandomState(seed)
        self.verbose = verbose

        self.num_levels = levels
        self.num_docs = len(corpus)
        self.num_terms = len(vocabulary)
        self.eta_sum = self.eta * self.num_terms

        self.root = HLDA_Node(self.num_levels, self.vocab)
        # Keep track of leaf node for each document
        self.doc_to_leaf = {}
        # For each doc, keep track of the level assignment for each token
        self.token_levels = np.zeros(self.num_docs, dtype=object)

        # Temporary place holder of path for each of the document. Initialised for every single doc
        path_placeholder = np.zeros(self.num_levels, dtype=object) 
        path_placeholder[0] = self.root
        
        # Assign each document to an initial path
        for doc_idx, doc in enumerate(self.documents):
            doc_len = len(doc)
            self.root.num_customers += 1
            
            for depth in range(1, self.num_levels):
                parent = path_placeholder[depth - 1]
                chosen_child = parent.CRP(self.gamma)
                chosen_child.num_customers += 1
                path_placeholder[depth] = chosen_child

            # Store the leaf node for this doc
            leaf_for_doc = path_placeholder[self.num_levels - 1]
            self.doc_to_leaf[doc_idx] = leaf_for_doc

            # Randomly assign each token in doc to one of the levels in the path
            self.token_levels[doc_idx] = np.zeros(doc_len, dtype=int) 
            for token_i, word_id in enumerate(doc):
                chosen_level = self.rng.randint(self.num_levels)
                chosen_node = path_placeholder[chosen_level]
                chosen_node.word_freqs[word_id] += 1
                chosen_node.word_total += 1
                self.token_levels[doc_idx][token_i] = chosen_level

    def gibbs_sampling(self, iterations, topic_display_interval=50, top_n_words=5, show_word_counts=True):
        """
        Main sampling loop for Hierarchical LDA.
        """
        print('Starting Hierarchical LDA sampling\n')
        printing_counter = 0
        for it in range(iterations):
            # Calculate percentage of completion
            percent_done = (it + 1) / iterations * 100
            
            # Print every 10% progress
            if (it + 1) % (iterations // 10) == 0:  
                print(f"{int(percent_done)}% done")

            # 1) Path sampling
            for d_idx in range(self.num_docs):
                self.sample_new_path(d_idx)

            # 2) Level sampling
            for d_idx in range(self.num_docs):
                self.sample_new_levels(d_idx)
            
            # Display topics
            if (it > 0) and ((it + 1) % topic_display_interval == 0):
                printing_counter += 1
                print(f"*********************The {printing_counter} result**************************")
                self.exhibit_nodes(top_n_words, show_word_counts)
            
        print('Gibbs sampling completed')


    def sample_new_path(self, doc_index):
        """
        Sample a new path for a particular document through the NCRP tree.
        """
        # Retrieve path from the doc's leaf
        path_nodes = np.zeros(self.num_levels, dtype=object)
        node_cursor = self.doc_to_leaf[doc_index]
        for lvl in range(self.num_levels - 1, -1, -1):
            path_nodes[lvl] = node_cursor
            node_cursor = node_cursor.parent_node

        # Temporarily remove this doc's assignment from the path
        self.doc_to_leaf[doc_index].decrement_path()

        # Calculate prior: p(c_d | c_{-d})
        path_weights = {}
        self._calc_ncrp_prior(path_weights, self.root, 0.0)

        # Token allocation of current dox
        doc_levels = self.token_levels[doc_index] 
        # Current Document
        current_doc = self.documents[doc_index]
        # Empty dictionary for how many times each word appears at each level. Dict of Dict
        level_word_hist = {lvl: {} for lvl in range(self.num_levels)}

        # Remove doc's counts from those nodes
        for i, w in enumerate(current_doc): # Looping through tokens of current document
            level_assn = doc_levels[i]
            level_word_hist[level_assn][w] = level_word_hist[level_assn].get(w, 0) + 1

            # Update node
            path_nodes[level_assn].word_freqs[w] -= 1
            path_nodes[level_assn].word_total -= 1
            assert path_nodes[level_assn].word_freqs[w] >= 0
            assert path_nodes[level_assn].word_total >= 0

        # Incorporate likelihood: p(w_d | c, w_{-d}, z)
        self._calc_doc_likelihood(path_weights, level_word_hist)

        # Normalize weights and choose a new node
        candidate_nodes = np.array(list(path_weights.keys()))
        values = np.array([path_weights[n] for n in candidate_nodes])
        # Use log-sum-exp trick for numerical stability
        values = np.exp(values - np.max(values))
        values = values / np.sum(values)

        chosen_idx = self.rng.multinomial(1, values).argmax()
        new_node = candidate_nodes[chosen_idx]

        # If chosen node is not a leaf, grow it to a leaf
        if not new_node.is_bottom_level():
            new_node = new_node.grow_branch_to_leaf()

        # Add doc back in
        new_node.increment_path()
        self.doc_to_leaf[doc_index] = new_node

        # Restore word counts
        for lvl in range(self.num_levels - 1, -1, -1):
            for (w, cnt) in level_word_hist[lvl].items():
                new_node.word_freqs[w] += cnt
                new_node.word_total += cnt
            new_node = new_node.parent_node

    def _calc_ncrp_prior(self, weight_map, node, log_prob):
        """
        Recursively compute the nested CRP prior along all branches.
        """
        for child in node.child_nodes:
            branch_weight = log(float(child.num_customers) / (node.num_customers + self.gamma))
            self._calc_ncrp_prior(weight_map, child, log_prob + branch_weight)

        # Finally, add in the probability of a new table
        weight_map[node] = log_prob + log(self.gamma / (node.num_customers + self.gamma)) 

    def _calc_doc_likelihood(self, weight_map, level_word_hist):
        """
        Compute the document-specific likelihood term for each possible path.
        """
        # Pre-calculate log-likelihood for new (hypothetical) topics at each level
        new_topic_likelihood = np.zeros(self.num_levels)
        for lvl in range(1, self.num_levels):  # skip root
            word_dict = level_word_hist[lvl]
            total_tokens = 0

            for w, c in word_dict.items():
                for i in range(c):
                    new_topic_likelihood[lvl] += log((self.eta + i) / (self.eta_sum + total_tokens))
                    total_tokens += 1

        # Recurse down from the root
        self._accumulate_word_likelihood(weight_map, self.root, 0.0, level_word_hist, new_topic_likelihood, 0)

    def _accumulate_word_likelihood(self, weight_map, node, base_val, level_word_hist, new_topic_likelihood, depth):
        """
        Recursively compute the word likelihood for each node in the path.
        """
        local_contrib = 0.0
        word_counts = level_word_hist[depth]
        tokens_counted = 0

        for w, c in word_counts.items():
            for i in range(c):
                local_contrib += log((self.eta + node.word_freqs[w] + i) /
                                     (self.eta_sum + node.word_total + tokens_counted))
                tokens_counted += 1

        # Descend to children
        for child in node.child_nodes:
            self._accumulate_word_likelihood(
                weight_map,
                child,
                base_val + local_contrib,
                level_word_hist,
                new_topic_likelihood,
                depth + 1
            )

        # If not a leaf, add the hypothetical new-topic weights for deeper levels
        adjusted_depth = depth + 1
        modified_contrib = local_contrib
        while adjusted_depth < self.num_levels:
            modified_contrib += new_topic_likelihood[adjusted_depth]
            adjusted_depth += 1

        weight_map[node] = weight_map.get(node, base_val) + modified_contrib

    def sample_new_levels(self, doc_index):
        """
        Resample the topic level assignment for each token in a single document.
        """
        current_doc = self.documents[doc_index]
        doc_assignments = self.token_levels[doc_index]
        level_counts = np.zeros(self.num_levels, dtype=int)  # Changed dtype=np.int to dtype=int

        # Tally current level assignments
        for lvl in doc_assignments:
            level_counts[lvl] += 1

        # Identify the path (from bottom up)
        path_refs = np.zeros(self.num_levels, dtype=object)
        node_cursor = self.doc_to_leaf[doc_index]
        for lvl in range(self.num_levels - 1, -1, -1):
            path_refs[lvl] = node_cursor
            node_cursor = node_cursor.parent_node

        # For each token, sample a new level
        for i, w_id in enumerate(current_doc):
            old_level = doc_assignments[i]

            # Remove from current level
            level_counts[old_level] -= 1
            path_refs[old_level].word_freqs[w_id] -= 1
            path_refs[old_level].word_total -= 1

            # Compute level probabilities
            prob_vector = np.zeros(self.num_levels)
            for lvl in range(self.num_levels):
                prob_vector[lvl] = ((self.alpha + level_counts[lvl]) *
                                    (self.eta + path_refs[lvl].word_freqs[w_id]) /
                                    (self.eta_sum + path_refs[lvl].word_total))

            prob_vector /= np.sum(prob_vector)
            new_level = self.rng.multinomial(1, prob_vector).argmax()

            # Update with new assignment
            doc_assignments[i] = new_level
            level_counts[new_level] += 1
            path_refs[new_level].word_freqs[w_id] += 1
            path_refs[new_level].word_total += 1
            
#########################################################################
# Utility functions
#########################################################################

    def exhibit_nodes(self, n_top_terms, show_weights):
        """
        Helper function to print out topic nodes in a hierarchical fashion.
        """
        self._display_node(self.root, 0, n_top_terms, show_weights)

    def _display_node(self, node, indent, n_top_terms, show_weights):
        prefix = '    ' * indent
        node_info = (f'{prefix}topic={node.node_index} level={node.level_id} '
                     f'(docs={node.num_customers}): ')
        node_info += node.top_n_words(n_top_terms, show_weights)
        print(node_info)
        for ch in node.child_nodes:
            self._display_node(ch, indent + 1, n_top_terms, show_weights)
    
    def get_path_from_leaf(self, leaf_node):
        """
        Traverse from leaf_node up to the root, returning a list of nodes 
        in root->leaf order.
        Returns
        -------
        path_nodes: list of HLDA_Node from root -> leaf
        """
        path = []

        # Climb up the chain
        cursor = leaf_node
        while cursor is not None:
            path.append(cursor)
            cursor = cursor.parent_node

        # Returning the path from root to leaf
        path.reverse()
        return path