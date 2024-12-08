import numpy as np
from collections import defaultdict
from math import log
from functools import lru_cache
from scipy.special import gammaln

class Node:
    """
    Represents a node in the hierarchical tree for hLDA.

    Attributes:
        children (dict): Mapping from topic IDs to child Node instances.
        documents (int): Number of documents assigned to this node.
        word_counts (defaultdict): Counts of words assigned to this node.
        total_words (int): Total number of words assigned to this node.
        parent (Node): Parent node in the tree.
        level (int): Depth level of the node in the tree.
    """
    def __init__(self, parent=None, level=0):
        """
        Initialize a Node.

        Args:
            parent (Node, optional): The parent node. Defaults to None.
            level (int, optional): The depth level of the node in the tree. Defaults to 0.
        """
        self.children = {}
        self.documents = 0
        self.word_counts = defaultdict(int)
        self.total_words = 0
        self.parent = parent
        self.level = level

    def is_leaf(self):
        """
        Check if the node is a leaf (i.e., has no children).

        Returns:
            bool: True if the node has no children, False otherwise.
        """
        return len(self.children) == 0

    def add_child(self, topic_id):
        """
        Add a child node with the specified topic ID.

        Args:
            topic_id (int): Identifier for the child topic.

        Returns:
            Node: The newly created child node.
        """
        child_node = Node(parent=self, level=self.level + 1)
        self.children[topic_id] = child_node
        return child_node

    def remove_child(self, topic_id):
        """
        Remove a child node with the specified topic ID.

        Args:
            topic_id (int): Identifier of the child topic to remove.
        """
        if topic_id in self.children:
            del self.children[topic_id]


class nCRPTree:
    """
    Implements the nested Chinese Restaurant Process (nCRP) tree for hLDA.

    Attributes:
        root (Node): The root node of the hierarchical tree.
        gamma (float): Concentration parameter for the nCRP.
        eta (float): Smoothing parameter for topic-word distributions.
        V (int): Vocabulary size.
        eta_sum (float): Precomputed value of eta multiplied by vocabulary size.
        num_levels (int): Maximum depth of the hierarchical tree.
        paths (dict): Mapping from document IDs to their assigned path nodes.
        levels (dict): Mapping from document IDs to their word-level assignments.
        document_words (dict): Mapping from document IDs to their words.
        m (float): Hyperparameter controlling the probability of assigning to deeper levels.
        pi (float): Hyperparameter influencing the prior over levels.
    """
    def __init__(self, gamma, eta, num_levels, vocab, m=0.5, pi=1.0):
        """
        Initialize the nCRPTree.

        Args:
            gamma (float): Concentration parameter for the nCRP.
            eta (float): Smoothing parameter for topic-word distributions.
            num_levels (int): Maximum depth of the hierarchical tree.
            vocab (list): List of words in the vocabulary.
            m (float, optional): Hyperparameter for level assignments. Defaults to 0.5.
            pi (float, optional): Hyperparameter for level assignments. Defaults to 1.0.
        """
        self.root = Node()
        self.gamma = gamma
        self.eta = eta
        self.V = len(vocab)
        self.eta_sum = self.eta * self.V
        self.num_levels = num_levels
        self.paths = {}
        self.levels = {}
        self.document_words = {}
        self.m = m
        self.pi = pi

    @lru_cache(maxsize=None)
    def cached_gammaln(self, x):
        """
        Cached computation of the gammaln function.

        Args:
            x (float): Input value for gammaln.

        Returns:
            float: The gammaln of x.
        """
        return gammaln(x)

    def sample_ncrp_path(self, node):
        """
        Sample a child from the nCRP at a given node during initialization.

        Probability of existing child: child.documents / (node.documents + gamma)
        Probability of new child: gamma / (node.documents + gamma)

        Args:
            node (Node): The current node from which to sample.

        Returns:
            tuple:
                chosen_topic_id (int): The ID of the chosen topic.
                is_new (bool): True if a new child was created, False otherwise.
        """
        total_customers = node.documents
        topic_probabilities = {}

        # Existing children probabilities
        for topic_id, child in node.children.items():
            topic_probabilities[topic_id] = child.documents / (total_customers + self.gamma)

        # New child probability
        new_topic_key = max(node.children.keys(), default=0) + 1
        topic_probabilities[new_topic_key] = self.gamma / (total_customers + self.gamma)

        # Sampling
        topics = list(topic_probabilities.keys())
        probs = np.array(list(topic_probabilities.values()))
        probs /= probs.sum()  # Ensure normalization
        chosen = np.random.choice(topics, p=probs)
        is_new = (chosen not in node.children)
        return chosen, is_new

    def initialize_new_path(self, max_depth, document_id):
        """
        Initialize a new path for a document during tree initialization.

        Args:
            max_depth (int): Maximum depth of the tree.
            document_id (int): Identifier for the document.

        Returns:
            list: List of nodes representing the path assigned to the document.
        """
        current_node = self.root
        current_node.documents += 1
        path_nodes = [current_node]

        for level in range(1, max_depth):
            topic_id, is_new = self.sample_ncrp_path(current_node)
            if is_new:
                child_node = current_node.add_child(topic_id)
            else:
                child_node = current_node.children[topic_id]
            child_node.documents += 1
            path_nodes.append(child_node)
            current_node = child_node

        self.paths[document_id] = path_nodes
        return path_nodes

    def initialise_tree(self, corpus, max_depth):
        """
        Initialize the tree with a corpus of documents.

        Args:
            corpus (list of lists): Corpus where each document is a list of words.
            max_depth (int): Maximum depth of the tree.
        """
        for doc_id, doc_words in enumerate(corpus):
            self.document_words[doc_id] = doc_words
            path_nodes = self.initialize_new_path(max_depth, doc_id)
            doc_levels = []
            num_levels = len(path_nodes)
            for w in doc_words:
                level = np.random.randint(0, num_levels)
                doc_levels.append(level)
                node = path_nodes[level]
                node.word_counts[w] += 1
                node.total_words += 1
            self.levels[doc_id] = doc_levels

    def add_document(self, document_id, path_nodes, level_word_counts):
        """
        Add a document to the tree by updating path assignments and counts.

        Args:
            document_id (int): Identifier for the document.
            path_nodes (list of Node): List of nodes representing the document's path.
            level_word_counts (dict): Mapping from levels to word counts for the document.
        """
        self.paths[document_id] = path_nodes
        for node in path_nodes:
            node.documents += 1
        for level, w_counts in level_word_counts.items():
            node = path_nodes[level]
            for w, cnt in w_counts.items():
                node.word_counts[w] += cnt
                node.total_words += cnt

    def remove_document(self, document_id):
        """
        Remove a document from the tree, updating counts and pruning if necessary.

        Args:
            document_id (int): Identifier for the document to remove.
        """
        if document_id not in self.paths:
            return
        path_nodes = self.paths[document_id]
        doc_levels = self.levels[document_id]
        doc_words = self.document_words[document_id]

        # Decrement word counts
        for w, lvl in zip(doc_words, doc_levels):
            node = path_nodes[lvl]
            node.word_counts[w] -= 1
            if node.word_counts[w] == 0:
                del node.word_counts[w]
            node.total_words -= 1

        # Decrement document counts
        for node in path_nodes:
            node.documents -= 1

        # Prune empty leaves
        for node in reversed(path_nodes):
            if node.documents == 0 and node.is_leaf() and node.parent is not None:
                parent = node.parent
                remove_id = None
                for tid, cnode in parent.children.items():
                    if cnode == node:
                        remove_id = tid
                        break
                if remove_id is not None:
                    parent.remove_child(remove_id)

        # Remove document mappings
        del self.paths[document_id]
        del self.levels[document_id]
        del self.document_words[document_id]

    def get_level_word_counts(self, document_words, document_levels):
        """
        Aggregate word counts per level for a document.

        Args:
            document_words (list of str): List of words in the document.
            document_levels (list of int): List of level assignments for each word.

        Returns:
            dict: Mapping from levels to word counts.
        """
        level_word_counts = {}
        for w, lvl in zip(document_words, document_levels):
            if lvl not in level_word_counts:
                level_word_counts[lvl] = {}
            level_word_counts[lvl][w] = level_word_counts[lvl].get(w, 0) + 1
        return level_word_counts

    def level_likelihood(self, node, M):
        """
        Compute the integrated likelihood contribution at a single level.

        Args:
            node (Node): The node representing the current level.
            M (dict of str: int): A dictionary mapping words to their counts in the document at this level.

        Returns:
            float: The log integrated likelihood for this level.
        """
        Nminus = node.word_counts
        sumNminus = node.total_words
        sumM = sum(M.values())
        sumN = sumNminus + sumM

        # Compute log terms
        # part1 = logGamma(sumNminus + Veta) - sum_w logGamma(Nminus(w) + eta)
        log_part1 = self.cached_gammaln(sumNminus + self.eta_sum)
        for w_count in Nminus.values():
            log_part1 -= self.cached_gammaln(w_count + self.eta)

        # part2 = sum_w logGamma(Nminus(w) + M(w) + eta) - logGamma(sumN + Veta)
        log_part2 = 0.0
        for w, n_w_minus in Nminus.items():
            n_w = n_w_minus + M.get(w, 0)
            log_part2 += self.cached_gammaln(n_w + self.eta)
        for w, mcount in M.items():
            if w not in Nminus:
                log_part2 += self.cached_gammaln(mcount + self.eta)

        log_part2 -= self.cached_gammaln(sumN + self.eta_sum)

        return log_part1 + log_part2

    def level_prior(self, parent, child_is_new, child_node=None):
        """
        Compute the log prior contribution from the nCRP at one level.

        Args:
            parent (Node): The parent node in the tree.
            child_is_new (bool): True if the child is a new node, False otherwise.
            child_node (Node, optional): The child node if existing. Defaults to None.

        Returns:
            float: The log prior contribution for this level.
        """
        total_customers = parent.documents
        if child_is_new:
            # New path
            return log(self.gamma) - log(total_customers + self.gamma)
        else:
            # Existing path
            return log(child_node.documents) - log(total_customers + self.gamma)

    def sample_path_level(self, parent_node, level_word_counts, level_index):
        """
        Sample a node at the given level from the parent node.

        Args:
            parent_node (Node): The parent node from which to sample.
            level_word_counts (dict): Mapping from levels to word counts for the document.
            level_index (int): The current level index being sampled.

        Returns:
            Node: The sampled child node.
        """
        candidates = list(parent_node.children.items())  # (topic_id, node)
        new_topic_id = max(parent_node.children.keys(), default=0) + 1

        M = level_word_counts.get(level_index, {})

        # Compute log probabilities for existing children
        log_probs = []
        for topic_id, child_node in candidates:
            lp = self.level_prior(parent_node, False, child_node)
            lp += self.level_likelihood(child_node, M)
            log_probs.append((lp, topic_id, False))  # (log_prob, topic_id, is_new)

        # Compute log probability for a new child
        # For a new node, Nminus=0 for all words
        fake_node = Node(parent=parent_node, level=parent_node.level + 1)
        lp_new = self.level_prior(parent_node, True)
        lp_new += self.level_likelihood(fake_node, M)
        log_probs.append((lp_new, new_topic_id, True))

        # Normalize and sample using log-sum-exp trick for numerical stability
        lps = [x[0] for x in log_probs]
        max_lp = max(lps)
        weights = np.exp([lp - max_lp for lp in lps])
        probs = weights / weights.sum()

        chosen_index = np.random.choice(len(probs), p=probs)
        chosen_lp, chosen_topic_id, chosen_is_new = log_probs[chosen_index]

        if chosen_is_new:
            child_node = parent_node.add_child(chosen_topic_id)
        else:
            child_node = parent_node.children[chosen_topic_id]

        return child_node

    def sample_path(self, document_id, document_words, document_levels):
        """
        Sample a new path for a document using a top-down approach.

        Steps:
            1. Remove the document from its current path (if assigned).
            2. Compute word counts per level for the document.
            3. Sequentially sample each level's node based on current tree state and word counts.
            4. Assign the sampled path to the document.

        Args:
            document_id (int): Identifier for the document.
            document_words (list of str): List of words in the document.
            document_levels (list of int): List of level assignments for each word in the document.
        """
        # Remove the document to reflect N_{-d}
        if document_id in self.paths:
            self.remove_document(document_id)

        # Aggregate word counts per level
        level_word_counts = self.get_level_word_counts(document_words, document_levels)

        # Top-down sampling
        current_node = self.root
        path_nodes = [current_node]

        for ell in range(1, self.num_levels):
            child_node = self.sample_path_level(current_node, level_word_counts, ell)
            path_nodes.append(child_node)
            current_node = child_node

        # Add the document back to the sampled path
        self.add_document(document_id, path_nodes, level_word_counts)
        self.levels[document_id] = document_levels
        self.document_words[document_id] = document_words

        # Optional: Validate sampled path against document_levels
        max_level = max(document_levels) if document_levels else 0
        assert max_level < self.num_levels, "Document level assignments exceed maximum tree depth."

    def compute_level_prior_probs(self, z_counts, max_z, m, pi):
        """
        Compute the vector of p(z_{d,n}=k | z_{d,-n}, m, pi) for k=0,...,max_z
        using the formula from the paper.

        Args:
            z_counts (list of int): List of counts for each level up to max_z.
            max_z (int): Maximum level currently assigned in the document.
            m (float): Hyperparameter controlling the probability of assigning to deeper levels.
            pi (float): Hyperparameter influencing the prior over levels.

        Returns:
            list of float: List of prior probabilities for each level.
        """
        # z_counts[k] = number of times level k appears (excluding current word)
        # Compute cumulative sums from top
        sum_ge = [0]*(max_z+1)
        running_sum = 0
        for level_idx in range(max_z, -1, -1):
            running_sum += z_counts[level_idx]
            sum_ge[level_idx] = running_sum

        probs = []
        for k in range(max_z+1):
            # numerator for level k
            numerator = (m * pi + z_counts[k])
            denominator = (pi + sum_ge[k]) if sum_ge[k] > 0 else pi
            level_prob = numerator / denominator

            # product term for j=0 to k-1
            for j in range(k):
                numerator_j = ((1 - m) * pi + z_counts[j])
                denominator_j = (pi + sum_ge[j]) if sum_ge[j] > 0 else pi
                level_prob *= (numerator_j / denominator_j)
            probs.append(level_prob)

        return probs

    def sample_level_assignment_for_word(self, document_id, n):
        """
        Sample a new level assignment for word n in document_id.
        Follows Section 5.1 of the hLDA paper.

        Args:
            document_id (int): Identifier for the document.
            n (int): Index of the word in the document.
        """
        doc_words = self.document_words[document_id]
        doc_levels = self.levels[document_id]
        old_level = doc_levels[n]
        w = doc_words[n]

        path_nodes = self.paths[document_id]

        # Remove current word from the node
        old_node = path_nodes[old_level]
        old_node.word_counts[w] -= 1
        if old_node.word_counts[w] == 0:
            del old_node.word_counts[w]
        old_node.total_words -= 1

        # Update doc_levels temporarily
        doc_levels[n] = -1

        # Compute current z_counts for this document
        # z_counts[k] = number of times level k appears (excluding current word)
        current_levels = doc_levels
        z_counts = defaultdict(int)
        for lvl in current_levels:
            if lvl >= 0:
                z_counts[lvl] += 1

        # Find max_z (the maximum assigned level excluding this word)
        max_z = max(z_counts.keys()) if z_counts else 0

        # Compute prior probabilities for levels 0 to max_z
        level_range = list(range(max_z + 1))
        z_counts_list = [z_counts.get(k, 0) for k in level_range]
        prior_probs = self.compute_level_prior_probs(z_counts_list, max_z, self.m, self.pi)

        # Compute word likelihoods for existing levels
        word_likelihoods = []
        for k in level_range:
            node = path_nodes[k]
            w_count = node.word_counts.get(w, 0)
            likelihood = (w_count + self.eta) / (node.total_words + self.eta_sum)
            word_likelihoods.append(likelihood)

        # Incorporate word likelihood into existing level probabilities
        for i in range(len(prior_probs)):
            prior_probs[i] *= word_likelihoods[i]

        # Compute the probability mass remaining for new levels
        sum_existing = sum(prior_probs)
        p_remaining = 1.0 - sum_existing

        # Initialize lists for extended levels
        extended_levels = []
        extended_probs = []

        # Sequentially consider new levels beyond max_z
        current_p = p_remaining
        current_level = max_z + 1

        while current_level < self.num_levels and current_p > 1e-15:
            # For a new level, the likelihood is eta / (eta_sum)
            new_level_likelihood = self.eta / self.eta_sum

            # Prior probability for the new level
            # According to the paper, the probability of going deeper involves m
            p_new_level = (1 - self.m) * new_level_likelihood * current_p

            # Append to extended levels
            extended_levels.append(current_level)
            extended_probs.append(p_new_level)

            # Update the remaining probability
            current_p -= p_new_level

            # Move to the next level
            current_level += 1

        # Combine existing and new level probabilities
        final_levels = level_range + extended_levels
        final_probs = prior_probs + extended_probs

        # Normalize probabilities to sum to 1
        total = sum(final_probs)
        if total == 0:
            # Assign to the maximum existing level if no probability mass remains
            new_level = max_z
        else:
            probs = [p / total for p in final_probs]
            new_level = np.random.choice(final_levels, p=probs)

        # Assign the new level
        new_node = path_nodes[new_level]
        new_node.word_counts[w] += 1
        new_node.total_words += 1
        doc_levels[n] = new_level

    def sample_levels_for_document(self, document_id):
        """
        Sample levels z_{d,n} for each word in document d.

        Args:
            document_id (int): Identifier for the document.
        """
        doc_words = self.document_words[document_id]
        for n in range(len(doc_words)):
            self.sample_level_assignment_for_word(document_id, n)

    def gibbs_sampling(self, corpus, num_iterations, burn_in=100, thinning=10):
        """
        Perform Gibbs sampling on the corpus.

        Args:
            corpus (list of lists): Corpus where each document is a list of words.
            num_iterations (int): Number of Gibbs sampling iterations.
            burn_in (int, optional): Number of initial iterations to discard. Defaults to 100.
            thinning (int, optional): Interval for collecting samples. Defaults to 10.
        """
        # Initialize the tree with the corpus
        self.initialise_tree(corpus, max_depth=self.num_levels)

        for it in range(num_iterations):
            for doc_id in range(len(corpus)):
                # Sample path for the document
                document_words = self.document_words[doc_id]
                document_levels = self.levels[doc_id]
                self.sample_path(doc_id, document_words, document_levels)

                # Sample level assignments for each word in the document
                self.sample_levels_for_document(doc_id)

            # Progress reporting
            if (it + 1) % thinning == 0:
                print(f"Iteration {it + 1} completed.")
            if (it + 1) == burn_in:
                print(f"Burn-in period of {burn_in} iterations completed.")

        print("Gibbs sampling completed.")
        
        
