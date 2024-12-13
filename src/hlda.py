import numpy as np
from collections import defaultdict
from math import log
from functools import lru_cache
from scipy.special import gammaln
from graphviz import Digraph

class Node:
    """
    Represents a node in a hierarchical topic tree.

    Attributes:
        children (dict): A dictionary mapping topic IDs to child Node instances.
        documents (int): Number of documents passing through this node.
        word_counts (defaultdict): A dictionary counting the occurrences of each word in this node.
        total_words (int): Total number of non-unique words in this node.
        parent (Node, optional): Reference to the parent Node. Defaults to None.
        level (int): The depth level of this node in the tree. Root node has level 0.
    """

    def __init__(self, parent=None, level=0):
        """
        Initializes a new Node instance.

        Args:
            parent (Node, optional): The parent node in the hierarchy. Defaults to None.
            level (int, optional): The depth level of the node in the tree. Root node has level 0. Defaults to 0.
        """
        self.children = {} 
        self.documents = 0  
        self.word_counts = defaultdict(int)
        self.total_words = 0 
        self.parent = parent
        self.level = level

    def is_leaf(self):
        """
        Determines whether the node is a leaf node (i.e., has no children).

        Returns:
            bool: True if the node has no children, False otherwise.
        """
        return len(self.children) == 0

    def add_child(self, topic_id):
        """
        Adds a child node with the specified topic ID to the current node.

        Args:
            topic_id (hashable): The identifier for the child topic.

        Returns:
            Node: The newly created child Node instance.
        """
        child_node = Node(parent=self, level=self.level + 1)
        self.children[topic_id] = child_node
        return child_node

    def remove_child(self, topic_id):
        """
        Removes the child node with the specified topic ID from the current node.

        Args:
            topic_id (hashable): The identifier of the child topic to remove.

        Raises:
            KeyError: If the specified topic_id does not exist among the children.
        """
        if topic_id in self.children:
            del self.children[topic_id]
        else:
            raise KeyError(f"Topic ID {topic_id} not found among the children.")
        
class nCRPTree:
    """
    Attributes:
        root (Node): The root node of the nCRP tree.
        gamma (float): Concentration parameter for the nCRP, controlling the likelihood of creating new topics.
        eta (float): Smoothing parameter for topic-word distributions.
        V (int): Vocabulary size.
        vocab (list): List of vocabulary words.
        eta_sum (float): Precomputed sum of eta multiplied by vocabulary size for efficiency.
        num_levels (int): Maximum depth of the hierarchical tree.
        paths (dict): Maps document IDs to their assigned path of nodes in the tree.
        levels (dict): Maps document IDs to their word-level assignments.
        document_words (dict): Maps document IDs to their preprocessed word lists.
        m (float): Hyperparameter influencing level assignments.
        pi (float): Hyperparameter influencing level assignments.
    """
    
    def __init__(self, gamma, eta, num_levels, vocab, m=0.5, pi=1.0):
        """
        Initializes the nCRP tree with specified hyperparameters and vocabulary.

        Args:
            gamma (float): Concentration parameter for the nCRP.
            eta (float): Smoothing parameter for topic-word distributions.
            num_levels (int): Maximum depth of the hierarchical tree.
            vocab (list): List of vocabulary words of the entire corpus
            m (float, optional): Hyperparameter influencing level assignments. Defaults to 0.5.
            pi (float, optional): Hyperparameter influencing level assignments. Defaults to 1.0.
        """
        self.root = Node()
        self.gamma = gamma
        self.eta = eta
        self.V = len(vocab)
        self.vocab = vocab
        self.eta_sum = self.eta * self.V
        self.num_levels = num_levels
        self.paths = {}
        self.levels = {}               
        self.document_words = {}        ## Pre-processed vacabulary for each of the document.
        self.m = m
        self.pi = pi

    @lru_cache(maxsize=None)
    def cached_gammaln(self, x):
        """
        Caches the computation of the logarithm of the gamma function for efficiency.

        Args:
            x (float): The input value for which to compute gammaln(x).

        Returns:
            float: The computed value of gammaln(x).
        """
        return gammaln(x)

    def sample_ncrp_path(self, node):
        """
        Samples a path (topic) for a new document based on the current node's children.

        It calculates the probability of assigning the document to each existing child topic
        and the probability of creating a new topic. It then samples a topic based on these probabilities.

        Args:
            node (Node): The current node from which to sample the next topic in the path.

        Returns:
            tuple:
                chosen (hashable): The chosen topic ID (existing or new).
                is_new (bool): Flag indicating whether a new topic was created.
        """
        total_customers = node.documents
        topic_probabilities = {}

        # Existing children probabilities
        for topic_id, child in node.children.items():
            topic_probabilities[topic_id] = child.documents / (total_customers + self.gamma)

        # New child probability
        new_topic_key = max(node.children.keys(), default=0) + 1
        topic_probabilities[new_topic_key] = self.gamma / (total_customers + self.gamma)

        topics = list(topic_probabilities.keys())
        probs = np.array(list(topic_probabilities.values()))
        probs /= probs.sum()
        chosen = np.random.choice(topics, p=probs)
        is_new = (chosen not in node.children)
        return chosen, is_new

    def initialize_new_path(self, max_depth, document_id):
        """
        Initializes a new path for a document up to the specified maximum depth.

        It assigns the document to a path by sampling topics at each level and updating
        document counts for the nodes along the path.

        Args:
            max_depth (int): The maximum depth to assign the document in the tree.
            document_id (int): The unique identifier for the document being added.

        Returns:
            list: A list of Node instances representing the path assigned to the document.
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
        Initializes the nCRP tree by assigning each document in the corpus to a path.

        For each document, it assigns a path through the tree and assigns words to levels
        within that path based on random sampling.

        Args:
            corpus (list of list of str): The preprocessed corpus where each document is a list of words.
            max_depth (int): The maximum depth to assign in the tree.
        """
        for doc_id, doc_words in enumerate(corpus):
            self.document_words[doc_id] = doc_words
            
            # Path assignment
            path_nodes = self.initialize_new_path(max_depth, doc_id)
            doc_levels = []
            num_levels = len(path_nodes)
            
            # Level assignment
            for w in doc_words: 
                level = np.random.randint(0, num_levels)
                doc_levels.append(level)
                node = path_nodes[level]
                node.word_counts[w] += 1
                node.total_words += 1
            self.levels[doc_id] = doc_levels

    def add_document(self, document_id, path_nodes, level_word_counts):
        """
        Adds a document to the tree by updating document paths and word counts.

        Args:
            document_id (int): The unique identifier for the document being added.
            path_nodes (list of Node): The path of nodes assigned to the document.
            level_word_counts (dict): A mapping from level indices to word count dictionaries.
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
        Removes a document from the tree, updating document counts and pruning empty nodes.

        Args:
            document_id (int): The unique identifier for the document being removed.

        Raises:
            KeyError: If the document ID does not exist in the tree.
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

        del self.paths[document_id]
        del self.levels[document_id]
        del self.document_words[document_id]
        
    
    # Sampling Paths
    def get_level_word_counts(self, document_words, document_levels):
        """
        Aggregates word counts for each level in a document.

        Args:
            document_words (list of str): The list of words in the document.
            document_levels (list of int): The list of level assignments for each word.

        Returns:
            dict: A mapping from level indices to dictionaries of word counts.
        """
        level_word_counts = {}
        for w, lvl in zip(document_words, document_levels):
            if lvl not in level_word_counts:
                level_word_counts[lvl] = {}
            level_word_counts[lvl][w] = level_word_counts[lvl].get(w, 0) + 1
        return level_word_counts

    def level_likelihood(self, node, M): 
        """
        Computes the log-likelihood of words at a given level in a node.

        Args:
            node (Node): The node representing the current topic.
            M (dict): A dictionary of word counts assigned to this level.

        Returns:
            float: The computed log-likelihood.
        """
        Nminus = node.word_counts # The current counts of words in the node (excluding the new assignments).
        sumNminus = node.total_words # Total number of words in the node before adding the new assignments.
        sumM = sum(M.values()) # Total number of new word assignments.
        sumN = sumNminus + sumM # Updated total word count after adding new assignments.

        log_part1 = self.cached_gammaln(sumNminus + self.eta_sum)
        for w_count in Nminus.values():
            log_part1 -= self.cached_gammaln(w_count + self.eta)

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
        Computes the log prior probability for assigning a child node.

        Args:
            parent (Node): The parent node in the hierarchy.
            child_is_new (bool): Flag indicating whether the child node is new.
            child_node (Node, optional): The child node instance. Required if child_is_new is False. Defaults to None.

        Returns:
            float: The computed log prior probability.

        Raises:
            ValueError: If child_is_new is False and child_node is not provided.
        """
        total_customers = parent.documents
        if child_is_new:
            return log(self.gamma) - log(total_customers + self.gamma)
        else:
            return log(child_node.documents) - log(total_customers + self.gamma)

    def sample_path_level(self, parent_node, level_word_counts, level_index):
        """
        Samples a topic for a specific level in the path based on prior and likelihood.

        Args:
            parent_node (Node): The parent node at the current level.
            level_word_counts (dict): Word counts assigned to the current level.
            level_index (int): The current level index in the path.

        Returns:
            Node: The child node assigned to the current level.
        """
        candidates = list(parent_node.children.items())  # (topic_id, node)
        new_topic_id = max(parent_node.children.keys(), default=0) + 1

        M = level_word_counts.get(level_index, {})

        # Existing children
        log_probs = []
        for topic_id, child_node in candidates:
            lp = self.level_prior(parent_node, False, child_node)
            lp += self.level_likelihood(child_node, M)
            log_probs.append((lp, topic_id, False))

        # New child
        fake_node = Node(parent=parent_node, level=parent_node.level + 1)
        lp_new = self.level_prior(parent_node, True)
        lp_new += self.level_likelihood(fake_node, M)
        log_probs.append((lp_new, new_topic_id, True))

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
        Samples a new path for a document, reassigning its path in the tree.

        It removes the document from its current path, samples a new path based on word assignments,
        and updates the tree accordingly.

        Args:
            document_id (int): The unique identifier for the document being sampled.
            document_words (list of str): The list of words in the document.
            document_levels (list of int): The list of level assignments for each word in the document.
        """
        if document_id in self.paths:
            self.remove_document(document_id)

        level_word_counts = self.get_level_word_counts(document_words, document_levels)
        current_node = self.root
        path_nodes = [current_node]

        for ell in range(1, self.num_levels):
            child_node = self.sample_path_level(current_node, level_word_counts, ell)
            path_nodes.append(child_node)
            current_node = child_node

        self.add_document(document_id, path_nodes, level_word_counts)
        self.levels[document_id] = document_levels
        self.document_words[document_id] = document_words
        max_level = max(document_levels) if document_levels else 0
        assert max_level < self.num_levels, "Document level assignments exceed maximum tree depth."

    ## Sampling Levels
    def compute_level_prior_probs(self, z_counts, max_z, m, pi):
        """
        Computes prior probabilities for levels based on current assignments.

        Args:
            z_counts (list of int): Counts of assignments at each level up to max_z.
            max_z (int): The maximum level index with assignments.
            m (float): Hyperparameter influencing level assignments.
            pi (float): Hyperparameter influencing level assignments.

        Returns:
            list of float: The computed prior probabilities for each level.
        """
        sum_ge = [0]*(max_z+1)
        running_sum = 0
        for level_idx in range(max_z, -1, -1):
            running_sum += z_counts[level_idx]
            sum_ge[level_idx] = running_sum

        probs = []
        for k in range(max_z+1):
            numerator = (m * pi + z_counts[k])
            denominator = (pi + sum_ge[k]) if sum_ge[k] > 0 else pi
            level_prob = numerator / denominator

            for j in range(k):
                numerator_j = ((1 - m) * pi + z_counts[j])
                denominator_j = (pi + sum_ge[j]) if sum_ge[j] > 0 else pi
                level_prob *= (numerator_j / denominator_j)
            probs.append(level_prob)

        return probs

    def sample_level_assignment_for_word(self, document_id, n):
        """
        Samples a new level assignment for the nth word in a document.

        It updates the word counts and level assignments accordingly.

        Args:
            document_id (int): The unique identifier for the document.
            n (int): The index of the word in the document to resample.
        """
        doc_words = self.document_words[document_id]
        doc_levels = self.levels[document_id]
        old_level = doc_levels[n]
        w = doc_words[n]

        path_nodes = self.paths[document_id]
        old_node = path_nodes[old_level]
        old_node.word_counts[w] -= 1
        
        # Pruning
        if old_node.word_counts[w] == 0:
            del old_node.word_counts[w] 
        old_node.total_words -= 1

        # Marking word as unassigned by setting its level to -1.
        doc_levels[n] = -1
        
        # Counts the number of words assigned to each level in the document, excluding the word being resampled.
        z_counts = defaultdict(int)
        for lvl in doc_levels:
            if lvl >= 0:
                z_counts[lvl] += 1

        # Determine Maximum Current Level 
        max_z = max(z_counts.keys()) if z_counts else 0
        
        # Prior distribution over levels based on current assignments and hyperparameters
        level_range = list(range(max_z + 1))
        z_counts_list = [z_counts.get(k, 0) for k in level_range]
        prior_probs = self.compute_level_prior_probs(z_counts_list, max_z, self.m, self.pi)

        # Word likelihood for existing levels
        word_likelihoods = []
        for k in level_range:
            node = path_nodes[k]
            w_count = node.word_counts.get(w, 0)
            likelihood = (w_count + self.eta) / (node.total_words + self.eta_sum)
            word_likelihoods.append(likelihood)

        for i in range(len(prior_probs)):
            prior_probs[i] *= word_likelihoods[i]

        # The remaining probability mass, representing the possibility of assigning the word to a deeper level beyond max_z.
        sum_existing = sum(prior_probs)
        leftover = 1.0 - sum_existing
        final_levels = level_range[:]
        final_probs = prior_probs[:]

        # Consider going beyond max_z as per eq (3)
        if leftover > 1e-15:
            ell = max_z + 1
            p_w_new = self.eta / self.eta_sum
            chosen_level = None
            while ell < self.num_levels:
                p_success = (1 - self.m)*p_w_new
                # Bernoulli trial:
                success = (np.random.rand() < p_success)
                if success:
                    chosen_level = ell
                    break
                else:
                    ell += 1

            if chosen_level is None:
                # If we fail all the way, assign the deepest level:
                chosen_level = self.num_levels - 1

            new_level = chosen_level
        else:
            # Just choose from existing levels
            total = sum_existing
            if total == 0:
                # rare fallback
                new_level = max_z
            else:
                probs = [p/total for p in final_probs]
                new_level = np.random.choice(final_levels, p=probs)

        # Update counts
        new_node = path_nodes[new_level]
        new_node.word_counts[w] = new_node.word_counts.get(w,0) + 1
        new_node.total_words += 1
        doc_levels[n] = new_level

    def sample_levels_for_document(self, document_id):
        """
        Samples new level assignments for all words in a document.

        It iterates through each word in the document and resamples its level assignment.

        Args:
            document_id (int): The unique identifier for the document.
        """
        doc_words = self.document_words[document_id]
        for n in range(len(doc_words)):
            self.sample_level_assignment_for_word(document_id, n)

    def gibbs_sampling(self, corpus, num_iterations, burn_in=100, thinning=10):
        """
        Performs Gibbs sampling to infer topic assignments and update the nCRP tree.

        The method iteratively samples path assignments and word level assignments for each document
        in the corpus, updating the tree structure accordingly. It also handles burn-in and thinning
        periods to ensure proper convergence.

        Args:
            corpus (list of list of str): The preprocessed corpus where each document is a list of words.
            num_iterations (int): Total number of Gibbs sampling iterations to perform.
            burn_in (int, optional): Number of initial iterations to discard as burn-in. Defaults to 100.
            thinning (int, optional): Interval for collecting samples (i.e., collect a sample every 'thinning' iterations). Defaults to 10.
        """
        self.initialise_tree(corpus, max_depth=self.num_levels)
        for it in range(num_iterations):
            for doc_id in range(len(corpus)):
                document_words = self.document_words[doc_id]
                document_levels = self.levels[doc_id]
                self.sample_path(doc_id, document_words, document_levels)
                self.sample_levels_for_document(doc_id)
        print("Gibbs sampling completed.")