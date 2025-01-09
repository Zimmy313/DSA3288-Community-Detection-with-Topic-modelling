import numpy as np
from collections import defaultdict
from math import log
from functools import lru_cache
from scipy.special import gammaln

class Node:
    """
    Represents a node in a hierarchical topic tree.
    Each node has a unique ID and we track the total number of active nodes.
    """

    # Class-level counters
    total_nodes = 0
    last_node_id = 0

    def __init__(self, parent=None, level=0):
        Node.last_node_id += 1
        self.node_id = Node.last_node_id
        Node.total_nodes += 1

        self.children = {}
        self.documents = 0
        self.word_counts = defaultdict(int)
        self.total_words = 0
        self.parent = parent
        self.level = level

    def __del__(self):
        Node.total_nodes -= 1

    def is_leaf(self):
        return len(self.children) == 0

    def add_child(self, topic_id):
        child_node = Node(parent=self, level=self.level + 1)
        self.children[topic_id] = child_node
        return child_node

    def remove_child(self, topic_id):
        if topic_id in self.children:
            child = self.children[topic_id]
            del self.children[topic_id]
        else:
            raise KeyError(f"Topic ID {topic_id} not found among the children.")

class nCRPTree:
    """
    A nested Chinese Restaurant Process tree for hierarchical LDA.
    """

    def __init__(self, corpus, gamma, eta, max_level, vocab, m=0.5, pi=1.0):
        """
        Args:
            corpus (list of list of str): The preprocessed corpus (each doc is a list of words).
            gamma (float): nCRP concentration parameter.
            eta (float): Smoothing for topic-word distributions.
            max_level (int): Max depth (levels in [0..max_level-1]).
            vocab (list[str]): List of vocabulary tokens.
            m, pi (float): Hyperparams for level priors.
        """
        self.root = Node()
        self.gamma = gamma
        self.eta = eta
        self.V = len(vocab)
        self.vocab = vocab
        self.eta_sum = self.eta * self.V
        self.m = m
        self.pi = pi

        self.max_level = max_level

        self.paths = {}           # doc_id -> list[Node]
        self.levels = {}          # doc_id -> list[int]
        self.document_words = {}  # doc_id -> list[str]

        for doc_id, doc_words in enumerate(corpus):
            self.document_words[doc_id] = doc_words
            path_nodes = self.initialize_new_path(doc_id)
            doc_levels = []
            for w in doc_words:
                level = np.random.randint(0, len(path_nodes))
                doc_levels.append(level)
                node = path_nodes[level]
                node.word_counts[w] += 1
                node.total_words += 1

            self.levels[doc_id] = doc_levels

    @lru_cache(maxsize=None)
    def cached_gammaln(self, x):
        return gammaln(x)

    def initialize_new_path(self, document_id):
        current_node = self.root
        current_node.documents += 1
        path_nodes = [current_node]

        for level in range(1, self.max_level):
            topic_id, is_new = self.sample_ncrp_process(current_node)
            if is_new:
                child_node = current_node.add_child(topic_id)
            else:
                child_node = current_node.children[topic_id]
            child_node.documents += 1
            path_nodes.append(child_node)
            current_node = child_node

        self.paths[document_id] = path_nodes
        return path_nodes

    def sample_ncrp_process(self, node):
        total_customers = node.documents
        topic_probabilities = {}

        # existing children
        for topic_id, child in node.children.items():
            topic_probabilities[topic_id] = child.documents / (total_customers + self.gamma)
        # new child
        new_topic_key = max(node.children.keys(), default=0) + 1
        topic_probabilities[new_topic_key] = self.gamma / (total_customers + self.gamma)

        topics = list(topic_probabilities.keys())
        probs = np.array(list(topic_probabilities.values()))
        probs /= probs.sum()

        chosen = np.random.choice(topics, p=probs)
        is_new = (chosen not in node.children)
        return chosen, is_new

    # --------------------------------------------------------------------------
    # Document add/remove
    # --------------------------------------------------------------------------
    def add_document(self, document_id, path_nodes, level_word_counts, doc_words):
        """
        Adds a document to the tree. We require doc_words
        so we can store them in self.document_words.
        """
        self.paths[document_id] = path_nodes
        self.document_words[document_id] = doc_words  # ensure doc_id is in .document_words

        for node in path_nodes:
            node.documents += 1

        for lvl, w_counts in level_word_counts.items():
            node = path_nodes[lvl]
            for w, count in w_counts.items():
                node.word_counts[w] += count
                node.total_words += count

    def remove_document(self, document_id):
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

        # Decrement doc counts
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

    # --------------------------------------------------------------------------
    # Path Sampling
    # --------------------------------------------------------------------------
    def get_level_word_counts(self, document_words, document_levels):
        level_word_counts = {}
        for w, lvl in zip(document_words, document_levels):
            level_word_counts.setdefault(lvl, {})
            level_word_counts[lvl][w] = level_word_counts[lvl].get(w, 0) + 1
        return level_word_counts

    def level_likelihood(self, node, M):
        Nminus = node.word_counts
        sumNminus = node.total_words
        sumM = sum(M.values())
        sumN = sumNminus + sumM

        lp1 = self.cached_gammaln(sumNminus + self.eta_sum)
        for wc in Nminus.values():
            lp1 -= self.cached_gammaln(wc + self.eta)

        lp2 = 0.0
        for w, old_count in Nminus.items():
            new_count = old_count + M.get(w, 0)
            lp2 += self.cached_gammaln(new_count + self.eta)
        for w, cnt in M.items():
            if w not in Nminus:
                lp2 += self.cached_gammaln(cnt + self.eta)

        lp2 -= self.cached_gammaln(sumN + self.eta_sum)
        return lp1 + lp2

    def level_prior(self, parent, child_is_new, child_node=None):
        total_customers = parent.documents
        if child_is_new:
            return log(self.gamma) - log(total_customers + self.gamma)
        else:
            return log(child_node.documents) - log(total_customers + self.gamma)

    def sample_path_level(self, parent_node, level_word_counts, level_index):
        candidates = list(parent_node.children.items())
        new_topic_id = max(parent_node.children.keys(), default=0) + 1

        M = level_word_counts.get(level_index, {})

        log_probs = []
        # existing children
        for tid, child in candidates:
            lp = self.level_prior(parent_node, False, child)
            lp += self.level_likelihood(child, M)
            log_probs.append((lp, tid, False))

        # new child
        fake_node = Node(parent=parent_node, level=parent_node.level + 1)
        lp_new = self.level_prior(parent_node, True)
        lp_new += self.level_likelihood(fake_node, M)
        log_probs.append((lp_new, new_topic_id, True))

        max_lp = max(lp for lp, _, _ in log_probs)
        weights = np.exp([lp - max_lp for lp, _, _ in log_probs])
        probs = weights / weights.sum()

        chosen_index = np.random.choice(len(probs), p=probs)
        chosen_lp, chosen_tid, chosen_is_new = log_probs[chosen_index]

        if chosen_is_new:
            child_node = parent_node.add_child(chosen_tid)
        else:
            child_node = parent_node.children[chosen_tid]
        return child_node

    def sample_path(self, document_id, document_words, document_levels):
        """
        Re-sample path for an existing doc.
        """
        if document_id in self.paths:
            self.remove_document(document_id)

        lvl_word_counts = self.get_level_word_counts(document_words, document_levels)
        current = self.root
        path_nodes = [current]

        for ell in range(1, self.max_level):
            child_node = self.sample_path_level(current, lvl_word_counts, ell)
            path_nodes.append(child_node)
            current = child_node

        # Re-add the doc
        self.add_document(document_id, path_nodes, lvl_word_counts, doc_words=document_words)
        self.levels[document_id] = document_levels
        # doc_words already stored in add_document

        max_l_used = max(document_levels) if document_levels else 0
        assert max_l_used < self.max_level, "Document level assignments exceed max_level."

    # --------------------------------------------------------------------------
    # Level Sampling
    # --------------------------------------------------------------------------
    def compute_level_prior_probs(self, z_counts, max_z, m, pi):
        sum_ge = [0]*(max_z+1)
        sum_gt = [0]*(max_z+1)

        running = 0
        for i in reversed(range(max_z+1)):
            running += z_counts[i]
            sum_ge[i] = running

        for i in range(max_z):
            sum_gt[i] = sum_ge[i+1]

        probs = []
        for k in range(max_z+1):
            # main factor
            numerator_main = (m*pi + z_counts[k])
            denominator_main = (pi + sum_ge[k]) if sum_ge[k] > 0 else pi
            lvl_prob = numerator_main / denominator_main

            # product
            for j in range(1, k):
                numer_j = ((1 - m)*pi + sum_gt[j])
                denom_j = (pi + sum_ge[j]) if sum_ge[j] > 0 else pi
                lvl_prob *= (numer_j / denom_j)

            probs.append(lvl_prob)
        return probs

    def sample_level_assignment_for_word(self, document_id, n):
        doc_words = self.document_words[document_id]
        doc_levels = self.levels[document_id]
        w = doc_words[n]
        old_lvl = doc_levels[n]

        path_nodes = self.paths[document_id]
        old_node = path_nodes[old_lvl]
        old_node.word_counts[w] -= 1
        if old_node.word_counts[w] == 0:
            del old_node.word_counts[w]
        old_node.total_words -= 1

        doc_levels[n] = -1

        z_counts_dict = defaultdict(int)
        for lvl in doc_levels:
            if lvl >= 0:
                z_counts_dict[lvl] += 1

        max_z = max(z_counts_dict.keys()) if z_counts_dict else 0
        if max_z >= self.max_level:
            max_z = self.max_level - 1

        z_counts = [z_counts_dict.get(k,0) for k in range(max_z+1)]
        prior_probs = self.compute_level_prior_probs(z_counts, max_z, self.m, self.pi)

        word_likelihoods = []
        for k in range(max_z+1):
            node = path_nodes[k]
            w_count = node.word_counts.get(w, 0)
            likelihood = (w_count + self.eta)/(node.total_words + self.eta_sum)
            word_likelihoods.append(likelihood)

        # multiply prior & likelihood
        for i in range(len(prior_probs)):
            prior_probs[i] *= word_likelihoods[i]

        sum_existing = sum(prior_probs)
        leftover = 1.0 - sum_existing

        if leftover > 1e-15:
            ell = max_z + 1
            chosen_lvl = None
            p_w_new = self.eta / self.eta_sum
            while ell < self.max_level:
                p_succ = (1-self.m)*p_w_new
                if np.random.rand() < p_succ:
                    chosen_lvl = ell
                    break
                else:
                    ell += 1
            if chosen_lvl is None:
                chosen_lvl = self.max_level - 1
        else:
            if sum_existing < 1e-30:
                chosen_lvl = max_z
            else:
                norm_probs = [p/sum_existing for p in prior_probs]
                chosen_lvl = np.random.choice(range(max_z+1), p=norm_probs)

        new_node = path_nodes[chosen_lvl]
        new_node.word_counts[w] = new_node.word_counts.get(w,0) + 1
        new_node.total_words += 1
        doc_levels[n] = chosen_lvl

    def sample_levels_for_document(self, document_id):
        doc_words = self.document_words[document_id]
        for n in range(len(doc_words)):
            self.sample_level_assignment_for_word(document_id, n)

    def gibbs_sampling(self, num_iterations):
        for _ in range(num_iterations):
            for doc_id in list(self.paths.keys()):
                dw = self.document_words[doc_id]
                dl = self.levels[doc_id]
                self.sample_path(doc_id, dw, dl)
                self.sample_levels_for_document(doc_id)

        print("Gibbs sampling completed.")
        print(f"Total nodes created ever: {Node.last_node_id}")
        print(f"Active nodes in the tree now: {Node.total_nodes}")