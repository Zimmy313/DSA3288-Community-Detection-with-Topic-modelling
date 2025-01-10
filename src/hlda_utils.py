import nltk
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import defaultdict

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def preprocess_text(text, 
                   stop_words=None,
                   stemmer=None,
                   lemmatizer=None,
                   min_word_length=2):
    """
    Preprocesses the input text by:
    1. Lowercasing
    2. Tokenizing
    3. Removing non-alphabetic tokens and stopwords
    4. Applying stemming and lemmatization
    5. Filtering out very short tokens
    """
    if stop_words is None:
        stop_words = set(stopwords.words('english'))
    if stemmer is None:
        stemmer = PorterStemmer()
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Tokenize
    tokens = word_tokenize(text)
    
    # 3. Remove non-alphabetic tokens and stopwords
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words and len(t) >= min_word_length]
    
    # 4. Apply Stemming and Lemmatization
    stemmed = [stemmer.stem(t) for t in tokens]
    lemmatized = [lemmatizer.lemmatize(t) for t in stemmed]
    
    return lemmatized

def build_vocabulary(docs, min_freq=5):
    """
    Builds a vocabulary dictionary mapping words to unique indices.
    Words with frequency less than min_freq are excluded.
    
    Parameters:
    - docs: List of documents, where each document is a list of words.
    - min_freq: Minimum frequency a word must have to be included in the vocabulary.
    
    Returns:
    - vocab: Sorted list of vocabulary words.
    - word2idx: Dictionary mapping words to their unique indices.
    - idx2word: Dictionary mapping indices to their corresponding words.
    """
    word_freq = defaultdict(int)
    for doc in docs:
        for word in doc:
            word_freq[word] += 1
    vocab = sorted([word for word, freq in word_freq.items() if freq >= min_freq])
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    return vocab, word2idx, idx2word

def convert_docs_to_indices(docs, word2idx):
    """
    Converts a list of documents (list of words) to a corpus of word indices.
    Words not in the vocabulary are ignored.
    
    Parameters:
    - docs: List of documents, where each document is a list of words.
    - word2idx: Dictionary mapping words to their unique indices.
    
    Returns:
    - corpus: List of documents represented as lists of word indices.
    """
    corpus = []
    for doc in docs:
        indexed_doc = [word2idx[word] for word in doc if word in word2idx]
        corpus.append(indexed_doc)
    return corpus


def preprocess_and_filter_empty_with_labels(docs, labels, stop_words=None, stemmer=None, lemmatizer=None, min_word_length=2):
    """
    Preprocesses a list of documents, filters out empty documents, and returns the 
    filtered documents and corresponding labels.
    """
    # Apply preprocessing to each document
    processed_docs = [preprocess_text(doc, stop_words, stemmer, lemmatizer, min_word_length) for doc in docs]
    
    # Filter out empty documents and their corresponding labels
    filtered_docs_labels = [(doc, label) for doc, label in zip(processed_docs, labels) if doc]
    
    # Separate documents and labels
    filtered_docs, filtered_labels = zip(*filtered_docs_labels) if filtered_docs_labels else ([], [])
    
    return list(filtered_docs), list(filtered_labels)


# Pipeline Function
def full_preprocessing_pipeline(docs, labels, stop_words=None, stemmer=None, lemmatizer=None, min_word_length=2, min_freq=5):
    """
    This function integrates the entire preprocessing pipeline:
    - Preprocess the documents
    - Filter out empty documents and corresponding labels
    - Build vocabulary
    - Convert documents to word indices
    
    It prints intermediate results at each step.
    """
    # Step 1: Preprocess and filter empty documents
    filtered_docs, filtered_labels = preprocess_and_filter_empty_with_labels(docs, labels, stop_words, stemmer, lemmatizer, min_word_length)
    print(f"Number of documents after filtering empty ones: {len(filtered_docs)}")
    print(f"Number of labels after filtering: {len(filtered_labels)}")
    
    # Step 2: Build vocabulary
    vocab, word2idx, idx2word = build_vocabulary(filtered_docs, min_freq)
    print("")
    print(f"Vocabulary size (words with freq >= {min_freq}): {len(vocab)}")
    print(f"Sample vocabulary words: {vocab[:10]}")
    
    # Step 3: Convert documents to indices
    corpus = convert_docs_to_indices(filtered_docs, word2idx)
    print("Preprecessing done")
    return filtered_docs, filtered_labels, vocab, word2idx, idx2word, corpus


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
        synthetic_corpus[i] is a synthetic document (list of word indices).
    doc_paths : list of list of tuples
        doc_paths[i] is the list of (node_index, level_id) along the path used to 
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

        # Fix rounding
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
            freqs = node.word_freqs
            total_freq = float(node.word_total)

            if total_freq <= 0:
                # Fallback to uniform distribution
                probs = np.ones(len(freqs)) / len(freqs)
            else:
                probs = freqs / total_freq

            # Sample word indices based on probabilities
            word_ids = np.arange(len(freqs))
            chosen = rng.choice(word_ids, size=level_counts[lvl], replace=True, p=probs)
            doc_tokens.extend(chosen.tolist())  # Ensure tokens are integers

        synthetic_corpus.append(doc_tokens)
        # Store path as list of tuples (node_index, level_id)
        doc_paths.append([(node.node_index, node.level_id) for node in path_nodes])

    return synthetic_corpus, doc_paths