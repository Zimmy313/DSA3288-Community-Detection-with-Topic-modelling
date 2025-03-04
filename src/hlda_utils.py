import nltk
import numpy as np
import pandas as pd

from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import defaultdict
from hlda_final import NCRPNode

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

def preprocess_text(text, stemmer=None, lemmatizer=None, min_word_length=2):
    """
    Preprocesses the input text by:
    1. Lowercasing
    2. Tokenizing
    3. Removing non-alphabetic tokens
    4. Applying stemming and lemmatization
    5. Filtering out very short tokens
    """
    if stemmer is None:
        stemmer = PorterStemmer()
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Tokenize
    tokens = word_tokenize(text)
    
    # 3. Remove non-alphabetic tokens
    tokens = [t for t in tokens if t.isalpha() and len(t) >= min_word_length]
    
    # 4. Apply Stemming and Lemmatization
    stemmed = [stemmer.stem(t) for t in tokens]
    lemmatized = [lemmatizer.lemmatize(t) for t in stemmed]
    
    return lemmatized

def preprocess_and_filter_empty_with_labels(docs, labels, stemmer=None, lemmatizer=None, min_word_length=2):
    """
    Preprocesses a list of documents, filters out empty documents, and returns the 
    filtered documents and corresponding labels.
    """
    processed_docs = [preprocess_text(doc, stemmer, lemmatizer, min_word_length) for doc in docs]
    
    filtered_docs_labels = [(doc, label) for doc, label in zip(processed_docs, labels) if doc]
    
    filtered_docs, filtered_labels = zip(*filtered_docs_labels) if filtered_docs_labels else ([], [])
    
    return list(filtered_docs), list(filtered_labels)

def build_vocabulary(docs, min_freq=5):
    """
    Builds a vocabulary dictionary mapping words to unique indices.
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
    Converts a list of documents to word indices.
    """
    corpus = [[word2idx[word] for word in doc if word in word2idx] for doc in docs]
    return corpus

def full_preprocessing_pipeline(docs, labels, stemmer=None, lemmatizer=None, min_word_length=2, min_freq=5):
    """
    Full text preprocessing pipeline.
    """
    filtered_docs, filtered_labels = preprocess_and_filter_empty_with_labels(docs, labels, stemmer, lemmatizer, min_word_length)
    print(f"Documents after filtering: {len(filtered_docs)}")
    
    vocab, word2idx, idx2word = build_vocabulary(filtered_docs, min_freq)
    print(f"Vocabulary size: {len(vocab)}")
    
    corpus = convert_docs_to_indices(filtered_docs, word2idx)
    print("Preprocessing complete.")
    
    return filtered_docs, filtered_labels, vocab, word2idx, idx2word, corpus

def summarise(df, hlda_model, model_name, aim, documents, time):
    """
    Function to extract the required information and save it into the dataset
    """
    gamma_result = hlda_model.gamma_eval()
    eta_result = hlda_model.eta_eval()
    eta_result = [round(val, 2) for val in eta_result]
    alpha_results = []
    
    for doc_id in documents:
        alpha_eval = hlda_model.alpha_eval(doc_id)
        alpha_results.append([round(val, 2) for val in alpha_eval])
    
    data = {
        'Model': model_name,
        'aim' : aim,
        'alpha': hlda_model.alpha,
        'gamma': hlda_model.gamma,
        'eta': hlda_model.eta,
        'time': time,
        'total_table': NCRPNode.total_created_nodes,
        'gamma_eval': gamma_result,
        'eta_eval': eta_result,
        'alpha_eval': alpha_results
    }
    print(data)
    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    return df


def jensen_shannon_divergence(model1, topic_index1, model2, topic_index2, eps=1e-12):
    """
    Calculate the Jensen-Shannon divergence between two topic distributions,
    one from each model. The topic distributions are obtained from the word 
    count vector (n_w) of each node (topic) and normalized to form a probability 
    distribution.
    
    Parameters
    ----------
    model1 : hLDA
        The first hLDA model.
    topic_index1 : int
        The topic number (node index) to compare from model1.
    model2 : hLDA
        The second hLDA model.
    topic_index2 : int
        The topic number (node index) to compare from model2.
    eps : float, optional
        A small constant added to denominators to avoid division by zero.
    
    Returns
    -------
    float
        The Jensen-Shannon divergence between the two topic distributions.
    """
    
    def get_node_by_index(model, topic_index):
        for node in model.traverse_tree():
            if node.node_index == topic_index:
                return node
        return None
    
    node1 = get_node_by_index(model1, topic_index1)
    node2 = get_node_by_index(model2, topic_index2)
    
    if node1 is None:
        raise ValueError(f"Topic {topic_index1} not found in model1.")
    if node2 is None:
        raise ValueError(f"Topic {topic_index2} not found in model2.")
    
    # Convert raw counts to probability distributions.
    p1 = node1.n_w.astype(np.float64)
    p2 = node2.n_w.astype(np.float64)
    
    # Normalize counts to probabilities; if total count is 0, use a uniform distribution.
    if node1.n_sum > 0:
        p1 /= node1.n_sum
    else:
        p1 = np.ones_like(p1) / len(p1)
        
    if node2.n_sum > 0:
        p2 /= node2.n_sum
    else:
        p2 = np.ones_like(p2) / len(p2)
        
    # Compute the average distribution.
    m = 0.5 * (p1 + p2)
    
    # Safe KL divergence: only consider indices where the first distribution is nonzero,
    # and add a small epsilon to avoid log(0) and division-by-zero.
    def kl_divergence(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log(a[mask] / (b[mask] + eps)))
    
    jsd = 0.5 * kl_divergence(p1, m) + 0.5 * kl_divergence(p2, m)
    return jsd