import nltk
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


