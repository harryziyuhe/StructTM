import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary
from typing import List, Dict, Optional

def kappa_init(documents, K, V, A, interactions):
    """
    Initialize the kappa parameters.
    
    Parameters
    ----------
    documents : list of np.ndarray
        List of document-term matrices in sparse format.
    K : int
        Number of topics.
    V : int
        Vocabulary size.
    A : int
        Number of aspects.
    interactions : bool
        Whether to include interactions in the model.

    Returns
    -------
    dict
        Initialized kappa parameters.
    """
    # Calculate baseline log-probability (m)
    freq = np.zeros(V)
    for doc in documents:
        for term, count in zip(doc[0], doc[1]):
            freq[term - 1] += count # Adjust 1-indexing to 0-indexing
    freq = freq / freq.sum()
    m = np.log(freq) - np.log(freq.mean())

    aspectmod = A > 1
    interact = interactions and aspectmod

    par_length = K + A * aspectmod + (K * A) * interact
    params = [np.zeros(V) for _ in range(par_length)]

    kappasum = [np.title(m, (K, 1)) for _ in range(A)]

    covar = {
        "k": list(range(1, K + 1)) + ([None] * A if aspectmod else []),
        "a": ([None] * K) + list(range(1, A + 1)) if aspectmod else [],
        "type": [1] * K + ([2] * A if aspectmod else [])
    }

    if interact:
        covar["k"] += [k for k in range(1, K + 1) for _ in range(A)]
        covar["a"] += [a for _ in range(K) for a in range(1, A + 1)]
        covar["type"] += [3] * (K * A)

    return {
        "m": m,
        "params": params,
        "kappasum": kappasum,
        "covar": covar,
    }

def lda_initialization(
        documents: List,
        K: int,
        seed: Optional[int] = None
) -> Dict:
    """
    Initializes parameters using Latent Dirichlet Allocation (LDA).

    Parameters
    ----------
    documents : list of np.ndarray
        List of document-term matrices in sparse format, where each document is represented as a 2D array with two rows: term indices and counts.
    K : int
        Number of topics.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
        -`beta`(np.ndarray): K-by-V matrix of log-probabilities for topic-word distributions.
        -`lambda`(np.ndarray): N-by-K matrix of document-topic proportions.
    """
    # Convert to Gensim format
    gensim_corpus = [
        [(int(term - 1), int(count)) for term, count in zip(doc[0], doc[1])]
        for doc in documents
    ]

    vocab_size = max(term for doc in gensim_corpus for term, _ in doc) + 1
    id2word = Dictionary()
    id2word.token2id = {str(i): i for i in range(vocab_size)}
    id2word.id2token = {i: str(i) for i in range(vocab_size)}

    # Fit LDA model
    lda_model = LdaModel(
        corpus=gensim_corpus,
        num_topics=K,
        id2word=id2word,
        random_state=seed,
        passes=5
    )

    # Extract beta (topic_word distributions)
    beta = np.zeros((K, vocab_size))
    for k in range(K):
        for term_id, prob in lda_model.get_topic_terms(k, topn=vocab_size):
            beta[k, term_id] = prob
    beta = np.log(beta + 1e-12) # Convert to log-space with small stability adjustment

    #Extract lambda (document-topic proportions)
    lambda_matrix = np.zeros((len(documents), K))
    for i, doc in enumerate(gensim_corpus):
        doc_topics = lda_model.get_document_topics(doc, minimum_probability=0)
        lambda_matrix[i, :] = [prob for _, prob in doc_topics]
    
    return {"beta": beta, "lambda": lambda_matrix}

def spectral_initialization(
        documents: csr_matrix,
        K: int,
        settings: Dict
) -> np.ndarray:
    """
    Initializes parameters using the Spectral initialization method.

    Parameters
    ----------
    documents : scipy.sparse.csr_matrix
        Document-term matrix where rows are documents and columns are terms.
    K : int
        Number of topics.
    settings : dict
        Additional settings, such as `maxV` for maximum vocabulary size.

    Returns
    -------
    np.ndarray
        A K-by-V matrix of topic-word distributions (in log-space).
    """
    verbose = settings.get("verbose", False)
    maxV = settings.get("maxV", None)

    # Ensure document-term matrix is dense enough for SVD
    term_doc_matrix = documents.T # Term-by-document matrix
    vocab_size = term_doc_matrix.shape[0]

    if maxV is not None and maxV < vocab_size:
        if verbose:
            print(f"Reducing vocabulary size to {maxV} most frequent terms.")
        term_sums = term_doc_matrix.sum(axis=1).A1
        top_terms = np.argsort(-term_sums)[:maxV]
        term_doc_matrix = term_doc_matrix[top_terms, :]

    # SVD decomposition
    if verbose:
        print("Performing SVD decomposition...")
    u, s, vt = svds(term_doc_matrix, k=K)

    # Normalize rows of U to unit length
    u_norm = u / np.linalg.norm(u, axis=1, keepdims=True)

    # Recover data
    beta = np.abs(u_norm)
    beta = beta / beta.sum(axis=1, keepdims=True)

    # Return in log-space
    return np.log(beta + 1e-12)


def stm_init(documents, settings):
    """
    Initialize parameters for the Structural Topic Model.

    Parameters
    ----------
    documents : list of np.ndarray
        List of document-term matrices in sparse format.
    settings : dict
        Model settings, including dimensions (K, V, A) and initialization options.

    Returns
    -------
    dict
        Initialized model parameters (mu, sigma, beta, lambda, kappa).
    """
    K = settings["dim"]["K"]
    V = settings["dim"]["V"]
    A = settings["dim"]["A"]
    N = settings["dim"]["N"]
    mode = settings["init"]["mode"]

    if mode in ["Random", "Custom"]:
        # Random initialization
        mu = np.zeros((K - 1, 1))
        sigma = np.eye(K - 1) * 20
        beta = np.random.gamma(0.1, 1.0, (K, V))
        beta /= beta.sum(axis=1, keepdims=True)
        lambda_matrix = np.zeros((N, K - 1))
    
    elif mode == "LDA":
        # Initialize using LDA
        lda_result = lda_initialization(documents, K, seed=settings["init"].get("seed"))
        beta = np.exp(lda_result["beta"]) # Convert log probabilities
        mu = np.mean(lda_result["lambda"], axis=0, keepdims=True).T
        sigma = np.cov(lda_result["lambda"].T)
        lambda_matrix = lda_result["lambda"]
    
    elif mode in ["Spectral", "SpectralRP"]:
        # Spectral initialization
        if K >= V:
            raise ValueError("Spectral initialization cannot be used when K >= V.")
        beta = spectral_initialization(documents, K, settings)
        mu = np.zeros((K - 1, 1))
        sigma = np.eye(K - 1) * 20
        lambda_matrix = np.zeros((N, K - 1))

    else:
        raise ValueError(f"Unknown initialization mode: {mode}")
    
    # Handle custom beta initialization
    if mode == "Custom":
        custom_beta = settings["init"]["custom"]
        if not isinstance(custom_beta, list):
            raise ValueError("Custom beta input must be a list.")
        if len(custom_beta) != A:
            raise ValueError("Custom beta list length does not match the number of aspects.")
        if custom_beta[0].shape != beta.shape:
            raise ValueError("Dimensions of custom beta do not match the model specification.")
        beta = [np.exp(beta_matrix) for beta_matrix in custom_beta]

    # Initialize kappa parameters
    kappa = None
    if not settings["kappa"]["LDAbeta"]:
        kappa = kappa_init(documents, K, V, A, settings["kappa"]["interactions"])
    
    return {
        "mu": mu,
        "sigma": sigma,
        "beta": beta if A == 1 else [beta] * A,
        "lambda": lambda_matrix,
        "kappa": kappa,
    }